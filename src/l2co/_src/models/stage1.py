"""Stage-1 optimizer-selection models (trajectory-aware reassessment)."""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from l2co_optimizers import OptimizationStep

from ..optimization_dynamics import OptimizationDynamics
from ..task_features import FeatureProcessor, MixedEmbedding
from ..tree_utils import traj_and_stair
from .base import AbstractModel
from .components import MLP, LSTM_Encoder
from .training import train_equinox

# =============================================================================


class Stage1Model(AbstractModel):
    """Abstract base class for stage-1 optimizer selection.

    Stage-1 models predict optimizer logits using task features,
    optimizer features, and optimization history (trajectory
    data), enabling dynamic optimizer switching during a run.

    Attributes
    ----------
    mixed_embedding_task : MixedEmbedding
        Embedding layer for mixed task features.
    mixed_embedding_optimizer : MixedEmbedding
        Embedding layer for mixed optimizer features.
    feature_processor_task : FeatureProcessor
        Processor for raw task features.
    feature_processor_optimizer : FeatureProcessor
        Processor for raw optimizer features.
    history_encoder : eqx.Module
        Encoder for optimization trajectory history.
    model : eqx.Module
        Classification head (e.g., MLP).
    optimizers : Iterable[OptimizationStep]
        Available optimizer configurations.
    """

    mixed_embedding_task: MixedEmbedding
    mixed_embedding_optimizer: MixedEmbedding
    feature_processor_task: FeatureProcessor
    feature_processor_optimizer: FeatureProcessor
    history_encoder: eqx.Module
    model: eqx.Module
    optimizers: Iterable[OptimizationStep]

    @classmethod
    @abstractmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        feature_processor_optimizer: FeatureProcessor,
        mixed_embedding_task: MixedEmbedding,
        mixed_embedding_optimizer: MixedEmbedding,
        history_encoder: eqx.Module,
        model: eqx.Module,
        optimizers: Iterable[OptimizationStep],
    ) -> Stage1Model:
        """Initialize a Stage1Model instance.

        Parameters
        ----------
        feature_processor_task : FeatureProcessor
            Feature processor for task features.
        feature_processor_optimizer : FeatureProcessor
            Feature processor for optimizer features.
        mixed_embedding_task : MixedEmbedding
            Mixed embedding for task features.
        mixed_embedding_optimizer : MixedEmbedding
            Mixed embedding for optimizer features.
        history_encoder : eqx.Module
            Encoder for optimization history.
        model : eqx.Module
            Underlying model (e.g., MLP).
        optimizers : Iterable[OptimizationStep]
            List of possible optimization steps.

        Returns
        -------
        Stage1Model
            Initialized Stage1Model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        """Forward pass returning logits."""
        raise NotImplementedError

    @abstractmethod
    def pred_logits(self, data: OptimizationDynamics) -> xr.DataArray:
        """Predict raw optimizer logits for all samples.

        Parameters
        ----------
        data : OptimizationDynamics
            Dataset with task features, optimizer features,
            history, and labels.

        Returns
        -------
        xr.DataArray
            Predicted logits with sample and label coordinates.
        """
        raise NotImplementedError

    def predict(
        self,
        x_feature_task: dict[str, Any],
        x_feature_optimizer: dict[str, Any],
        x_output_mean: jax.Array,
        x_output_min: jax.Array,
        global_min: jax.Array,
    ) -> OptimizationStep:
        """Select the best optimizer given features and history.

        Transforms raw features, encodes the optimization
        trajectory, and returns the optimizer with the highest
        predicted logit.

        Parameters
        ----------
        x_feature_task : dict[str, Any]
            Raw task feature dictionary.
        x_feature_optimizer : dict[str, Any]
            Raw optimizer feature dictionary.
        x_output_mean : jax.Array
            Mean trajectory values per budget step.
        x_output_min : jax.Array
            Minimum trajectory values per budget step.
        global_min : jax.Array
            Global minimum value observed so far.

        Returns
        -------
        OptimizationStep
            The optimizer with the highest predicted logit.
        """
        x = (
            self.feature_processor_task.transform(x_feature_task),
            self.feature_processor_optimizer.transform(x_feature_optimizer),
            traj_and_stair(x_output_mean, x_output_min, global_min),
        )

        logits = self(x)
        action: jax.Array = jnp.argmax(logits)
        optimizer = self.optimizers[action]
        return optimizer

    def evaluate(self, data: OptimizationDynamics) -> xr.DataArray:
        """Shared logic for mapping logits to OptimizationSteps."""
        logits_xr = self.pred_logits(data)
        actions = jnp.argmax(logits_xr.values, axis=-1)

        # Map indices to the actual OptimizationStep objects
        step_results = [data.y["label"].values[int(k)] for k in actions]

        return xr.DataArray(
            np.array(step_results),
            coords={"sample": data.y["sample"].coords["sample"]},
            dims=["sample"],
        )

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> Stage1Model:
        """Replace the optimizer set and update the processor.

        Updates the optimizer feature processor name mapping
        and returns a new model with the given optimizers.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        Stage1Model
            New model instance with updated optimizers.
        """
        self.feature_processor_optimizer = (
            self.feature_processor_optimizer.swap_names(
                {opt.name: opt.name for opt in optimizers}
            )
        )
        return Stage1Model(
            mixed_embedding_optimizer=self.mixed_embedding_optimizer,
            mixed_embedding_task=self.mixed_embedding_task,
            feature_processor_task=self.feature_processor_task,
            feature_processor_optimizer=self.feature_processor_optimizer,
            history_encoder=self.history_encoder,
            optimizers=optimizers,
        )

    def with_feature_impute(self, impute: dict[str, Any]) -> Stage1Model:
        """Fill task-tag features the checkpoint's schema expects but a
        foreign task set does not carry.

        Generic over the concrete subclass: replaces only
        ``feature_processor_task`` via :func:`equinox.tree_at`, so
        every other field -- weights included -- is carried over
        unchanged. Only the *task* processor is touched;
        ``feature_processor_optimizer`` encodes the optimizer name,
        which every task set supplies the same way, so it never needs
        imputation. Delegates to
        :meth:`FeatureProcessor.with_imputation`, which leaves
        vocabularies and feature count untouched, so the loaded
        weights keep matching.

        Parameters
        ----------
        impute : dict[str, Any]
            Feature name to fill value, forwarded to
            ``FeatureProcessor.with_imputation``.

        Returns
        -------
        Stage1Model
            New model instance (same concrete class) with the filled
            task processor.
        """
        return eqx.tree_at(
            lambda m: m.feature_processor_task,
            self,
            self.feature_processor_task.with_imputation(impute),
        )


class EquinoxStage1Model(Stage1Model):
    """Equinox-based stage-1 model with LSTM history encoder.

    Encodes optimization history with an LSTM, embeds task and
    optimizer features via ``MixedEmbedding`` layers, and
    concatenates all representations for an MLP classification
    head that outputs per-optimizer logits.

    Attributes
    ----------
    history_encoder : LSTM_Encoder
        LSTM encoder for optimization trajectory.
    model : MLP
        MLP classification head.
    """

    mixed_embedding_task: MixedEmbedding
    mixed_embedding_optimizer: MixedEmbedding
    feature_processor_task: FeatureProcessor
    feature_processor_optimizer: FeatureProcessor
    history_encoder: LSTM_Encoder
    model: MLP
    optimizers: Iterable[OptimizationStep]

    @classmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        feature_processor_optimizer: FeatureProcessor,
        mixed_embedding_task: MixedEmbedding,
        mixed_embedding_optimizer: MixedEmbedding,
        history_encoder: LSTM_Encoder,
        model: MLP,
        optimizers: Iterable[OptimizationStep],
    ) -> EquinoxStage1Model:
        """Create an EquinoxStage1Model from its components.

        Parameters
        ----------
        feature_processor_task : FeatureProcessor
            Processor for task features.
        feature_processor_optimizer : FeatureProcessor
            Processor for optimizer features.
        mixed_embedding_task : MixedEmbedding
            Embedding layer for task features.
        mixed_embedding_optimizer : MixedEmbedding
            Embedding layer for optimizer features.
        history_encoder : LSTM_Encoder
            LSTM encoder for optimization trajectory.
        model : MLP
            MLP classification head.
        optimizers : Iterable[OptimizationStep]
            Available optimizer configurations.

        Returns
        -------
        EquinoxStage1Model
            Initialized model instance.
        """
        return cls(
            feature_processor_task=feature_processor_task,
            feature_processor_optimizer=feature_processor_optimizer,
            mixed_embedding_task=mixed_embedding_task,
            mixed_embedding_optimizer=mixed_embedding_optimizer,
            history_encoder=history_encoder,
            model=model,
            optimizers=optimizers,
        )

    def __call__(self, x: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        x_features_task, x_features_optimizer, x_history = x
        history_emb = self.history_encoder(x_history)
        features_task_emb = self.mixed_embedding_task(x_features_task)
        features_optimizer_emb = self.mixed_embedding_optimizer(
            x_features_optimizer
        )
        combined = jnp.concatenate(
            [features_task_emb, features_optimizer_emb, history_emb], axis=-1
        )
        return self.model(combined)

    def train(
        self,
        data: OptimizationDynamics,
        optimizer: optax.GradientTransformation,
        num_steps: int = 100,
        verbose: bool = False,
        scan: bool = True,
    ) -> tuple[EquinoxStage1Model, dict]:
        """Train the stage-1 model with MSE loss.

        Splits data 80/20 by problem to prevent leakage,
        then trains with ``train_equinox``.

        Parameters
        ----------
        data : OptimizationDynamics
            Training dataset (filtered internally).
        optimizer : optax.GradientTransformation
            Optax optimizer instance.
        num_steps : int, optional
            Number of gradient steps. Default is 100.
        verbose : bool, optional
            Show progress bar. Default is False.
        scan : bool, optional
            Use ``jax.lax.scan`` for training loop.
            Default is True.

        Returns
        -------
        tuple[EquinoxStage1Model, dict]
            Trained model and metrics dict with
            ``train_loss`` and ``val_loss`` arrays.
        """
        # Split by problem to prevent leakage
        data_train, data_val = data.filter().train_test_split(
            train_fraction=0.8, seed=42
        )

        x_train = (
            jnp.array(data_train.x_features_task),
            jnp.array(data_train.x_features_optimizer),
            jnp.array(data_train.x_history),
        )

        x_val = (
            jnp.array(data_val.x_features_task),
            jnp.array(data_val.x_features_optimizer),
            jnp.array(data_val.x_history),
        )

        y_train = jnp.array(data_train.y)
        y_val = jnp.array(data_val.y)

        model, metrics = train_equinox(
            model=self,
            optimizer=optimizer,
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            num_steps=num_steps,
            verbose=verbose,
            scan=scan,
        )
        return model, metrics

    def pred_logits(self, data: OptimizationDynamics) -> xr.DataArray:
        """Predict optimizer logits for all samples.

        Vectorizes the forward pass over samples using
        ``eqx.filter_vmap``.

        Parameters
        ----------
        data : OptimizationDynamics
            Dataset with task features, optimizer features,
            history, and labels.

        Returns
        -------
        xr.DataArray
            Predicted logits aligned to data coordinates.
        """
        x = (
            jnp.array(data.x_features_task),
            jnp.array(data.x_features_optimizer),
            jnp.array(data.x_history),
        )
        logits = eqx.filter_vmap(self)(x)
        y = xr.DataArray(
            logits,
            coords=data.y.coords,
            dims=data.y.dims,
        )
        return y

    def evaluate(
        self,
        data: OptimizationDynamics,
    ) -> xr.DataArray:
        """Evaluate the model and return predicted optimizers.

        Parameters
        ----------
        data : OptimizationDynamics
            Evaluation dataset.

        Returns
        -------
        xr.DataArray
            Array of optimizer names indexed by sample.
        """
        x_eval = (
            jnp.array(data.x_features_task),
            jnp.array(data.x_features_optimizer),
            jnp.array(data.x_history),
        )

        logits = eqx.filter_vmap(self)(x_eval)
        actions: jax.Array = jnp.argmax(logits, axis=-1)

        optimizers = [data.y["label"].values[int(k)] for k in actions]

        y = xr.DataArray(
            np.array(optimizers),
            coords={"sample": data.y["sample"].coords["sample"]},
            dims=["sample"],
        )

        return y

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all component configs:
            feature processors, embeddings, encoder, MLP,
            and optimizer list.
        """
        return {
            "feature_processor_task": self.feature_processor_task.to_dict(),
            "feature_processor_optimizer": self.feature_processor_optimizer.to_dict(),  # noqa
            "mixed_embedding_task": self.mixed_embedding_task.to_dict(),
            "mixed_embedding_optimizer": self.mixed_embedding_optimizer.to_dict(),  # noqa
            "history_encoder": self.history_encoder.to_dict(),
            "model": self.model.to_dict(),
            "optimizers": [step.to_dict() for step in self.optimizers],
        }

    @classmethod
    def reconstruct(
        cls,
        feature_processor_task: dict,
        feature_processor_optimizer: dict,
        mixed_embedding_task: dict,
        mixed_embedding_optimizer: dict,
        history_encoder: dict,
        model: dict,
        optimizers: Iterable[dict],
    ) -> EquinoxStage1Model:
        """Reconstruct an EquinoxStage1Model from serialized dicts.

        Each component is reconstructed from its own dict via
        the corresponding ``reconstruct`` classmethod.

        Parameters
        ----------
        feature_processor_task : dict
            Serialized task feature processor.
        feature_processor_optimizer : dict
            Serialized optimizer feature processor.
        mixed_embedding_task : dict
            Serialized task embedding config.
        mixed_embedding_optimizer : dict
            Serialized optimizer embedding config.
        history_encoder : dict
            Serialized LSTM encoder config.
        model : dict
            Serialized MLP config.
        optimizers : Iterable[dict]
            Serialized optimizer step dicts.

        Returns
        -------
        EquinoxStage1Model
            Reconstructed model (weights are placeholder
            until ``eqx.tree_deserialise_leaves`` is called).
        """
        feature_processor_task = FeatureProcessor.reconstruct(
            **feature_processor_task
        )

        feature_processor_optimizer = FeatureProcessor.reconstruct(
            **feature_processor_optimizer
        )

        mixed_embedding_task = MixedEmbedding.reconstruct(
            **mixed_embedding_task
        )
        mixed_embedding_optimizer = MixedEmbedding.reconstruct(
            **mixed_embedding_optimizer
        )

        history_encoder = LSTM_Encoder.reconstruct(**history_encoder)

        model = MLP.reconstruct(**model)

        optimizers = [OptimizationStep.from_dict(step) for step in optimizers]

        return cls(
            feature_processor_task=feature_processor_task,
            feature_processor_optimizer=feature_processor_optimizer,
            mixed_embedding_task=mixed_embedding_task,
            mixed_embedding_optimizer=mixed_embedding_optimizer,
            history_encoder=history_encoder,
            model=model,
            optimizers=optimizers,
        )

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> EquinoxStage1Model:
        """Replace the optimizer set and update the processor.

        Updates the optimizer feature processor name mapping
        and returns a new model with the given optimizers.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        EquinoxStage1Model
            New instance with updated optimizers and
            processor.
        """
        feature_processor_optimizer = (
            self.feature_processor_optimizer.swap_names(
                {opt.name: opt.name for opt in optimizers}
            )
        )
        return EquinoxStage1Model(
            mixed_embedding_optimizer=self.mixed_embedding_optimizer,
            mixed_embedding_task=self.mixed_embedding_task,
            feature_processor_task=self.feature_processor_task,
            feature_processor_optimizer=feature_processor_optimizer,
            history_encoder=self.history_encoder,
            model=self.model,
            optimizers=optimizers,
        )


class Stage1Loader:
    """
    Utility class for loading and storing Stage1Model instances.
    """

    @staticmethod
    def load(path: str) -> Stage1Model:
        """
        Load model from file.

        Parameters
        ----------
        path : str
            Path to model file (without extension).

        Returns
        -------
        Stage1Model
            Loaded model instance.
        """
        _path = Path(path)

        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams: dict = json.loads(f.readline().decode())
            model = EquinoxStage1Model.reconstruct(**hyperparams)
            model = eqx.tree_deserialise_leaves(f, model)
            return model

    @staticmethod
    def store(object: Stage1Model, path: str):
        """
        Store model to file.

        Parameters
        ----------
        object : Stage1Model
            Model instance to store.
        path : str
            Path to save model (without extension).

        Returns
        -------
        str
            Path to saved model file.
        """
        _path = Path(path)
        hyperparams = object.to_dict()

        with open(_path.with_suffix(".eqx"), "wb") as f:
            hyperparams_str = json.dumps(hyperparams)
            f.write((hyperparams_str + "\n").encode())
            eqx.tree_serialise_leaves(f, object)

        return str(_path)


# ============================================================================
