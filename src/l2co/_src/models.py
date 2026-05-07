"""
Neural network models for optimizer selection
(Stage0, Stage1, L2CO, MLP)
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import joblib
import numpy as np
import optax
import xarray as xr
from jax_tqdm import scan_tqdm
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from .data_loader import OptimizationDynamics
from .naming import FEATURE
from .optimizerschedule import OptimizationStep
from .task_features import FeatureProcessor, MixedEmbedding
from .tree_utils import traj_and_stair

# =============================================================================


T = TypeVar("T", bound="AbstractModel")


class AbstractModel(eqx.Module):
    """
    Abstract base class for policy models.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize model to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model configuration.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def reconstruct(cls: type[T], **kwargs) -> T:
        """
        Reconstruct model from dictionary.

        Parameters
        ----------
        **kwargs
            Model configuration key-value pairs.

        Returns
        -------
        T
            Reconstructed model instance.
        """
        raise NotImplementedError


class Stage0Model(AbstractModel):
    """Abstract base class for stage-0 optimizer selection.

    Stage-0 models predict initial optimizer logits from task
    features alone, before any optimization history is available.

    Attributes
    ----------
    feature_processor_task : FeatureProcessor
        Processor for transforming raw task features.
    optimizers : Iterable[OptimizationStep]
        Available optimizer configurations to select from.
    """

    feature_processor_task: FeatureProcessor
    optimizers: Iterable[OptimizationStep]

    @classmethod
    @abstractmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        model: Any,
        optimizers: Iterable[OptimizationStep],
    ) -> Stage0Model:
        """Initialize a Stage0Model instance.

        Parameters
        ----------
        feature_processor_task : FeatureProcessor
            Feature processor for task features.
        model : Any
            Underlying model (e.g., scikit-learn or Equinox model).
        optimizers : Iterable[OptimizationStep]
            List of possible optimization steps.

        Returns
        -------
        Stage0Model
            Initialized Stage0Model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass returning logits."""
        raise NotImplementedError

    @abstractmethod
    def pred_logits(self, data: OptimizationDynamics) -> xr.DataArray:
        """Predict raw optimizer logits for all samples.

        Parameters
        ----------
        data : OptimizationDynamics
            Dataset containing task features and labels.

        Returns
        -------
        xr.DataArray
            Predicted logits with sample and label coordinates.
        """
        raise NotImplementedError

    def evaluate(self, data: OptimizationDynamics) -> xr.DataArray:
        """Shared logic for mapping logits to OptimizationSteps."""
        logits_xr = self.pred_logits(data)
        actions = jnp.argmax(logits_xr.values, axis=-1)

        # Map indices to the actual OptimizationStep objects
        optimizers = [data.y["label"].values[int(k)] for k in actions]

        return xr.DataArray(
            np.array(optimizers),
            coords={"sample": data.y["sample"].coords["sample"]},
            dims=["sample"],
        )

    @classmethod
    def reconstruct(
        cls,
        feature_processor_task: dict,
        model: eqx.Module | BaseEstimator,
        optimizers: Iterable[OptimizationStep],
    ) -> Stage0Model:
        """Reconstruct a Stage0Model from serialized components.

        Parameters
        ----------
        feature_processor_task : dict
            Serialized feature processor configuration.
        model : eqx.Module | BaseEstimator
            Underlying model instance or deserialized model.
        optimizers : Iterable[OptimizationStep]
            Serialized optimizer step dictionaries.

        Returns
        -------
        Stage0Model
            Reconstructed model instance.
        """
        feature_processor_task = FeatureProcessor.reconstruct(
            **feature_processor_task
        )
        optimizers = [OptimizationStep.from_dict(step) for step in optimizers]
        return cls(
            feature_processor_task=feature_processor_task,
            model=model,
            optimizers=optimizers,
        )

    def predict(
        self,
        x_feature_task: dict[str, Any],
    ) -> OptimizationStep:
        """Select the best optimizer given task features.

        Parameters
        ----------
        x_feature_task : dict[str, Any]
            Raw task feature dictionary.

        Returns
        -------
        OptimizationStep
            The optimizer with the highest predicted logit.
        """
        x = self.feature_processor_task.transform(x_feature_task)
        logits = self(x)
        action: jax.Array = jnp.argmax(logits)
        optimizer = self.optimizers[action]
        return optimizer

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> Stage0Model:
        """Replace the optimizer set with a new collection.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        Stage0Model
            New model instance with updated optimizers.
        """
        return Stage0Model(
            feature_processor_task=self.feature_processor_task,
            optimizers=optimizers,
        )


# =============================================================================


class ScikitStage0Model(Stage0Model):
    """Stage-0 model backed by a scikit-learn estimator.

    Wraps any scikit-learn ``BaseEstimator`` in a
    ``MultiOutputRegressor`` to predict per-optimizer logits
    from task features.

    Attributes
    ----------
    model : MultiOutputRegressor
        Scikit-learn multi-output regression wrapper.
    """

    model: MultiOutputRegressor

    @classmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        model: BaseEstimator,
        optimizers: Iterable[OptimizationStep],
    ) -> ScikitStage0Model:
        """Create a ScikitStage0Model from a base estimator.

        The estimator is wrapped in a ``MultiOutputRegressor``
        so it can predict one logit per optimizer.

        Parameters
        ----------
        feature_processor_task : FeatureProcessor
            Processor for task features.
        model : BaseEstimator
            Scikit-learn estimator (e.g., Ridge, SVR).
        optimizers : Iterable[OptimizationStep]
            Available optimizer configurations.

        Returns
        -------
        ScikitStage0Model
            Initialized model instance.
        """
        model = MultiOutputRegressor(model)
        return cls(
            feature_processor_task=feature_processor_task,
            model=model,
            optimizers=optimizers,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Scikit-learn expects 2D numpy arrays
        if x.ndim == 1:
            x = x.reshape(1, -1)

        preds = self.model.predict(x)
        return jnp.array(preds)

    def pred_logits(self, data: OptimizationDynamics) -> xr.DataArray:
        """Predict optimizer logits using scikit-learn model.

        Parameters
        ----------
        data : OptimizationDynamics
            Dataset with task features and labels.

        Returns
        -------
        xr.DataArray
            Predicted logits aligned to data coordinates.
        """
        task_features = data.x_features_task.reindex(
            {FEATURE: self.feature_processor_task.vocabs.keys()}
        )
        x = jnp.array(task_features)
        logits = jnp.array(self.model.predict(x))

        y = xr.DataArray(
            logits,
            coords=data.y.coords,
            dims=data.y.dims,
        )
        return y

    def train(
        self, data: OptimizationDynamics, **kwargs
    ) -> tuple[ScikitStage0Model, dict]:
        """Fit the scikit-learn model on filtered data.

        Parameters
        ----------
        data : OptimizationDynamics
            Training dataset (filtered internally).
        **kwargs
            Additional keyword arguments passed to ``fit``.

        Returns
        -------
        tuple[ScikitStage0Model, dict]
            The trained model and an empty metrics dict.
        """
        data_filtered = data.filter()
        x = np.array(data_filtered.x_features_task)
        y = np.array(data_filtered.y)

        self.model.fit(x, y)
        return self, {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with type ``"scikit"``, feature processor
            config, the scikit-learn model, and optimizer list.
        """
        return {
            "feature_processor_task": self.feature_processor_task.to_dict(),
            "type": "scikit",
            "model": self.model,
            "optimizers": [step.to_dict() for step in self.optimizers],
        }

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> ScikitStage0Model:
        """Replace the optimizer set, keeping the trained model.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        ScikitStage0Model
            New instance with updated optimizers.
        """
        return ScikitStage0Model(
            feature_processor_task=self.feature_processor_task,
            model=self.model,
            optimizers=optimizers,
        )


class EquinoxStage0Model(Stage0Model):
    """Stage-0 model backed by an Equinox neural network.

    Uses an ``AbstractModel`` (typically an ``MLP``) to predict
    optimizer logits from task features. Trained with gradient
    descent via ``optax``.

    Attributes
    ----------
    model : AbstractModel
        Equinox neural network (e.g., MLP).
    """

    model: AbstractModel

    @classmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        model: AbstractModel,
        optimizers: Iterable[OptimizationStep],
    ) -> EquinoxStage0Model:
        """Create an EquinoxStage0Model.

        Parameters
        ----------
        feature_processor_task : FeatureProcessor
            Processor for task features.
        model : AbstractModel
            Equinox model (e.g., MLP).
        optimizers : Iterable[OptimizationStep]
            Available optimizer configurations.

        Returns
        -------
        EquinoxStage0Model
            Initialized model instance.
        """
        return cls(
            feature_processor_task=feature_processor_task,
            model=model,
            optimizers=optimizers,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(x)

    def pred_logits(self, data: OptimizationDynamics) -> xr.DataArray:
        """Predict optimizer logits using the Equinox model.

        Vectorizes over samples with ``eqx.filter_vmap``.

        Parameters
        ----------
        data : OptimizationDynamics
            Dataset with task features and labels.

        Returns
        -------
        xr.DataArray
            Predicted logits aligned to data coordinates.
        """
        x = jnp.array(data.x_features_task)
        logits = eqx.filter_vmap(self)(x)

        y = xr.DataArray(
            logits,
            coords=data.y.coords,
            dims=data.y.dims,
        )
        return y

    def train(
        self,
        data: OptimizationDynamics,
        optimizer: optax.GradientTransformation,
        num_steps: int = 100,
        verbose: bool = False,
        scan: bool = True,
        **kwargs,
    ) -> tuple[EquinoxStage0Model, dict]:
        """Train the Equinox model with MSE loss.

        Splits data into 80/20 train/val and delegates to
        ``train_equinox``.

        Parameters
        ----------
        data : OptimizationDynamics
            Training dataset (filtered internally).
        optimizer : optax.GradientTransformation
            Optax optimizer (e.g., ``optax.adam``).
        num_steps : int, optional
            Number of gradient steps. Default is 100.
        verbose : bool, optional
            Whether to show a progress bar. Default is False.
        scan : bool, optional
            Use ``jax.lax.scan`` for training loop.
            Default is True.
        **kwargs
            Extra arguments forwarded to ``train_equinox``.

        Returns
        -------
        tuple[EquinoxStage0Model, dict]
            Trained model and metrics dict with
            ``train_loss`` and ``val_loss`` arrays.
        """
        data_filtered = data.filter()

        data_train, data_val = data_filtered.train_test_split(
            train_fraction=0.8, seed=42
        )
        x_train = jnp.array(data_train.x_features_task)
        y_train = jnp.array(data_train.y)

        x_val = jnp.array(data_val.x_features_task)
        y_val = jnp.array(data_val.y)

        # Re-using your existing train_equinox utility
        model, metrics = train_equinox(
            model=self.model,
            optimizer=optimizer,
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            num_steps=num_steps,
            verbose=verbose,
            scan=scan,
            **kwargs,
        )
        l2co_model = EquinoxStage0Model(
            feature_processor_task=self.feature_processor_task,
            model=model,
            optimizers=self.optimizers,
        )

        return l2co_model, metrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with type ``"equinox"``, feature
            processor config, model config, and optimizer list.
        """
        return {
            "type": "equinox",
            "feature_processor_task": self.feature_processor_task.to_dict(),
            "model": self.model.to_dict(),
            "optimizers": [step.to_dict() for step in self.optimizers],
        }


class Stage0Loader:
    """
    Utility class for loading and storing Stage0Model instances.
    """

    @staticmethod
    def load(path: str) -> Stage0Model:
        """
        Load model from file.

        Parameters
        ----------
        path : str
            Path to model file (without extension).

        Returns
        -------
        Stage0Model
            Loaded model instance.
        """
        _path = Path(path)

        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams: dict = json.loads(f.readline().decode())
            type = hyperparams.pop("type")
            match type:
                case "scikit":
                    hyperparams["model"] = joblib.load(
                        _path.with_suffix(".joblib")
                    )
                    model = ScikitStage0Model.reconstruct(**hyperparams)
                    return model
                case _:
                    raise ValueError(f"Unknown model type {type}")

    @staticmethod
    def store(object: Stage0Model, path: str):
        """
        Store model to file.

        Parameters
        ----------
        object : Stage0Model
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

        if hyperparams["type"] == "scikit":
            joblib_path = _path.with_suffix(".joblib")
            joblib.dump(hyperparams["model"], filename=joblib_path)
            hyperparams["model"] = str(joblib_path)

        with open(_path.with_suffix(".eqx"), "wb") as f:
            hyperparams_str = json.dumps(hyperparams)
            f.write((hyperparams_str + "\n").encode())
            if hyperparams["type"] == "equinox":
                eqx.tree_serialise_leaves(f, object.model)

        return str(_path)


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


class L2COModel(eqx.Module):
    """Combined two-stage model for optimizer selection.

    Main inference entry point that delegates to a stage-0
    model (task features only) for initial selection and a
    stage-1 model (task + optimizer features + history) for
    per-step dynamic selection during optimization.

    Attributes
    ----------
    stage0_model : Stage0Model
        Model for initial optimizer selection.
    stage1_model : Stage1Model
        Model for per-step optimizer selection.
    """

    stage0_model: Stage0Model
    stage1_model: Stage1Model

    def predict_stage0(
        self, task_features: dict[str, Any]
    ) -> OptimizationStep:
        """Select initial optimizer using stage-0 model.

        Parameters
        ----------
        task_features : dict[str, Any]
            Raw task feature dictionary.

        Returns
        -------
        OptimizationStep
            Selected optimizer configuration.
        """
        return self.stage0_model.predict(task_features)

    def predict_stage1(
        self,
        task_features: dict[str, Any],
        optimizer_features: dict[str, Any],
        x_output_mean: jax.Array,
        x_output_min: jax.Array,
        global_min: jax.Array,
    ) -> OptimizationStep:
        """Select optimizer using stage-1 model with history.

        Parameters
        ----------
        task_features : dict[str, Any]
            Raw task feature dictionary.
        optimizer_features : dict[str, Any]
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
            Selected optimizer configuration.
        """
        return self.stage1_model.predict(
            x_feature_task=task_features,
            x_feature_optimizer=optimizer_features,
            x_output_mean=jnp.array(x_output_mean),
            x_output_min=jnp.array(x_output_min),
            global_min=jnp.array(global_min),
        )

    @classmethod
    def load(cls, stage0_model: str, stage1_model: str) -> L2COModel:
        """Load a full L2COModel from saved stage-0 and stage-1 files.

        Parameters
        ----------
        stage0_model : str
            Path to stage-0 model file (without extension).
        stage1_model : str
            Path to stage-1 model file (without extension).

        Returns
        -------
        L2COModel
            Loaded two-stage model.
        """
        stage0_model_loaded = Stage0Loader.load(stage0_model)
        stage1_model_loaded = Stage1Loader.load(stage1_model)
        return cls(
            stage0_model=stage0_model_loaded,
            stage1_model=stage1_model_loaded,
        )

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> L2COModel:
        """Replace optimizers in both stages.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        L2COModel
            New instance with updated optimizers in both
            stage-0 and stage-1 models.
        """
        new_stage0 = self.stage0_model.swap_optimizers(optimizers)
        new_stage1 = self.stage1_model.swap_optimizers(optimizers)
        return L2COModel(
            stage0_model=new_stage0,
            stage1_model=new_stage1,
        )


# =============================================================================
class LSTM_Encoder(AbstractModel):
    """
    LSTM Encoder module.

    Parameters
    ----------
    in_size : int
        Input size of the LSTM encoder.
    hidden_size : int
        Hidden size of the LSTM encoder.
    key : jax.Array
        Random key for initializing the LSTM cell.

    Attributes
    ----------
    hidden_size : int
        Hidden size of the LSTM encoder.
    cell : eqx.nn.LSTMCell
        LSTM cell used in the encoder.
    linear : eqx.nn.Linear
        Linear layer applied to the final hidden state.
    """

    hidden_size: int
    in_size: int
    out_size: int
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, *, key):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.in_size = in_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=key)

    @classmethod
    def init(self, in_size: int, hidden_size: int, out_size: int, *, key):
        return LSTM_Encoder(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            key=key,
        )

    def __call__(self, x):
        """
        Encode sequence using LSTM.

        Parameters
        ----------
        x : array-like
            Input sequence to encode.
        init_state : array-like
            Initial hidden state for LSTM.

        Returns
        -------
        array-like
            Final hidden state after encoding sequence.
        """

        def scan_fn(state, input):
            return (self.cell(input, state), None)

        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

        (h, c), _ = jax.lax.scan(scan_fn, init_state, x)
        h = self.linear(h)
        return h

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize LSTM encoder configuration.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary.
        """
        return {
            "in_size": self.in_size,
            "hidden_size": self.hidden_size,
            "out_size": self.out_size,
        }

    @classmethod
    def reconstruct(
        cls,
        in_size: int,
        hidden_size: int,
        out_size: int,
    ) -> LSTM_Encoder:
        return cls(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            key=jr.key(0),
        )


# =============================================================================


class MLP(AbstractModel):
    """Multi-layer perceptron wrapper around ``eqx.nn.MLP``.

    Provides ``init``, ``to_dict``, and ``reconstruct`` for
    convenient construction and serialization.

    Attributes
    ----------
    model : eqx.nn.MLP
        Underlying Equinox MLP module.
    """

    model: eqx.nn.MLP

    def __init__(
        self,
        model: eqx.Module,
    ):
        self.model = model

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        *,
        key,
    ):
        """Create an MLP with the given architecture.

        Parameters
        ----------
        in_size : int
            Input feature dimension.
        out_size : int
            Output dimension (number of classes/logits).
        width_size : int
            Width of each hidden layer.
        depth : int
            Number of hidden layers.
        activation : Callable, optional
            Activation function. Default is ``jax.nn.relu``.
        key : jax.Array
            PRNG key for weight initialization.

        Returns
        -------
        MLP
            Initialized MLP instance.
        """
        model = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )
        return cls(model=model)

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        return self.model(x)

    def to_dict(self) -> dict[str, Any]:
        """Serialize MLP configuration to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``in_size``, ``out_size``,
            ``width_size``, ``depth``, and ``activation``
            (as a string name).
        """
        match self.model.activation:
            case jax.nn.relu:
                activation_str = "relu"
            case jax.nn.tanh:
                activation_str = "tanh"
        return {
            "in_size": self.model.in_size,
            "out_size": self.model.out_size,
            "width_size": self.model.width_size,
            "depth": self.model.depth,
            "activation": activation_str,
        }

    @classmethod
    def reconstruct(
        cls,
        activation: Literal["relu", "tanh"],
        **kwargs,
    ) -> MLP:
        """Reconstruct an MLP from serialized config.

        Weights are initialized with a dummy PRNG key and
        should be overwritten via
        ``eqx.tree_deserialise_leaves`` when loading.

        Parameters
        ----------
        activation : {"relu", "tanh"}
            Activation function name.
        **kwargs
            Remaining MLP config (``in_size``, ``out_size``,
            ``width_size``, ``depth``).

        Returns
        -------
        MLP
            Reconstructed MLP instance.
        """
        match activation:
            case "relu":
                activation_fn = jax.nn.relu
            case "tanh":
                activation_fn = jax.nn.tanh
        model = eqx.nn.MLP(
            **kwargs,
            activation=activation_fn,
            key=jr.key(0),
        )

        return cls(
            model=model,
        )


def mse_loss(model: AbstractModel, x, y) -> jax.Array:
    """Mean squared error between model predictions and targets."""
    preds = eqx.filter_vmap(model)(x)
    return jnp.mean((preds - y) ** 2)


def train_equinox(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    train_data: tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array],
    val_data: tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]
    | None = None,
    num_steps: int = 100,
    verbose: bool = False,
    scan: bool = True,
) -> tuple[eqx.Module, dict[str, jax.Array]]:
    """Train an Equinox model with MSE loss via optax.

    Partitions the model into trainable parameters and static
    structure, then runs a training loop using either
    ``jax.lax.scan`` (JIT-compiled) or a Python for-loop.

    Parameters
    ----------
    model : eqx.Module
        Equinox model to train.
    optimizer : optax.GradientTransformation
        Optax optimizer instance.
    train_data : tuple
        Tuple of ``(x_train, y_train)`` arrays.
    val_data : tuple or None, optional
        Tuple of ``(x_val, y_val)`` arrays. If None,
        validation loss is reported as NaN. Default is None.
    num_steps : int, optional
        Number of gradient steps. Default is 100.
    verbose : bool, optional
        Show progress bar. Default is False.
    scan : bool, optional
        Use ``jax.lax.scan`` (True) or a Python for-loop
        (False). Default is True.

    Returns
    -------
    tuple[eqx.Module, dict[str, jax.Array]]
        Trained model and metrics dict with
        ``train_loss`` and ``val_loss`` arrays of shape
        ``(num_steps,)``.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    loss_fn = mse_loss
    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def make_step(carry, _):
        params, opt_state = carry
        model = eqx.combine(params, static)

        # Training Branch
        train_loss, grads = loss_and_grad_fn(model, *train_data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)

        # Validation Branch (Conditional)
        val_loss = (
            loss_fn(model, *val_data) if val_data is not None else jnp.nan
        )

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        return (new_params, opt_state), metrics

    # 2. Execution Logic (Scan vs Loop)
    if scan:
        step_fn = scan_tqdm(num_steps)(make_step) if verbose else make_step
        (params, opt_state), metrics = jax.lax.scan(
            step_fn,
            (params, opt_state),
            xs=jnp.arange(num_steps),
            length=num_steps,
        )
    else:
        metrics_list = []
        for _ in tqdm(range(num_steps), disable=not verbose):
            (params, opt_state), step_metrics = make_step(
                (params, opt_state), None
            )
            metrics_list.append(step_metrics)
        metrics = {
            k: jnp.stack([m[k] for m in metrics_list]) for k in metrics_list[0]
        }

    return eqx.combine(params, static), metrics
