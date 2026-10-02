"""Stage-0 optimizer-selection models (cold-start from task features)."""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import xarray as xr
from l2co_optimizers import OptimizationStep
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor

from ..naming import FEATURE
from ..optimization_dynamics import OptimizationDynamics
from ..task_features import FeatureProcessor
from .base import AbstractModel
from .training import train_equinox

# =============================================================================


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

        Generic over the concrete subclass: replaces only the
        ``optimizers`` field via :func:`equinox.tree_at`, so every
        other field (including subclass-specific ones like ``model``)
        is carried over unchanged. The previous implementation
        instantiated the abstract base directly, which raised
        ``TypeError`` for every concrete model.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        Stage0Model
            New model instance (same concrete class) with updated
            optimizers.
        """
        return eqx.tree_at(
            lambda m: m.optimizers,
            self,
            list(optimizers),
            is_leaf=lambda x: isinstance(x, list),
        )

    def with_feature_impute(self, impute: dict[str, Any]) -> Stage0Model:
        """Fill task-tag features the checkpoint's schema expects but a
        foreign task set does not carry.

        Generic over the concrete subclass: replaces only
        ``feature_processor_task`` via :func:`equinox.tree_at` (same
        idiom as :meth:`swap_optimizers`), so every other field --
        including the trained ``model`` -- is carried over unchanged.
        Delegates to :meth:`FeatureProcessor.with_imputation`, which
        leaves vocabularies and feature count untouched, so the loaded
        weights keep matching.

        Parameters
        ----------
        impute : dict[str, Any]
            Feature name to fill value, forwarded to
            ``FeatureProcessor.with_imputation``.

        Returns
        -------
        Stage0Model
            New model instance (same concrete class) with the filled
            processor.
        """
        return eqx.tree_at(
            lambda m: m.feature_processor_task,
            self,
            self.feature_processor_task.with_imputation(impute),
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
