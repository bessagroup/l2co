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
        data : dict[str, Any]
            Dictionary containing model configuration.

        Returns
        -------
        T
            Reconstructed model instance.
        """
        raise NotImplementedError


class Stage0Model(AbstractModel):
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
        x = self.feature_processor_task.transform(x_feature_task)
        logits = self(x)
        action: jax.Array = jnp.argmax(logits)
        optimizer = self.optimizers[action]
        return optimizer

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> Stage0Model:
        return Stage0Model(
            feature_processor_task=self.feature_processor_task,
            optimizers=optimizers,
        )


# =============================================================================


class ScikitStage0Model(Stage0Model):
    model: MultiOutputRegressor

    @classmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        model: BaseEstimator,
        optimizers: Iterable[OptimizationStep],
    ) -> ScikitStage0Model:
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
        task_features = data.x_features_task.reindex(
            {FEATURE: self.feature_processor_task.schema.keys()}
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
        data_filtered = data.filter()
        x = np.array(data_filtered.x_features_task)
        y = np.array(data_filtered.y)

        self.model.fit(x, y)
        return self, {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_processor_task": self.feature_processor_task.to_dict(),
            "type": "scikit",
            "model": self.model,
            "optimizers": [step.to_dict() for step in self.optimizers],
        }

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> ScikitStage0Model:
        return ScikitStage0Model(
            feature_processor_task=self.feature_processor_task,
            model=self.model,
            optimizers=optimizers,
        )


class EquinoxStage0Model(Stage0Model):
    model: AbstractModel  # This would be your MLP or other Equinox module

    @classmethod
    def init(
        cls,
        feature_processor_task: FeatureProcessor,
        model: AbstractModel,
        optimizers: Iterable[OptimizationStep],
    ) -> EquinoxStage0Model:
        return cls(
            feature_processor_task=feature_processor_task,
            model=model,
            optimizers=optimizers,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(x)

    def pred_logits(self, data: OptimizationDynamics) -> xr.DataArray:
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
        raise NotImplementedError

    def predict(
        self,
        x_feature_task: dict[str, Any],
        x_feature_optimizer: dict[str, Any],
        x_output_mean: jax.Array,
        x_output_min: jax.Array,
        global_min: jax.Array,
    ) -> OptimizationStep:
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
        # 2. Split by problem to prevent leakage
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
    stage0_model: Stage0Model
    stage1_model: Stage1Model

    def predict_stage0(
        self, task_features: dict[str, Any]
    ) -> OptimizationStep:
        return self.stage0_model.predict(task_features)

    def predict_stage1(
        self,
        task_features: dict[str, Any],
        optimizer_features: dict[str, Any],
        x_output_mean: jax.Array,
        x_output_min: jax.Array,
        global_min: jax.Array,
    ) -> OptimizationStep:
        return self.stage1_model.predict(
            x_feature_task=task_features,
            x_feature_optimizer=optimizer_features,
            x_output_mean=jnp.array(x_output_mean),
            x_output_min=jnp.array(x_output_min),
            global_min=jnp.array(global_min),
        )

    @classmethod
    def load(cls, stage0_model: str, stage1_model: str) -> L2COModel:
        stage0_model_loaded = Stage0Loader.load(stage0_model)
        stage1_model_loaded = Stage1Loader.load(stage1_model)
        return cls(
            stage0_model=stage0_model_loaded,
            stage1_model=stage1_model_loaded,
        )

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> L2COModel:
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

# =============================================================================


def make(*, model_type: str, **kwargs) -> AbstractModel:
    """
    Factory function to create policy models.

    Parameters
    ----------
    model_type : str
        Type of model to create.
    **kwargs
        Model-specific configuration parameters.

    Returns
    -------
    AbstractModel
        Instantiated policy model.

    Raises
    ------
    ValueError
        If model_type is unknown.
    """
    match model_type:
        case "mlp":
            return MLP.reconstruct(**kwargs)
        case "l2comodel":
            return Stage0Model.reconstruct(**kwargs)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================


def train_scikit(
    model: BaseEstimator,
    x: jax.Array,
    y: jax.Array,
    **kwargs,
) -> tuple[BaseEstimator, dict]:
    model.fit(x, y, **kwargs)
    return model, {}


def predict_scikit(
    model: BaseEstimator,
    x: jax.Array,
) -> jax.Array:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y_pred: np.ndarray = model.predict(x)
    return jnp.array(y_pred)


def predict_equinox(
    model: eqx.Module,
    x: jax.Array,
) -> jax.Array:
    logits = model(x)
    return logits


class MLP(AbstractModel):
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
    | None = None,  # Optional validation
    num_steps: int = 100,
    verbose: bool = False,
    scan: bool = True,
) -> tuple[eqx.Module, dict[str, jax.Array]]:
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
