from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import joblib
import numpy as np
import optax
from jaxtyping import PRNGKeyArray
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from .data_loader import OptimizationDynamics
from .optimizerschedule import OptimizationStep
from .task_features import FeatureProcessor, MixedEmbedding

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
    def reconstruct(cls: type[T], data: dict[str, Any]) -> T:
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


# =============================================================================


class L2COModel(AbstractModel):
    feature_processor: FeatureProcessor
    model: BaseEstimator | AbstractModel
    type: Literal["scikit", "equinox"]
    optimizers: Iterable[OptimizationStep]

    @classmethod
    def init(
        cls,
        feature_processor: FeatureProcessor,
        model: BaseEstimator | AbstractModel,
        type: Literal["scikit", "equinox"],
        optimizers: Iterable[OptimizationStep],
    ) -> L2COModel:
        if type not in ["scikit", "equinox"]:
            raise ValueError(f"Unknown model type {type}")

        if type == "scikit":
            model = MultiOutputRegressor(model)

        return cls(
            feature_processor=feature_processor,
            model=model,
            type=type,
            optimizers=optimizers,
        )

    def train(
        self, data: OptimizationDynamics, **kwargs
    ) -> tuple[L2COModel, dict]:
        x_array = jnp.stack(
            [self.feature_processor.transform(xi) for xi in data.to_tag_list()]
        )

        if self.type == "scikit":
            model, metrics = train_scikit(
                model=self.model, x=x_array, y=jnp.array(data.y), **kwargs
            )
        elif self.type == "equinox":
            model, metrics = train_equinox(
                model=self.model, x=x_array, y=jnp.array(data.y), **kwargs
            )
        else:
            raise ValueError(f"Unknown model type {self.type}")

        l2co_model = L2COModel(
            feature_processor=self.feature_processor,
            model=model,
            type=self.type,
            optimizers=self.optimizers,
        )

        return l2co_model, metrics

    def predict_logits(self, x: dict[str, Any]) -> jax.Array:
        x_array = self.feature_processor.transform(x)
        if self.type == "scikit":
            logits = predict_scikit(self.model, x_array)
        elif self.type == "equinox":
            logits = predict_equinox(self.model, x_array)
        else:
            raise ValueError(f"Unknown model type {self.type}")

        return logits

    def predict(
        self, x: dict[str, Any], key: PRNGKeyArray, n_realizations: int = 1
    ) -> jax.Array:
        logits = self.predict_logits(x)
        keys = jr.split(key, num=n_realizations)
        actions = jax.vmap(jr.categorical, in_axes=(0, None))(keys, logits)

        action_list = list(map(int, actions))
        steps = [self.optimizers[i] for i in action_list]
        return steps

    def batch_predict(self, x: dict[str, Any], key: PRNGKeyArray) -> jax.Array:
        x_array = jnp.stack([self.feature_processor.transform(xi) for xi in x])
        if self.type == "scikit":
            logits = predict_scikit(self.model, x_array)
        elif self.type == "equinox":
            logits = predict_equinox(self.model, x_array)
        else:
            raise ValueError(f"Unknown model type {self.type}")

        # Sample from logits
        keys = jr.split(key, num=len(x))
        actions = jax.vmap(jr.categorical)(keys, logits)
        action_list = list(map(int, actions))
        steps = [self.optimizers[i] for i in action_list]
        return steps

    def to_dict(self) -> dict[str, Any]:
        match self.type:
            case "scikit":
                model = self.model
            case "equinox":
                model = self.model.to_dict()

        return {
            "feature_processor": self.feature_processor.to_dict(),
            "model": model,
            "type": self.type,
            "optimizers": [step.to_dict() for step in self.optimizers],
        }

    @classmethod
    def reconstruct(
        cls,
        feature_processor: dict,
        optimizers: Iterable[dict],
        model: dict | BaseEstimator,
        type: Literal["scikit", "equinox"],
        key: PRNGKeyArray,
        **kwargs,
    ) -> L2COModel:
        feature_processor = FeatureProcessor.reconstruct(**feature_processor)
        optimizers = [
            OptimizationStep.from_yaml(step_dict) for step_dict in optimizers
        ]
        match type:
            case "scikit":
                pass
            case "equinox":
                model = make(key=key, **model)
            case _:
                raise ValueError(f"Unknown model type {type}")

        return cls(
            feature_processor=feature_processor,
            model=model,
            optimizers=optimizers,
            type=type,
            **kwargs,
        )


# =============================================================================


class L2COLoader:
    """
    Utility class for loading and storing Equinox models.
    """

    @staticmethod
    def load(path: str) -> L2COModel:
        """
        Load model from file.

        Parameters
        ----------
        path : str
            Path to model file (without extension).

        Returns
        -------
        eqx.Module
            Loaded model instance.
        """
        _path = Path(path)
        key = jr.key(0)

        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            if hyperparams["type"] == "equinox":
                model = make(key=key, **hyperparams)
                return eqx.tree_deserialise_leaves(f, model)
            elif hyperparams["type"] == "scikit":
                hyperparams["model"] = joblib.load(hyperparams["model"])
                model = make(key=key, **hyperparams)
                return model
            else:
                raise ValueError(f"Unknown model type {hyperparams['type']}")

    @staticmethod
    def store(object: L2COModel, path: str):
        """
        Store model to file.

        Parameters
        ----------
        object : AbstractModel
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
        hyperparams["model_type"] = object.__class__.__name__.lower()

        if hyperparams["type"] == "scikit":
            joblib_path = _path.with_suffix(".joblib")
            joblib.dump(object.model, filename=joblib_path)
            hyperparams["model"] = str(joblib_path)

        with open(_path.with_suffix(".eqx"), "wb") as f:
            hyperparams_str = json.dumps(hyperparams)
            f.write((hyperparams_str + "\n").encode())
            if hyperparams["type"] == "equinox":
                eqx.tree_serialise_leaves(f, object)

        return str(_path)


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
            return L2COModel.reconstruct(**kwargs)
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
    mixed_embedding: MixedEmbedding
    model: eqx.nn.MLP

    def __init__(
        self,
        mixed_embedding: MixedEmbedding,
        model: eqx.Module,
    ):
        self.mixed_embedding = mixed_embedding
        self.model = model

    def __call__(
        self,
        x: dict[str, Any],
    ) -> jax.Array:
        return self.model(self.mixed_embedding(x))

    def to_dict(self) -> dict[str, Any]:
        match self.model.activation:
            case jax.nn.relu:
                activation_str = "relu"
            case jax.nn.tanh:
                activation_str = "tanh"
        return {
            "model_type": self.__class__.__name__.lower(),
            "mixed_embedding": self.mixed_embedding.to_dict(),
            "model": {
                "in_size": self.model.in_size,
                "out_size": self.model.out_size,
                "width_size": self.model.width_size,
                "depth": self.model.depth,
                "activation": activation_str,
            },
        }

    @classmethod
    def reconstruct(
        cls,
        mixed_embedding: dict,
        model: dict,
        key: PRNGKeyArray,
        **kwargs,
    ) -> MLP:
        mixed_embedding = MixedEmbedding.reconstruct(
            **mixed_embedding, key=key
        )

        match model["activation"]:
            case "relu":
                model["activation"] = jax.nn.relu
            case "tanh":
                model["activation"] = jax.nn.tanh

        mlp_model = eqx.nn.MLP(
            **model,
            key=key,
        )

        return cls(
            mixed_embedding=mixed_embedding,
            model=mlp_model,
        )


def mse_loss(
    model: AbstractModel, x: dict[str, Any], y: jax.Array
) -> jax.Array:
    """Mean squared error between model predictions and targets."""
    preds = eqx.filter_vmap(model)(x)
    return jnp.mean((preds - y) ** 2)


def train_equinox(
    model: AbstractModel,
    x: jax.Array,
    y: jax.Array,
    optimizer: optax.GradientTransformation,
    num_steps: int = 100,
) -> tuple[AbstractModel, list[float]]:
    """Train the model with Adam on an iterable of (x, y) batches."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_and_grad = eqx.filter_value_and_grad(mse_loss)

    loss_history: list[float] = []

    for _ in tqdm(range(num_steps)):
        loss, grads = loss_and_grad(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        loss_history.append(float(loss))

    return model, loss_history
