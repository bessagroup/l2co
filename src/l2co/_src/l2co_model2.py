from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import PRNGKeyArray
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from .data_loader import OptimizationDynamics
from .optimizerschedule import OptimizationStep
from .task_features import FeatureProcessor, MixedEmbedding


@dataclass
class L2COModel:
    feature_processor: FeatureProcessor
    model: BaseEstimator | eqx.Module
    type: Literal["scikit", "equinox"]
    optimizers: Iterable[OptimizationStep]

    def __post_init__(self):
        if self.type not in ["scikit", "equinox"]:
            raise ValueError(f"Unknown model type {self.type}")

        if self.type == "scikit":
            self.model = MultiOutputRegressor(self.model)

    def train(self, data: OptimizationDynamics, **kwargs):
        x_array = jnp.stack(
            [self.feature_processor.transform(xi) for xi in data.to_tag_list()]
        )

        if self.type == "scikit":
            self.model = train_scikit(
                model=self.model, x=x_array, y=jnp.array(data.y), **kwargs
            )
        elif self.type == "equinox":
            self.model = train_equinox(
                model=self.model, x=x_array, y=jnp.array(data.y), **kwargs
            )
        else:
            raise ValueError(f"Unknown model type {self.type}")

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
        if self.type == "scikit":
            model = ...
        elif self.type == "equinox":
            model = ...
        else:
            raise ValueError(f"Unknown model type {self.type}")

        return {
            "feature_processor": self.feature_processor.to_dict(),
            "model": model,
            "type": self.type,
            "optimizers": [step.to_dict() for step in self.optimizers],
        }

    @classmethod
    def reconstruct(
        cls, feature_processor, optimizers, model, **kwargs
    ) -> L2COModel:
        feature_processor = FeatureProcessor.reconstruct(**feature_processor)
        optimizers = [
            OptimizationStep.from_yaml(**step_dict) for step_dict in optimizers
        ]

        ## TODO: reconstruct model based on type.
        return cls(
            feature_processor=feature_processor,
            model=model,
            type=kwargs["type"],
            optimizers=optimizers,
        )


# =============================================================================


def train_scikit(
    model: BaseEstimator,
    x: jax.Array,
    y: jax.Array,
    **kwargs,
):
    model.fit(x, y, **kwargs)
    return model


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
) -> jax.Array: ...


class MLPModel(eqx.Module):
    mixed_embedding: MixedEmbedding
    model: eqx.Module

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


def mse_loss(model: L2COModel, x: dict[str, Any], y: jax.Array) -> jax.Array:
    """Mean squared error between model predictions and targets."""
    preds = eqx.filter_vmap(model)(x)
    return jnp.mean((preds - y) ** 2)


def train_equinox(
    model: L2COModel,
    x: jax.Array,
    y: jax.Array,
    optimizer: optax.GradientTransformation,
    num_steps: int = 100,
) -> tuple[L2COModel, list[float]]:
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
