from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .data_loader import OptimizationDynamics
from .task_features import FeatureProcessor, MixedEmbedding


class L2COModel:
    def __init__(
        self, feature_processor: FeatureProcessor, model: Any, type: str
    ):
        self.feature_processor = feature_processor
        self.model = model
        self.type = type

    def train(self, data: OptimizationDynamics):
        if self.type == "scikit":
            self._train_scikit(data.x, data.y)
        else:
            raise ValueError(f"Unknown model type {self.type}")

    def predict(self, data: OptimizationDynamics) -> jax.Array:
        if self.type == "scikit":
            return self._predict_scikit(data.x)
        else:
            raise ValueError(f"Unknown model type {self.type}")

    def _train_scikit(self, x: Iterable[dict[str, Any]], y: jax.Array):
        x_array = jnp.stack([self.feature_processor.transform(xi) for xi in x])

        self.model.fit(x_array, y)

    def _predict_scikit(self, x: Iterable[dict[str, Any]]) -> jax.Array:
        x_array = jnp.stack([self.feature_processor.transform(xi) for xi in x])

        y_pred: np.ndarray = self.model.predict(x_array)
        return jnp.array(y_pred)


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


def train(
    model: L2COModel,
    data: tuple[jax.Array, jax.Array],
    optimizer: optax.GradientTransformation,
    num_steps: int = 100,
) -> tuple[L2COModel, list[float]]:
    """Train the model with Adam on an iterable of (x, y) batches."""
    x, y = data
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_and_grad = eqx.filter_value_and_grad(mse_loss)

    loss_history: list[float] = []

    for _ in tqdm(range(num_steps)):
        loss, grads = loss_and_grad(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        loss_history.append(float(loss))

    return model, loss_history
