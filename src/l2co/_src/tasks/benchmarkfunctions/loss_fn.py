import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree


def mse_loss_fn(model: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


def mean_categorical_cross_entropy_loss_fn(
        model: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    y_pred = jax.vmap(model)(x)
    return optax.losses.softmax_cross_entropy_with_integer_labels(
        y_pred, y).mean()
