import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def mse_loss_fn(model: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)
