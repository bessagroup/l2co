# Third-party
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree

# =============================================================================


def mse_loss_fn(model: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


def mean_categorical_cross_entropy_loss_fn(
        model: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    y_pred = jax.vmap(model)(x)
    return optax.losses.softmax_cross_entropy_with_integer_labels(
        y_pred, y).mean()


def mean_categorical_cross_entropy_loss_fn_l2(
        model: PyTree, x: jnp.ndarray, y: jnp.ndarray,
        l2_lambda: float = 0.005):
    """
    Computes mean categorical cross-entropy loss with L2 regularization.

    Args:
        model: PyTree representing model parameters.
        x: Input data.
        y: Integer class labels.
        l2_lambda: L2 regularization strength.

    Returns:
        Loss value.
    """
    y_pred = jax.vmap(model)(x)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(
        y_pred, y).mean()

    # Compute L2 regularization term
    l2_penalty = sum(jnp.sum(jnp.square(p))
                     for p in jax.tree_util.tree_leaves(model))
    loss += l2_lambda * l2_penalty

    return loss
