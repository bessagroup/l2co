"""Loss function definitions for optimization tasks (MSE, cross-entropy)."""

# Third-party
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree

# =============================================================================


def mse_loss_fn(model: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    """
    Compute mean squared error loss.

    Parameters
    ----------
    model : PyTree
        Model parameters.
    x : jnp.ndarray
        Input data array.
    y : jnp.ndarray
        Target values array.

    Returns
    -------
    jnp.ndarray
        Mean squared error loss value.

    Notes
    -----
    Applies the model via vmap to all inputs and computes MSE:
    $L = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$.
    """
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


def mean_categorical_cross_entropy_loss_fn(
    model: PyTree, x: jnp.ndarray, y: jnp.ndarray
):
    """
    Compute mean categorical cross-entropy loss.

    Parameters
    ----------
    model : PyTree
        Model parameters.
    x : jnp.ndarray
        Input data array.
    y : jnp.ndarray
        Integer class labels.

    Returns
    -------
    jnp.ndarray
        Mean categorical cross-entropy loss value.

    Notes
    -----
    Uses softmax cross-entropy with integer labels from optax.
    """
    y_pred = jax.vmap(model)(x)
    return optax.losses.softmax_cross_entropy_with_integer_labels(
        y_pred, y
    ).mean()


def mean_categorical_cross_entropy_loss_fn_l2(
    model: PyTree, x: jnp.ndarray, y: jnp.ndarray, l2_lambda: float = 0.005
):
    """
    Compute mean categorical cross-entropy loss with L2 regularization.

    Parameters
    ----------
    model : PyTree
        Model parameters.
    x : jnp.ndarray
        Input data array.
    y : jnp.ndarray
        Integer class labels.
    l2_lambda : float, optional
        L2 regularization strength, by default 0.005.

    Returns
    -------
    jnp.ndarray
        Loss value including L2 regularization term.

    Notes
    -----
    Total loss is: $L = L_{CE} + \\lambda \\sum_p p^2$, where
    $L_{CE}$ is cross-entropy and $\\lambda$ is l2_lambda.
    """
    y_pred = jax.vmap(model)(x)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(
        y_pred, y
    ).mean()

    # Compute L2 regularization term
    l2_penalty = sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(model)
    )
    loss += l2_lambda * l2_penalty

    return loss
