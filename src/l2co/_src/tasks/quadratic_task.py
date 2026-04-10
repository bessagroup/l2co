"""Quadratic optimization task creation with random matrices."""

from functools import partial
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from ..task import Task

# =============================================================================


def generate_dataset(dimensionality: int, *, key: jr.PRNGKey):
    """Generate a random quadratic problem ``(W, y)``.

    Parameters
    ----------
    dimensionality : int
        Size of the square matrix and target vector.
    key : jr.PRNGKey
        Random key for sampling.

    Returns
    -------
    dict[str, jnp.ndarray]
        ``'W'`` matrix and ``'y'`` target vector.
    """
    key_W, key_y = jr.split(key)
    Ws = jr.normal(key_W, shape=(dimensionality, dimensionality))
    ys = jr.normal(key_y, shape=(dimensionality,))
    return {"W": Ws, "y": ys}


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    """Save a dataset dictionary to an ``.npz`` file.

    Parameters
    ----------
    dataset : dict[str, jnp.ndarray]
        Dictionary of arrays to persist.
    path : str or Path
        Destination file path.
    """
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


# =============================================================================


def compute_quadratic_loss(model, W, y):
    """Compute the squared residual loss ``||W x - y||^2``.

    Parameters
    ----------
    model : jnp.ndarray
        Current parameter vector.
    W : jnp.ndarray
        Coefficient matrix.
    y : jnp.ndarray
        Target vector.

    Returns
    -------
    jnp.ndarray
        Scalar sum of squared residuals.
    """
    residual = jnp.dot(W, model) - y
    return jnp.sum(residual**2)


def create_quadratic_task(dimensionality: int, seed: int) -> Task:
    """Create a quadratic optimisation task ``min ||W x - y||^2``.

    Parameters
    ----------
    dimensionality : int
        Problem dimensionality.
    seed : int
        Random seed for matrix and vector generation.

    Returns
    -------
    Task
        Configured quadratic optimisation task.
    """
    tag = {}
    tag["seed"] = seed

    key = jr.key(int(seed))

    model_key, dataset_key, batch_key = jr.split(key, 3)

    tag["task_name"] = "quadratic_functions"

    model = jnp.zeros(
        dimensionality,
    )

    tag["dimensionality"] = dimensionality

    dataset = generate_dataset(dimensionality=dimensionality, key=dataset_key)

    loss_fn = partial(compute_quadratic_loss, W=dataset["W"], y=dataset["y"])

    return Task(model=model, loss_fn=loss_fn, dataset=None, tag=tag)
