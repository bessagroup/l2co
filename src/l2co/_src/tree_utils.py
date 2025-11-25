"""
Module for pytree utilities.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable
from pathlib import Path

# Third-party
import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

# Local
from .performance import EPSILON

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def combined_loss(model: PyTree, loss_fn: Callable, sample: dict) -> jax.Array:
    """
    Compute the combined loss for a given set of parameters.

    Parameters
    ----------
    model : PyTree
        The model parameters.
    loss_fn : Callable
        The loss function to compute the loss.
    sample : dict
        Sample data for loss computation.

    Returns
    -------
    jax.Array
        The computed loss.
    """
    return loss_fn(model, **sample)


vmapped_loss = eqx.filter_vmap(
    eqx.filter_jit(combined_loss), in_axes=(0, None, None)
)


def ffill(x: jnp.ndarray) -> jnp.ndarray:
    def scan_fn(carry, xi):
        carry = jnp.where(jnp.isnan(xi), carry, xi)
        return carry, carry

    _, filled = jax.lax.scan(scan_fn, jnp.nan, x)
    return filled


def bfill(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.flip(ffill(jnp.flip(x)))


def signlog1p(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def signed_softlog(x: jax.Array, alpha: float = EPSILON) -> jax.Array:
    scaled = jnp.abs(x) / alpha
    y = jnp.log1p(scaled) / jnp.log1p(1.0 / alpha)
    y = jnp.sign(x) * y
    return y


def interpolate(x: jnp.ndarray, n_points: int) -> jnp.ndarray:
    return jax.vmap(
        lambda xi: jnp.interp(
            jnp.linspace(0, 1, n_points), jnp.linspace(0, 1, len(xi)), xi
        )
    )(x)


def minmax_scale(x: jax.Array, min: jax.Array, max: jax.Array) -> jax.Array:
    normed = (x - min) / (max - min + EPSILON)
    # Do this to avoid floating point issues with jitting
    return jnp.clip(normed, min=None, max=1.0)


def log_stabilized_minmax_scale(x: jax.Array) -> jax.Array:
    p = x - jnp.min(x)

    # Only take positive entries for epsilon computation
    pos_vals = jnp.where(p > 0, p, jnp.inf)
    eps = jnp.min(pos_vals)
    eps = jnp.where(jnp.isinf(eps), EPSILON, eps) + EPSILON

    x_log10 = jnp.log10(p + eps)
    x_min = jnp.min(x_log10)
    x_max = jnp.max(x_log10)

    normed = (x_log10 - x_min) / (x_max - x_min + EPSILON)

    # Do this to avoid floating point issues with jitting
    return jnp.clip(normed, 0.0, 1.0)


def log_stabilized_minmax_scale_with_global_min(
    x: jax.Array, global_min: jax.Array
) -> jax.Array:
    shift = jnp.minimum(jnp.min(x), global_min)
    p = x - shift

    # Only take positive entries for epsilon computation
    pos_vals = jnp.where(p > 0, p, jnp.inf)
    eps = jnp.min(pos_vals)
    eps = jnp.where(jnp.isinf(eps), EPSILON, eps) + EPSILON

    x_log10 = jnp.log10(p + eps)

    # x_min = jnp.min(x_log10)
    x_min = jnp.log10(global_min - shift + EPSILON)
    x_max = jnp.max(x_log10)

    normed = (x_log10 - x_min) / (x_max - x_min + EPSILON)

    # Do this to avoid floating point issues with jitting
    return jnp.clip(normed, 0.0, 1.0)


def output_cummin(x: jax.Array) -> jax.Array:
    """
    Compute the cumulative minimum along the first axis.

    Parameters
    ----------
    x : jax.Array
        Input array.

    Returns
    -------
    jax.Array
        Array with cumulative minimum values.
    """
    return jax.lax.cummin(x, axis=0)


def get_best(params: PyTree, losses: jax.Array):
    """
    Retrieve the best parameters and corresponding loss.

    Parameters
    ----------
    params : PyTree
        Parameter tree with shape (n_iters * popsize, dim).
    losses : jax.Array
        Loss array with shape (n_iters * popsize, 1).

    Returns
    -------
    Tuple[PyTree, jax.Array]
        Best parameters and corresponding loss.
    """
    # params: shape = (n_iters*popsize, dim)
    # losses: shape = (n_iters*popsize, 1)
    idx = jnp.argmin(losses)
    best_params = jax.tree.map(lambda x: x[idx], params)
    best_loss = losses[idx]
    return best_params, best_loss


def tree_where(pred, true, false):
    """
    Apply a conditional operation to a PyTree.

    Parameters
    ----------
    pred : jax.Array
        Boolean array for the condition.
    true : PyTree
        PyTree to use when the condition is True.
    false : PyTree
        PyTree to use when the condition is False.

    Returns
    -------
    PyTree
        Resulting PyTree after applying the condition.
    """

    def keep(a, b):
        return jnp.where(pred, a, b)

    return jax.tree.map(keep, true, false)


#                                                  Loading and saving functions
# =============================================================================


def save_model(model, path: str | Path) -> None:
    _path = Path(path)
    model_shape = eqx.filter_eval_shape(lambda x: model, None)

    with open(_path.with_suffix(".pkl"), "wb") as f:
        cloudpickle.dump(model_shape, f)

    eqx.tree_serialise_leaves(_path.with_suffix(".eqx"), model)


def load_model(model_path: str, shape_path: str) -> eqx.Module:
    _model_path = Path(model_path)
    _shape_path = Path(shape_path)
    with open(_shape_path.with_suffix(".pkl"), "rb") as f:
        model_shape = cloudpickle.load(f)

    model = eqx.tree_deserialise_leaves(
        _model_path.with_suffix(".eqx"), model_shape
    )

    return model


def load_jnparray(path: str | Path) -> jnp.ndarray:
    # TODO: investigate this flatten behaviour
    return jnp.load(Path(path)).flatten()


def save_jnparray(model, path: str | Path) -> None:
    jnp.save(path, model)


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def load_dataset(path: str | Path | None) -> dict[str, jnp.ndarray]:
    if path is None:
        return {}
    loaded_data = jnp.load(Path(path).with_suffix(".npz"))
    return {k: jnp.array(v) for k, v in loaded_data.items()}
