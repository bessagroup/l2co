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
from jaxtyping import PRNGKeyArray, PyTree

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


def combined_loss_with_rng(
    model: PyTree, loss_fn: Callable, sample: dict, key: PRNGKeyArray
) -> jax.Array:
    """
    Compute the combined loss for a given set of parameters with RNG.

    Parameters
    ----------
    model : PyTree
        The model parameters.
    loss_fn : Callable
        The loss function to compute the loss.
    sample : dict
        Sample data for loss computation.
    key : PRNGKeyArray
        Random number generator key.

    Returns
    -------
    jax.Array
        The computed loss.
    """
    return loss_fn(model, **sample, key=key)


vmapped_loss_and_grad = eqx.filter_vmap(
    eqx.filter_value_and_grad(combined_loss), in_axes=(0, None, None)
)
vmapped_loss = eqx.filter_vmap(combined_loss, in_axes=(0, None, None))


vmapped_loss_and_grad_with_rng = eqx.filter_vmap(
    eqx.filter_value_and_grad(combined_loss_with_rng),
    in_axes=(0, None, None, 0),
)

vmapped_loss_with_rng = eqx.filter_vmap(
    combined_loss_with_rng,
    in_axes=(0, None, None, 0),
)


def ffill(x: jnp.ndarray) -> jnp.ndarray:
    """Forward fill NaN values in array.

    Parameters
    ----------
    x : jnp.ndarray
        Input array with potential NaN values.

    Returns
    -------
    jnp.ndarray
        Array with NaN values forward filled.
    """

    def scan_fn(carry, xi):
        carry = jnp.where(jnp.isnan(xi), carry, xi)
        return carry, carry

    _, filled = jax.lax.scan(scan_fn, jnp.nan, x)
    return filled


def bfill(x: jnp.ndarray) -> jnp.ndarray:
    """Backward fill NaN values in array.

    Parameters
    ----------
    x : jnp.ndarray
        Input array with potential NaN values.

    Returns
    -------
    jnp.ndarray
        Array with NaN values backward filled.
    """
    return jnp.flip(ffill(jnp.flip(x)))


def signlog1p(x: jnp.ndarray) -> jnp.ndarray:
    """Apply signed log1p transformation.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        Transformed array with sign(x) * log1p(|x|).
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def signed_softlog(x: jax.Array, alpha: float = EPSILON) -> jax.Array:
    """Apply signed soft logarithm transformation.

    Parameters
    ----------
    x : jax.Array
        Input array.
    alpha : float, optional
        Scaling parameter, by default EPSILON.

    Returns
    -------
    jax.Array
        Transformed array with signed soft log.
    """
    scaled = jnp.abs(x) / alpha
    y = jnp.log1p(scaled) / jnp.log1p(1.0 / alpha)
    y = jnp.sign(x) * y
    return y


def interpolate(x: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """Interpolate array to specified number of points.

    Parameters
    ----------
    x : jnp.ndarray
        Input array to interpolate.
    n_points : int
        Target number of points.

    Returns
    -------
    jnp.ndarray
        Interpolated array with n_points elements.
    """
    return jax.vmap(
        lambda xi: jnp.interp(
            jnp.linspace(0, 1, n_points), jnp.linspace(0, 1, len(xi)), xi
        )
    )(x)


def minmax_scale(x: jax.Array, min: jax.Array, max: jax.Array) -> jax.Array:
    """Scale array using min-max normalization.

    Parameters
    ----------
    x : jax.Array
        Input array to scale.
    min : jax.Array
        Minimum value for scaling.
    max : jax.Array
        Maximum value for scaling.

    Returns
    -------
    jax.Array
        Scaled array in range [0, 1].
    """
    normed = (x - min) / (max - min + EPSILON)
    # Do this to avoid floating point issues with jitting
    return jnp.clip(normed, min=None, max=1.0)


def log_stabilized_minmax_scale(
    x: jax.Array, global_min: jax.Array
) -> jax.Array:
    """Apply log-stabilized min-max scaling.

    Parameters
    ----------
    x : jax.Array
        Input array to scale.
    global_min : jax.Array
        Global minimum value for reference.

    Returns
    -------
    jax.Array
        Log-stabilized scaled array in range [0, 1].
    """
    p = x - global_min

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


# =============================================================================


def log_stabilized_minmax_scale_with_global_min(
    x: jax.Array, global_min: jax.Array
) -> jax.Array:
    """Apply log-stabilized min-max scaling with global minimum reference.

    Parameters
    ----------
    x : jax.Array
        Input array to scale.
    global_min : jax.Array
        Global minimum value for reference.

    Returns
    -------
    jax.Array
        Log-stabilized scaled array in range [0, 1].
    """
    p = x - global_min
    p_log10 = jnp.log10(p + EPSILON)
    normed = (p_log10 - jnp.log10(EPSILON)) / (
        jnp.max(p_log10, axis=0) - jnp.log10(EPSILON) + EPSILON
    )
    return jnp.clip(normed, 0.0, 1.0)


def traj_and_stair(x_output_mean, x_output_min, global_min):
    traj = ffill(bfill(x_output_mean))
    stair = output_cummin(ffill(bfill(x_output_min)))
    return log_stabilized_minmax_scale_with_global_min(
        jnp.stack((traj, stair), axis=1),
        global_min,
    )


# =============================================================================


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
    """Save model to disk using cloudpickle and equinox serialization.

    Parameters
    ----------
    model : eqx.Module
        Model to save.
    path : str | Path
        Path to save the model files.
    """
    _path = Path(path)
    model_shape = eqx.filter_eval_shape(lambda x: model, None)

    with open(_path.with_suffix(".pkl"), "wb") as f:
        cloudpickle.dump(model_shape, f)

    eqx.tree_serialise_leaves(_path.with_suffix(".eqx"), model)


def load_model(model_path: str, shape_path: str) -> eqx.Module:
    """Load model from disk.

    Parameters
    ----------
    model_path : str
        Path to the model weights file (.eqx).
    shape_path : str
        Path to the model shape file (.pkl).

    Returns
    -------
    eqx.Module
        Loaded model.
    """
    _model_path = Path(model_path)
    _shape_path = Path(shape_path)
    with open(_shape_path.with_suffix(".pkl"), "rb") as f:
        model_shape = cloudpickle.load(f)

    model = eqx.tree_deserialise_leaves(
        _model_path.with_suffix(".eqx"), model_shape
    )

    return model


def load_jnparray(path: str | Path) -> jnp.ndarray:
    """Load JAX numpy array from disk.

    Parameters
    ----------
    path : str | Path
        Path to the numpy file.

    Returns
    -------
    jnp.ndarray
        Loaded array (flattened).
    """
    # TODO: investigate this flatten behaviour
    return jnp.load(Path(path)).flatten()


def save_jnparray(model, path: str | Path) -> None:
    """Save JAX numpy array to disk.

    Parameters
    ----------
    model : jnp.ndarray
        Array to save.
    path : str | Path
        Path to save the array.
    """
    jnp.save(path, model)


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    """Save dataset dictionary to disk as npz file.

    Parameters
    ----------
    dataset : dict[str, jnp.ndarray]
        Dataset dictionary to save.
    path : str | Path
        Path to save the dataset.
    """
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def load_dataset(path: str | Path | None) -> dict[str, jnp.ndarray]:
    """Load dataset from npz file.

    Parameters
    ----------
    path : str | Path | None
        Path to the dataset file. If None, returns empty dict.

    Returns
    -------
    dict[str, jnp.ndarray]
        Loaded dataset dictionary.
    """
    if path is None:
        return {}
    loaded_data = jnp.load(Path(path).with_suffix(".npz"))
    return {k: jnp.array(v) for k, v in loaded_data.items()}
