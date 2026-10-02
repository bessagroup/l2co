"""
Module for pytree utilities.
"""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp

# The population loss evaluators live in ``l2co_optimizers`` (l2co ADR
# 0016); re-exported so ``l2co.tree_utils.vmapped_loss`` keeps working.
from l2co_optimizers import (  # noqa: F401 -- re-exported
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
    vmapped_loss_with_rng,
)

# Local
from .performance import EPSILON

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# def ffill(x: jnp.ndarray) -> jnp.ndarray:
#     """Forward fill NaN values in array.

#     Parameters
#     ----------
#     x : jnp.ndarray
#         Input array with potential NaN values.

#     Returns
#     -------
#     jnp.ndarray
#         Array with NaN values forward filled.
#     """

#     def scan_fn(carry, xi):
#         carry = jnp.where(jnp.isnan(xi), carry, xi)
#         return carry, carry

#     _, filled = jax.lax.scan(scan_fn, jnp.nan, x)
#     return filled


def ffill(x: jnp.ndarray) -> jnp.ndarray:
    """Forward fill NaN values in a 2D array along axis 0."""

    def scan_fn(carry, xi):
        """Replace NaN entries in xi with the carry value."""
        new_carry = jnp.where(jnp.isnan(xi), carry, xi)
        return new_carry, new_carry

    # Initialize carry with NaNs matching the inner dimension
    init_carry = jnp.full(x.shape[1:], jnp.nan)

    _, filled = jax.lax.scan(scan_fn, init_carry, x)
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


def get_interpolation_matrix(input_len: int, target_len: int):
    """Computes a static matrix M such that x_interp = x @ M."""
    # The positions we want to sample
    query_points = jnp.linspace(0, 1, target_len)
    # The positions we actually have
    ref_points = jnp.linspace(0, 1, input_len)

    # Calculate indices and weights for linear interpolation
    # This logic mimics jnp.interp but produces a reusable matrix
    def get_weights(q):
        """Build one row of the interpolation matrix for query point q."""
        idx = jnp.searchsorted(ref_points, q, side="right") - 1
        idx = jnp.clip(idx, 0, input_len - 2)

        low = ref_points[idx]
        high = ref_points[idx + 1]

        weight_high = (q - low) / (high - low)
        weight_low = 1.0 - weight_high

        # Create a row for the matrix
        row = jnp.zeros((input_len,))
        row = row.at[idx].set(weight_low)
        row = row.at[idx + 1].set(weight_high)
        return row

    return jax.vmap(get_weights)(query_points).T


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
    min_value = jnp.log10(EPSILON)
    max_value = p_log10[0]
    # max_value = jnp.max(p_log10, axis=0)
    normed = (p_log10 - min_value) / (max_value - min_value + EPSILON)
    return jnp.clip(normed, 0.0, 1.0)


def traj_and_stair(x_output_mean, x_output_min, global_min):
    """
    Process trajectory and staircase from optimization outputs.

    Parameters
    ----------
    x_output_mean : jax.Array
        Mean output trajectory with shape (n_steps, n_features).
    x_output_min : jax.Array
        Minimum output trajectory with shape (n_steps, n_features).
    global_min : jax.Array
        Global minimum value for scaling reference.

    Returns
    -------
    jax.Array
        Stacked and scaled trajectory and staircase arrays with
        shape (n_steps, 2, n_features).

    Notes
    -----
    Applies forward-backward fill to handle missing values,
    computes cumulative minimum for staircase, and applies
    log-stabilized min-max scaling.
    """
    traj = ffill(bfill(x_output_mean))
    stair = output_cummin(ffill(bfill(x_output_min)))
    y = log_stabilized_minmax_scale_with_global_min(
        jnp.stack((traj, stair), axis=1),
        global_min,
    )
    return ffill(bfill(y))


#                                                  Loading and saving functions
# =============================================================================


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
