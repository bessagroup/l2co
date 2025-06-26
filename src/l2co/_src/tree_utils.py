"""
Module for pytree utilities.
"""

#                                                                       Modules
# =============================================================================

# Standard
import json
from typing import List

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp
import xarray as xr
from f3dasm import ExperimentData
from jaxtyping import PyTree

# Local
from .naming import BUDGET, INPUT_DIM, OUTPUT_DIM, UPDATE_STEP
from .optimizerschedule import OptimizationSchedule
from .performance import EPSILON

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def combined_loss(params, loss_fn,
                  static: PyTree,
                  **rng_sample: dict) -> jax.Array:
    """
    Compute the combined loss for a given set of parameters.

    Parameters
    ----------
    params : PyTree
        The parameters to evaluate the loss for.
    loss_fn : Callable
        The loss function to compute the loss.
    static : PyTree
        Static parameters to combine with the input parameters.
    rng_sample : dict
        Random samples used in the loss computation.

    Returns
    -------
    jax.Array
        The computed loss.
    """
    return loss_fn(eqx.combine(params, static), **rng_sample)


def replace_nans_with_max_or_one(x: jax.Array) -> jax.Array:
    """
    Replace NaN values in the array with the maximum value or 1.0.

    Parameters
    ----------
    x : jax.Array
        Input array with potential NaN values.

    Returns
    -------
    jax.Array
        Array with NaN values replaced.
    """
    is_nan = jnp.isnan(x)
    non_nan_vals = jnp.where(is_nan, -jnp.inf, x)
    max_val = jnp.max(non_nan_vals)
    max_val = jnp.where(jnp.isneginf(max_val), 1.0, max_val)
    return jnp.where(is_nan, max_val, x)


def log_stabilized_minmax_scale(x: jax.Array) -> jax.Array:
    """
    Apply log-stabilized min-max scaling to the input array.

    Parameters
    ----------
    x : jax.Array
        Input array to scale.

    Returns
    -------
    jax.Array
        Scaled array.
    """
    p = x - jnp.min(x)
    eps = jnp.min(jnp.where(p > 0.0, p, jnp.max(p) + EPSILON)) + EPSILON
    x_log10 = jnp.log10(p + eps)
    x_min = jnp.min(x_log10)
    x_max = jnp.max(x_log10)
    return (x_log10 - x_min) / (x_max - x_min + EPSILON)


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


def staircase(x: jax.Array) -> jnp.ndarray:
    """
    Generate a staircase representation of the input array.

    Parameters
    ----------
    x : jax.Array
        Input array.

    Returns
    -------
    jnp.ndarray
        Staircase-transformed array.
    """
    x = replace_nans_with_max_or_one(x)
    x = output_cummin(x)
    x = log_stabilized_minmax_scale(x)
    return x


def trajectory(x: jax.Array) -> jnp.ndarray:
    """
    Generate a trajectory representation of the input array.

    Parameters
    ----------
    x : jax.Array
        Input array.

    Returns
    -------
    jnp.ndarray
        Trajectory-transformed array.
    """
    x = replace_nans_with_max_or_one(x)
    x = log_stabilized_minmax_scale(x)
    return x


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
    best_params = jax.tree.map(lambda x: x[idx][None], params)
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


def experimentdata_to_xarray(data: ExperimentData,
                             popsizes: List[int],
                             budgets: List[float] | float,
                             n_iterations: int,
                             include_input: bool = False) -> xr.Dataset:
    """
    Convert ExperimentData to an xarray.Dataset.

    Parameters
    ----------
    data : ExperimentData
        The experiment data to convert.
    popsizes : list of int
        List of population sizes used in the experiment.
    budgets : list of float or float
        List of budgets or a single budget value.
    n_iterations : int
        Number of iterations in the experiment.
    include_input : bool, optional
        Whether to include input data in the dataset, by default False.

    Returns
    -------
    xarray.Dataset
        The converted dataset.
    """
    y_arr = jnp.vstack([es.output_data['y']
                        for _, es in data]).T

    opt_step_arr = onp.vstack([es.output_data['opt_step']
                               for _, es in data]).T

    y = xr.DataArray(
        y_arr,
        dims=[OUTPUT_DIM, BUDGET],
        coords={OUTPUT_DIM: ['y']})

    optimizer = xr.DataArray(
        opt_step_arr,
        dims=[OUTPUT_DIM, BUDGET],
        coords={OUTPUT_DIM: ['y']})

    if include_input:
        x_arr = jnp.vstack([es.input_data['x']
                            for _, es in data]).T

        x = xr.DataArray(
            x_arr,
            dims=[INPUT_DIM, BUDGET],
            coords={INPUT_DIM: [f"x{i}" for i in range(x_arr.shape[0])]})

        ds = xr.Dataset({'input': x,
                         'output': y,
                         UPDATE_STEP: optimizer,
                         })

    else:
        ds = xr.Dataset({
            'output': y,
            UPDATE_STEP: optimizer,
        })

    ds.attrs['popsizes'] = popsizes
    ds.attrs['budgets'] = budgets
    ds.attrs['n_iterations'] = n_iterations
    return ds


def schedule_to_xarray(data: OptimizationSchedule) -> xr.DataArray:
    """
    Convert an OptimizationSchedule to an xarray.DataArray.

    Parameters
    ----------
    data : OptimizationSchedule
        The optimization schedule to convert.

    Returns
    -------
    xarray.DataArray
        The converted data array.
    """
    x = xr.DataArray(
        [json.dumps(data.to_dict())],
        dims=['schedule']
    )
    return x


def tree_flatten_population_dim(params: PyTree) -> PyTree:
    """
    Flatten the population dimension of a PyTree.

    Parameters
    ----------
    params : PyTree
        The PyTree with a population dimension to flatten.

    Returns
    -------
    PyTree
        The PyTree with the population dimension flattened.
    """
    def flatten_fn(x):
        # Get the shape of the parameter
        shape = x.shape
        # Assuming the shape is (batch_size, population_dim, *other_dims)
        batch_size, population_dim, *other_dims = shape
        # Reshape to (batch_size * population_dim, *other_dims)
        return x.reshape((batch_size * population_dim, *other_dims))

    # Apply flatten_fn to each element of the parameter tree
    return jax.tree_util.tree_map(flatten_fn, params)


def tree_to_dict_list(tree: PyTree, name: str) -> list[dict[str, PyTree]]:
    """
    Convert a PyTree into a list of dictionaries.

    Parameters
    ----------
    tree : PyTree
        The PyTree to convert.
    name : str
        The key name to use in the dictionaries.

    Returns
    -------
    list of dict
        A list of dictionaries representing the PyTree.
    """
    num_models = jax.tree_util.tree_leaves(
        tree)[0].shape[0]  # Get the batch size

    return [{name:
             jax.tree_util.tree_map(lambda param: param[i], tree)}
            for i in range(num_models)]


def dict_list_to_tree(dict_list: list[dict[str, PyTree]], name: str) -> PyTree:
    """
    Convert a list of dictionaries into a PyTree.

    Parameters
    ----------
    dict_list : list of dict
        The list of dictionaries to convert.
    name : str
        The key name to extract from the dictionaries.

    Returns
    -------
    PyTree
        The resulting PyTree.
    """
    return jax.tree_util.tree_map(
        lambda *params: jax.numpy.stack(params, axis=0),
        *[d[name] for d in dict_list])
