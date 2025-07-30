"""
Module for pytree utilities.
"""

#                                                                       Modules
# =============================================================================

# Standard
import json
from functools import partial
from pathlib import Path
from typing import List, Tuple

import cloudpickle
# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
import optax
import xarray as xr
from f3dasm import ExperimentData
from jaxtyping import PRNGKeyArray, PyTree

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


def ffill(x: jnp.ndarray) -> jnp.ndarray:
    def scan_fn(carry, xi):
        carry = jnp.where(jnp.isnan(xi), carry, xi)
        return carry, carry

    _, filled = jax.lax.scan(scan_fn, jnp.nan, x)
    return filled


def bfill(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.flip(ffill(jnp.flip(x)))


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


def replace_nans_with_first_valid(x: jnp.ndarray) -> jnp.ndarray:
    is_not_nan = ~jnp.isnan(x)

    # Create a mask that identifies the first non-NaN
    cumsum = jnp.cumsum(is_not_nan)
    first_mask = cumsum == 1  # True only at first non-NaN
    first_valid_value = jnp.sum(
        jnp.where(first_mask, x, 0.0))  # only one non-zero

    # Handle all-NaN case
    any_valid = jnp.any(is_not_nan)
    first_valid_value = jnp.where(any_valid, first_valid_value, 1.0)

    return jnp.where(jnp.isnan(x), first_valid_value, x)


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
    x = ffill(bfill(x))
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
    x = ffill(bfill(x))
    x = log_stabilized_minmax_scale(x)
    return x


def encode(x: jax.Array, keys: jax.Array,
           default_value: int = 0) -> jax.Array:
    values = jnp.arange(len(keys), dtype=jnp.uint32)

    def map_fn(xi):
        matches = (keys == xi)
        match_idx = jnp.argmax(matches)
        return jnp.where(jnp.any(matches), values[match_idx], default_value)

    return jax.vmap(map_fn)(x)


def one_hot_encode(x: jax.Array, keys: jax.Array) -> jax.Array:
    # Make sure keys is sorted
    sorted_keys = jnp.sort(keys)

    indices = jnp.searchsorted(sorted_keys, x)
    indices_clipped = jnp.clip(indices, 0, sorted_keys.shape[0] - 1)
    matches = (x == sorted_keys[indices_clipped])
    one_hot_raw = jax.nn.one_hot(
        indices_clipped, num_classes=sorted_keys.shape[0], dtype=jnp.uint8)
    return one_hot_raw * matches[:, None]


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


def sigmoid_mean(x: jax.Array, a: float, b: float, c: float) -> jax.Array:
    return a / (1 + jnp.exp(-b * (x - c)))


def loss_fn(params: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    a, b, c = params
    y_pred = sigmoid_mean(x, a, b, c)
    return jnp.mean((y - y_pred) ** 2)


@partial(jax.jit, static_argnames=("max_iter", "tol"))
def single_run_jit(init_params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray,
                   max_iter: int, tol: float
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    fun = partial(loss_fn, x=x, y=y)
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    state = opt.init(init_params)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return (params, state)

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, 'count')
        grad = optax.tree_utils.tree_get(state, 'grad')
        err = optax.tree_utils.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    final_params, _ = jax.lax.while_loop(
        continuing_criterion, step, (init_params, state))
    final_loss = fun(final_params)
    return final_params, final_loss


@partial(jax.jit, static_argnames=("max_iter", "tol", "num_restarts"))
def fit_basis_function(y: jnp.ndarray, max_iter: int = 100, tol: float = 1e-6,
                       num_restarts: int = 5, *,
                       key: PRNGKeyArray) -> jnp.ndarray:
    x = jnp.linspace(0., 1., len(y))

    keys = jr.split(key, num_restarts)
    init_guesses = jax.vmap(lambda k: jr.uniform(
        k, (3,), minval=-1.0, maxval=1.0))(keys)

    def run_for_init(params):
        return single_run_jit(params, x, y, max_iter, tol)

    params_all, losses_all = jax.vmap(run_for_init)(init_guesses)

    best_index = jnp.argmin(losses_all)
    return params_all[best_index]


def save_eqx_model(model: eqx.Module, path: str):
    _path = Path(path)
    model_shape = eqx.filter_eval_shape(lambda x: model, None)

    with open(_path.with_suffix('.pkl'), 'wb') as f:
        cloudpickle.dump(model_shape, f)

    eqx.tree_serialise_leaves(_path.with_suffix('.eqx'), model)


def load_eqx_model(path: str) -> eqx.Module:
    _path = Path(path)

    with open(_path.with_suffix('.pkl'), 'rb') as f:
        model_shape = cloudpickle.load(f)

    return eqx.tree_deserialise_leaves(
        _path.with_suffix('.eqx'),
        like=model_shape)
