# Standard
import json
from typing import List

# Third-party
import jax
import jax.numpy as jnp
import xarray as xr
from f3dasm import ExperimentData
from jaxtyping import PyTree

# Local
from .optimizerschedule import OptimizationSchedule

# =============================================================================


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

    y = xr.DataArray(
        y_arr,
        dims=['output_dim', 'budget'],
        coords={'output_dim': ['y']})

    if include_input:
        x_arr = jnp.vstack([es.input_data['x']
                            for _, es in data]).T

        x = xr.DataArray(
            x_arr,
            dims=['input_dim', 'budget'],
            coords={'input_dim': [f"x{i}" for i in range(x_arr.shape[0])]})

        ds = xr.Dataset({'input': x,
                         'output': y,
                         })

    else:
        ds = xr.Dataset({
            'output': y,
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
