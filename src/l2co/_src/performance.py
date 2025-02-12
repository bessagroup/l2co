"""
Module containing functions that compute the performance of optimizers

Functions
---------

output_cummin
    Calculate the cumulative minimum of the given dataset.
scale
    Scale the given array using the minimum over realizations.
scaling
    Create a performance value for each problem and iterations per optimizer.
add_random_samples_combinations
    Add random samples as optimizers to the optimization dataset.
calculate_performance_ratio
    Calculate the performance ratio for the given dataset.
create_performance_profile
    Create a performance profile for the given experiment data and iterations.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from typing import List, Optional

# Third-party
import numpy as np
import xarray as xr

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def output_cummin(da: xr.DataArray):
    """
    Calculate the cumulative minimum of the given data array.

    Parameters
    ----------
    da : xarray.DataArray
        The dataarray to calculate the cumulative minimum for.

    Returns
    -------
    xr.DataArray
        The dataarray with the cumulative minimum calculated.

    Notes
    -----
    The dataarray calculate the cumulative minimum for the coordinate named
    'iterations'.
    """
    return xr.apply_ufunc(
        np.minimum.accumulate,
        da,
        input_core_dims=[['iterations']],
        output_core_dims=[['iterations']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )


def scaling2(data: xr.Dataset) -> xr.DataArray:

    # 1. Calculate cumulative minimum (quality indictor value)
    ds_output = output_cummin(data.output)  # .chunk('auto')

    # Cap the quality indicator value at 1e-8
    ds_output = ds_output.where(
        (ds_output > 1e-8) | ds_output.isnull(), other=1e-8)

    # 2. Make sure the loss is non-negative and non-zero
    group_by_realizations = ds_output.groupby('itemID')

    def min_custom(x):
        # 3. Make sure the quality indicator value is non-negative
        p = (x - x.min())
        # 4. Add epsilon to make quality indicator value non-zero
        eps = p.where(p > 0.0, p.max()+1e-16).min()
        x_log10 = np.log10(p + eps)

        # 5. Min-max scaling of the data
        x_min = x_log10.min()
        x_max = x_log10.max(['iterations'])
        return (x_log10 - x_min) / (x_max - x_min + 1e-16)

    ds_log10_positive = group_by_realizations.map(min_custom).assign_coords({
        'problem': ds_output.problem})

    return ds_log10_positive

#                                                           Performance profile
# =============================================================================


def calculate_performance_ratio(
        ds_budget: xr.DataArray,
        best_optimizer_each_problem: xr.DataArray,
        max_f_value: float = 3.0) -> xr.DataArray:
    """
    Calculate the performance ratio for the given dataset.

    Parameters
    ----------

    ds_budget : xarray.DataArray
        The dataset to calculate the performance ratio for.

    best_optimizer_each_problem : xarray.DataArray
        The best optimizer for each problem.

    max_f_value : float, optional
        The maximum value of f, by default 10.0

    Returns
    -------
    performance_profile : xarray.DataArray
        The performance profile for the given dataset.
    """
    # Calculate the performance ratio rij
    rij = ds_budget / best_optimizer_each_problem

    # Create a logarithmic scale for f
    # combine rij and f in a dataset
    ds_f = xr.Dataset({'rij': rij, 'f': xr.DataArray(
        np.logspace(np.log10(1.0), np.log10(max_f_value), 1000), dims='f')})

    performance_profile = np.less_equal(ds_f, ds_f.f+1e-4).where(
        ~ds_f.isnull(), np.nan).mean(dim='problem').rij
    return performance_profile


def create_performance_profile(
        ds_raw: xr.Dataset,
        iterations: int,
        strategies: Optional[List[xr.DataArray]] = None) -> xr.Dataset:
    """
    Create a performance profile for the given experiment data and iterations.

    Parameters
    ----------
    ds_raw : xr.Dataset
        The optimization dynamics data for the given experiment data.
    iterations : int
        The total budget for each of the optimization problems
    strategies : List[xr.DataArray], optional
        The strategies to include in the performance profile, by default None

    Returns
    -------
    performance_profile : xr.Dataset
        The performance profile for the given experiment data and iterations.
        The variables are the different static optimizers, and the optional
        added strategies.

    Notes
    -----
    Strategies are optional, and can be added to the performance profile
    by passing a list of xr.DataArray objects. The xr.DataArray objects
    should have a 'problem' coordinate which is identical to the data provided
    in the ds_raw parameter.
    """
    # Select only the output variable
    ds_output = ds_raw.isel(output_dim=0, drop=True).output

    # Calculat the best found value (cummin)
    ds_output = output_cummin(ds_output)

    ds_positive = ds_output - \
        ds_output.min(dim=['optimizer', 'iterations']).where(
            ds_output.min(dim=['optimizer', 'iterations']
                          ) < 0, 0.0) + 1e-17

    # Slice at the given total number of iterations
    ds_budget = ds_positive.isel(iterations=iterations-1)
    ds_budget = ds_budget.sortby('problem')

    # Calculate the performance profile
    performance_profile = calculate_performance_ratio(
        ds_budget, ds_budget.min(dim='optimizer'))

    pp_to_dataset = performance_profile.to_dataset('optimizer')

    if strategies is not None:
        for strategy in strategies:
            _strategy = strategy.sortby('problem')
            pp_to_dataset[strategy.name] = calculate_performance_ratio(
                ds_budget.sel(optimizer=_strategy), ds_budget.min(
                    dim='optimizer')
            )

    return pp_to_dataset
