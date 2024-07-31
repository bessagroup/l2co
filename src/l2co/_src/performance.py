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
from sklearn.preprocessing import MinMaxScaler

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
    da : xr.DataArray
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


def scale(
        arr: np.ndarray,
        minimum_over_realizations: np.ndarray) -> np.ndarray:
    """
    Scale the given array using the minimum over realizations.

    Parameters
    ----------
    arr : np.ndarray
        The array to scale.
    minimum_over_realizations : np.ndarray
        The minimum over realizations.

    Returns
    -------
    np.ndarray
        The scaled array.
    """
    scaler = MinMaxScaler()
    scaler.fit(minimum_over_realizations.reshape(-1, 1))
    scaler.partial_fit(arr.flatten().reshape(-1, 1))
    return np.reshape(
        scaler.transform(arr.flatten().reshape(-1, 1)), arr.shape)


def scaling(data: xr.Dataset) -> xr.DataArray:
    """
    Create a performance value for each problem and iterations per optimizer.

    Parameters
    ----------
    data : xr.Dataset
        The optimization dataset to be evaluated

    Returns
    -------
    xr.DataArray
        The performance values
    """

    # 0. unstack the data
    data_ = data.unstack('problem')

    # 1. Calculate cumulative minimum (quality indictor value)
    data_['output_cummin'] = output_cummin(data_.output)

    # 2. Make sure the loss is non-negative
    y_min = data_.output_cummin.min(dim=['optimizer', 'iterations']).where(
        data_.output_cummin.min(dim=['optimizer', 'iterations']) < 0.0, 0.0)
    ds_positive = data_['output_cummin'] - y_min.min('realization')

    # 3. Make sure the loss is non-zero
    ds_add_to_become_non_zero = (
        ds_positive.min(dim=['optimizer', 'iterations']
                        ).min('realization') == 0.0).where(
        (~(ds_positive.min(dim=['optimizer', 'iterations']
                           ).min('realization') == 0.0)
         ), other=ds_positive.where(
             ds_positive > 0.0).min(
                 dim=['optimizer', 'iterations']).min('realization')
    )

    ds_positive += ds_add_to_become_non_zero

    # 4. Take the base 10 logarithm of the data
    ds_log10_positive = np.log10(ds_positive)

    # 5. Scale the data

    performance = xr.apply_ufunc(
        scale, ds_log10_positive,
        ds_log10_positive.min(
            dim=['optimizer', 'iterations', 'output_dim', 'realization']),
        input_core_dims=[
            ['optimizer', 'iterations', 'output_dim'],
            []],
        output_core_dims=[['optimizer', 'iterations', 'output_dim']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    return performance.stack(
        problem=('realization', 'itemID')).sel(
            problem=data.problem, output_dim='y', drop=True)


def add_random_samples_combinations(
        data: xr.Dataset, data_samples: xr.Dataset,
        iterations: int, t_switch: int) -> xr.Dataset:
    """
    Add random samples as optimizers to the optimization dataset.

    Parameters
    ----------
    data : xr.Dataset
        The performance dataset.
    data_samples : xr.Dataset
        The dataset with the random samples.
    iterations : int
        The total number of iterations.
    t_switch : int
        The number of iterations to switch from random samples to the
        optimization data.

    Returns
    -------
    xr.Dataset
        The performance dataset with the random samples optimizers added.
    """
    a = []

    opts = [entry for entry in data.optimizer.values if '+' not in entry]

    for opt in opts:
        a.append(xr.concat([data_samples.sel(
            iterations=np.arange(t_switch)).output, data.sel(
            optimizer=opt, iterations=np.arange(iterations-t_switch)
        ).assign_coords(iterations=np.arange(t_switch, iterations)).output],
            dim='iterations').assign_coords(optimizer=f"random_samples+{opt}"))

    random_combinations = xr.concat(a, dim='optimizer')

    return xr.concat(
        [data.output, random_combinations], dim='optimizer').to_dataset()


#                                                           Performance profile
# =============================================================================

def calculate_performance_ratio(
        ds_budget: xr.DataArray,
        best_optimizer_each_problem: xr.DataArray,
        max_f_value: float = 10.0) -> xr.DataArray:
    """
    Calculate the performance ratio for the given dataset.

    Parameters
    ----------

    ds_budget : xr.DataArray
        The dataset to calculate the performance ratio for.

    best_optimizer_each_problem : xr.DataArray
        The best optimizer for each problem.

    max_f_value : float, optional
        The maximum value of f, by default 10.0

    Returns
    -------
    performance_profile : xr.DataArray
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
    ds_output = ds_raw.sel(output_dim='y', drop=True).output

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
