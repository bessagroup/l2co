"""
Module containing functions that compute the performance of optimizers

Functions
---------

output_cummin
    Calculate the cumulative minimum of the given dataset.
query_t
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.
scaling
    Create a performance value for each problem and iterations per optimizer.
add_random_samples_combinations
    Add random samples as optimizers to the optimization dataset.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

# Local
from ._logging import logger

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


def query_t(da_performance: xr.DataArray, t: int) -> xr.DataArray:
    """
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.

    Parameters
    ----------

    performance_dataset : xarray.Dataset
        xarray Dataset object containing a floating point value for
        the performance of the optimizers on benchmark functions as a function
        of the number of function evaluations.
    t : int
        number of function evaluations

    Returns
    -------
    xarray.DataArray
        xarray DataArray containing the performance of the optimizers
        after a given number of function evaluations.

    Raises
    ------
    ValueError
        If the number of iterations is larger than the maximum number of
        iterations in the performance dataset.
    """
    if t > da_performance.iterations.max():
        raise ValueError(
            f'Number of iterations {t} is larger than the maximum number of '
            f'iterations in the performance dataset '
            f'{da_performance.iterations.max().values}')

    da_performance: xr.DataArray = da_performance.sel(
        iterations=t, drop=True)

    logger.info(f"Query t={t} from performance dataset")

    return da_performance


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
    def scale(
            arr: np.ndarray,
            minimum_over_realizations: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        scaler.fit(minimum_over_realizations.reshape(-1, 1))
        scaler.partial_fit(arr.flatten().reshape(-1, 1))
        return np.reshape(
            scaler.transform(arr.flatten().reshape(-1, 1)), arr.shape)

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
