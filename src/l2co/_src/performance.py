"""
Module containing functions that compute the performance of optimizers

Functions
---------

get_performance
    Generate a dataset that captures the performance of an optimizer
    on a benchmark function as function of the number of function evaluations.
query_t
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.
is_sparse
    Check if the performance DataArray is sparse.
collaborative_filtering
    Fill in missing values in the performance DataFrame using
    collaborative filtering.
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

INVALID_VALUE = 1.0

# =============================================================================


def output_cummin(da: xr.DataArray):
    """
    Calculate the cumulative minimum of the given dataset.

    Parameters
    ----------
    ds : xr.DataArray
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
    Scale the data to be used as input for the RTD algorithm.

    Parameters
    ----------
    data : xr.Dataset
        The data to be scaled.

    Returns
    -------
    xr.DataArray
        The scaled data.
    """
    # 1. Calculate cumulative minimum (quality indictor value)
    data['output_cummin'] = output_cummin(data.output)

    # 2. Make sure the loss is non-negative
    y_min = data.output_cummin.min(dim=['optimizer', 'iterations']).where(
        data.output_cummin.min(dim=['optimizer', 'iterations']) < 0.0, 0.0)
    ds_positive = data['output_cummin'] - y_min.min('realization')

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
    # scaler = MinMaxScaler()

    def scale(arr, minimum_over_realizations):
        scaler = MinMaxScaler()
        scaler.fit(minimum_over_realizations.reshape(-1, 1))
        scaler.partial_fit(arr.flatten().reshape(-1, 1))
        return np.reshape(
            scaler.transform(arr.flatten().reshape(-1, 1)), arr.shape)

    return xr.apply_ufunc(
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
