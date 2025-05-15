"""
Module containing functions that compute the performance of optimizers

Functions
---------

output_cummin
    Calculate the cumulative minimum of the given dataset.
scale
    Scale the given array using the minimum over realizations.
scaling
    Create a performance value for each problem and budget per optimizer.
add_random_samples_combinations
    Add random samples as optimizers to the optimization dataset.
calculate_performance_ratio
    Calculate the performance ratio for the given dataset.
create_performance_profile
    Create a performance profile for the given experiment data and budget.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import numpy as np
import xarray as xr

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================

EPSILON = 1e-8


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
    'budget'.
    """
    return xr.apply_ufunc(
        np.minimum.accumulate,
        da,
        input_core_dims=[['budget']],
        output_core_dims=[['budget']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )


def scaling2(data: xr.Dataset) -> xr.DataArray:

    # 1. Calculate cumulative minimum (quality indictor value)
    ds_output = output_cummin(data.output_min)  # .chunk('auto')

    # Cap the quality indicator value at 1e-8
    ds_output = ds_output.where(
        (ds_output > EPSILON) | (ds_output <= 0.) | ds_output.isnull(),
        other=EPSILON)

    # 2. Make sure the loss is non-negative and non-zero
    group_by_realizations = ds_output.groupby('itemID')

    def min_custom(x):
        # 3. Make sure the quality indicator value is non-negative
        p = (x - x.min())
        # 4. Add epsilon to make quality indicator value non-zero
        # the 1e-2 is to prevent really big differences in quality values
        eps = p.where(p > 0.0, p.max()+EPSILON).min() + EPSILON
        x_log10 = np.log10(p + eps)

        # 5. Min-max scaling of the data
        x_min = x_log10.min()
        x_max = x_log10.max(['budget'])
        return (x_log10 - x_min) / (x_max - x_min + EPSILON)
    ds_log10_positive = group_by_realizations.map(
        min_custom)  # .assign_coords({
    #    'problem': ds_output.problem})

    return ds_log10_positive


def scaling3(data: xr.Dataset) -> xr.DataArray:

    # 1. Calculate cumulative minimum (quality indictor value)
    ds_output = output_cummin(data.output_min)  # .chunk('auto')

    # Cap the quality indicator value at 1e-8
    ds_output = ds_output.where(
        (ds_output > EPSILON) | (ds_output <= 0.) | ds_output.isnull(),
        other=EPSILON)

    def f(x):
        p = (x - x.min())
        eps = (np.where(p > 0.0, p, p.max()+EPSILON)).min() + EPSILON
        x_log10 = np.log10(p + eps)

        # 5. Min-max scaling of the data
        x_min = x_log10.min()
        x_max = x_log10.max(axis=0)
        return (x_log10 - x_min) / (x_max - x_min + EPSILON)

    data_scaled = xr.apply_ufunc(
        f,
        ds_output,
        input_core_dims=[['budget', 'optimizer', 'output_dim']],
        output_core_dims=[['budget', 'optimizer', 'output_dim']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[ds_output.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}

    )

    return data_scaled.transpose(
        'optimizer', 'output_dim', 'problem', 'budget')
