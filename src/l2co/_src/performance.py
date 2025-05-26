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


def output_cummin(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the cumulative minimum of the given data array.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to calculate the cumulative minimum for.

    Returns
    -------
    xarray.DataArray
        The data array with the cumulative minimum calculated.

    Notes
    -----
    The cumulative minimum is calculated along the coordinate named 'budget'.
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


def scaling2(data: xr.DataArray) -> xr.DataArray:
    """
    Scale the given data array using a custom scaling method.

    Parameters
    ----------
    data : xarray.DataArray
        The data array to scale.

    Returns
    -------
    xarray.DataArray
        The scaled data array.

    Notes
    -----
    This function applies log scaling, ensures non-negativity, and performs
    min-max scaling for each realization in the data.
    """
    # Cap the quality indicator value at 1e-8
    ds_output = data.where(
        (data > EPSILON) | (data <= 0.) | data.isnull(),
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
        min_custom)

    return ds_log10_positive


def scaling3(data: xr.DataArray) -> xr.DataArray:
    """
    Scale the given data array using an alternative scaling method.

    Parameters
    ----------
    data : xarray.DataArray
        The data array to scale.

    Returns
    -------
    xarray.DataArray
        The scaled data array.

    Notes
    -----
    This function applies log scaling, ensures non-negativity, and performs
    min-max scaling across multiple dimensions.
    """

    # Cap the quality indicator value at 1e-8
    ds_output = data.where(
        (data > EPSILON) | (data <= 0.) | data.isnull(),
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
