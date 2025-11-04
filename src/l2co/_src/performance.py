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

from .naming import BUDGET, OUTPUT_DIM, SCHEDULE, TASK_ID

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
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
        input_core_dims=[[BUDGET]],
        output_core_dims=[[BUDGET]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def cap_value(data: xr.DataArray, cap: float = EPSILON) -> xr.DataArray:
    """
    Cap the values in the data array at a specified threshold. (0 < data < cap)

    Parameters
    ----------
    data : xarray.DataArray
        The data array to cap.
    cap : float, optional
        The threshold value to cap the data at, by default EPSILON.

    Returns
    -------
    xarray.DataArray
        The capped data array.
    """
    return data.where((data >= cap) | (data <= 0.0) | data.isnull(), other=cap)


def min_custom(x: xr.DataArray) -> xr.DataArray:
    p = x - np.nanmin(x)
    eps = (np.where(p > 0.0, p, np.nanmax(p) + EPSILON)).min() + EPSILON
    x_log10 = np.log10(p + eps)
    # 5. Min-max scaling of the data
    x_min = np.nanmin(x_log10)
    x_max = np.nanmax(x_log10)
    return (x_log10 - x_min) / (x_max - x_min + EPSILON)


# =============================================================================


def scale_by_task(data: xr.DataArray) -> xr.DataArray:
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
    # Cap the loss value at 1e-8
    ds_output = cap_value(data, cap=EPSILON)

    ds_log10_positive = ds_output.groupby(TASK_ID).map(min_custom)

    return ds_log10_positive


def scale_by_problem_and_schedule(data: xr.DataArray) -> xr.DataArray:
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
    ds_output = cap_value(data, cap=EPSILON)

    data_scaled = xr.apply_ufunc(
        min_custom,
        ds_output,
        input_core_dims=[[BUDGET, SCHEDULE, OUTPUT_DIM]],
        output_core_dims=[[BUDGET, SCHEDULE, OUTPUT_DIM]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds_output.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    return data_scaled


def scale_by_problem(data: xr.DataArray) -> xr.DataArray:
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
    ds_output = cap_value(data, cap=EPSILON)

    data_scaled = xr.apply_ufunc(
        min_custom,
        ds_output,
        input_core_dims=[[BUDGET, OUTPUT_DIM]],
        output_core_dims=[[BUDGET, OUTPUT_DIM]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds_output.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    return data_scaled
