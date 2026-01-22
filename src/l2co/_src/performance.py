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

from typing import Literal

# Third-party
import numpy as np
import xarray as xr

from .naming import (
    BUDGET,
    COST,
    GLOBAL_MIN,
    OUTPUT_DIM,
    OUTPUT_MIN,
    PROBLEM,
    REALIZATION,
    SCHEDULE,
    SEED,
    SUCCESS,
    TARGET,
    TASK_ID,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================

EPSILON = 1e-8
DEFAULT_COST_VALUES = np.logspace(-1, 6, 100) ** 0.8
DEFAULT_TARGETS = np.linspace(0.0, 1.0, 51)


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


def adaptive_worst_quantile_1d(x, q=0.99):
    T = x.shape[0]
    out = np.empty(T, dtype=x.dtype)
    ks = np.ceil(np.log(np.arange(T) + 3) ** 2 / 2).astype(int)

    for t, k in enumerate(ks):
        start = max(0, t - k + 1)
        w = x[start : t + 1]
        idx = int(np.ceil(q * (w.size - 1)))
        out[t] = np.partition(w, idx)[idx]

    return out


def output_quantile(da: xr.DataArray) -> xr.DataArray:
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
        adaptive_worst_quantile_1d,
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


# def min_custom(x: xr.DataArray) -> xr.DataArray:
#     p = x - np.nanmin(x)
#     eps = (np.where(p > 0.0, p, np.nanmax(p) + EPSILON)).min() + EPSILON
#     x_log10 = np.log10(p + eps)
#     # 5. Min-max scaling of the data
#     x_min = np.nanmin(x_log10)
#     x_max = np.nanmax(x_log10)
#     return (x_log10 - x_min) / (x_max - x_min + EPSILON)


def min_custom(ds: xr.Dataset) -> xr.DataArray:
    """
    Applies custom log scaling to 'output_min' based on the presence
    of 'global_min' within the dataset slice.
    """
    x = ds[OUTPUT_MIN]
    g_min = ds[GLOBAL_MIN]

    # 1. Determine p based on whether global_min is NaN
    # We check if all values in g_min are NaN (assuming one g_min per task)
    if g_min.isnull().all():
        p = x - x.min(skipna=True)
    else:
        p = x - g_min

    # 2. Calculate epsilon for the log shift
    # We find the smallest positive value to avoid log(0)
    p_positive = p.where(p > 0)
    eps = p_positive.min(skipna=True) + EPSILON

    # 3. Apply log10 scaling
    x_log10 = np.log10(p + eps)

    # 4. Min-max scaling
    x_min = x_log10.min(skipna=True)
    x_max = x_log10.max(skipna=True)

    return (x_log10 - x_min) / (x_max - x_min + EPSILON)


# =============================================================================


def scale_by_task(
    data: xr.DataArray, global_min: xr.DataArray
) -> xr.DataArray:
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
    ds_output = xr.Dataset(
        {OUTPUT_MIN: cap_value(data, cap=EPSILON), GLOBAL_MIN: global_min}
    )

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

    # Cap the quality indicator value at EPSILON
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


# =============================================================================


def performance_profile(
    performance: xr.DataArray, max_f_value: int = 2.0
) -> xr.DataArray:
    """
    Create a performance profile for the given performance data.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data for optimization problems.
    max_f_value : int, optional
        Maximum performance ratio, by default 2.0.

    Returns
    -------
    xr.DataArray
        Performance profile.
    """

    p = performance.isel({BUDGET: -1})
    rij = (p + 1.0) / 1.0

    ds_f = xr.Dataset(
        {
            "performance_profile": rij,
            "relative_error": xr.DataArray(
                np.logspace(np.log10(1.0), np.log10(max_f_value), 1000),
                dims="relative_error",
            ),
        }
    )

    performance_profile: xr.DataArray = (
        np.less_equal(ds_f, ds_f.relative_error + 1e-2)
        .where(~ds_f.isnull(), np.nan)
        .mean(dim=PROBLEM)["performance_profile"]
    )

    rel_error = (performance_profile["relative_error"] * 100) - 100

    return performance_profile.assign_coords({"relative_error": rel_error})


def rtd(performance: xr.DataArray) -> xr.DataArray:
    """
    Compute the runtime distribution (RTD) profile.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data for optimization problems.

    Returns
    -------
    xr.DataArray
        RTD profile.
    """

    return (1.0 - performance).mean(PROBLEM)


def ertd(
    performance: xr.Dataset,
    cost_measure: Literal["fevals", "iterations"],
    dims: xr.DataArray,
    cost_values: np.ndarray = DEFAULT_COST_VALUES,
) -> xr.DataArray:
    targets = xr.DataArray(DEFAULT_TARGETS, dims=TARGET)
    solved = performance[OUTPUT_MIN] <= targets
    success = solved.any(BUDGET)
    first_solved = solved.argmax(BUDGET).where(
        success, other=solved[BUDGET].argmax()
    )

    ds = xr.Dataset(
        {
            COST: performance[cost_measure].isel({BUDGET: first_solved}),
            SUCCESS: success,
            "dims": dims,
        }
    )

    ert_multi = compute_ert_multi_seed(ds, n_seeds=50, n_picks=50)

    values = np.unique(cost_values)

    budgets = xr.DataArray(
        values,
        dims="cost_per_dim",
        coords={"cost_per_dim": values},
    )
    ertd = (ert_multi <= budgets).mean([OUTPUT_DIM, TARGET])

    return ertd


def draw_samples(seed: int, n_realizations: int, n_picks: int):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.permutation(n_realizations + 1),
            rng.integers(
                0,
                n_realizations + 1,
                size=max(n_picks - n_realizations, 0),
            ),
        ]
    )


def compute_ert_group(
    ds: xr.Dataset,
    samples: xr.DataArray,
) -> xr.DataArray:
    ds = ds.isel({PROBLEM: samples})

    # Find index of first True along "problem"
    first_true = ds[SUCCESS].argmax(PROBLEM)

    # Handle case where there is no True at all
    first_true = first_true.where(ds[SUCCESS].any(PROBLEM), other=-1)

    # Coordinate index
    idx = xr.DataArray(np.arange(ds.sizes[PROBLEM]), dims=PROBLEM)

    # Mask up to first True
    mask = (idx <= first_true).fillna(False)

    # Sum cost up to first True
    ert = ds[COST].where(mask, 0).sum(PROBLEM)

    # If not any success, put NaN
    ert = ert.where(ds[SUCCESS].any(PROBLEM))

    # Divide by the dimensionality of the problem
    ert = ert / ds["dims"].mean(PROBLEM)

    return ert.mean(SEED)


def compute_ert_multi_seed(
    ds: xr.Dataset, n_seeds: int, n_picks: int
) -> xr.DataArray:
    """
    Compute ERT across multiple random seeds and return an xarray
    with dimensions (seed, taskID, ...).
    """
    n_realizations = int(ds[REALIZATION].max())
    samples = np.stack(
        [
            draw_samples(
                seed=s, n_realizations=n_realizations, n_picks=n_picks
            )
            for s in np.arange(n_seeds)
        ],
        axis=0,
    )

    samples_da = xr.DataArray(
        samples,
        dims=(SEED, PROBLEM),
    )

    ert_all = ds.groupby(TASK_ID).map(compute_ert_group, samples=samples_da)

    ert_full = ert_all.sel({TASK_ID: ds[TASK_ID]})

    return ert_full


# =============================================================================
