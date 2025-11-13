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

from functools import partial

# Third-party
import numpy as np
import xarray as xr

from .naming import (
    BUDGET,
    COST,
    OUTPUT_DIM,
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
    performance: xr.DataArray,
    cost_measure: xr.DataArray,
    dims: xr.DataArray,
    cost_values: np.ndarray = DEFAULT_COST_VALUES,
) -> xr.DataArray:
    targets = xr.DataArray(DEFAULT_TARGETS, dims=TARGET)
    solved = performance <= targets
    success = solved.any(BUDGET)
    first_solved = solved.argmax(BUDGET).where(
        success, other=solved[BUDGET].argmax()
    )

    ds = xr.Dataset(
        {
            COST: cost_measure.isel({BUDGET: first_solved}),
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
    ertd = (ert_multi <= budgets).mean([SEED, OUTPUT_DIM, TARGET])

    return ertd


def compute_ert_group(ds: xr.Dataset, seed: int, n_picks: int) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    n_realizations = int(ds[REALIZATION].max())
    samples = np.concat(
        [
            rng.permutation(n_realizations + 1),
            rng.integers(
                0, n_realizations + 1, size=max(n_picks - n_realizations, 0)
            ),
        ]
    )

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
    return ert


def compute_ert_multi_seed(
    ds: xr.Dataset, n_seeds: int, n_picks: int
) -> xr.DataArray:
    """
    Compute ERT across multiple random seeds and return an xarray
    with dimensions (seed, taskID, ...).
    """
    # Create a version of compute_ert_group with n_picks fixed
    compute_partial = partial(compute_ert_group, n_picks=n_picks)

    # Map over seeds in a vectorized fashion
    ert_list = [
        ds.groupby(TASK_ID)
        .map(compute_partial, seed=seed)
        .expand_dims({SEED: [seed]})
        for seed in np.arange(n_seeds)
    ]

    # Concatenate along the seed dimension
    ert_all = xr.concat(ert_list, dim=SEED)

    return ert_all


# =============================================================================
