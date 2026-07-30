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

from .naming import (
    BUDGET,
    COST,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
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
# DEFAULT_COST_VALUES = np.logspace(-1, 6, 100) ** 0.8
DEFAULT_COST_VALUES = np.append(np.logspace(-1, 6, 1000) ** 0.8, 10**5)
DEFAULT_TARGETS = np.linspace(0.0, 1.0, 51)

#: Bootstrap replicates the ERTD averages over (the ``seed`` axis).
ERTD_N_SEEDS = 50

#: Realizations drawn (with replacement) per replicate.
ERTD_N_PICKS = 50

#: Replicates evaluated at once inside :func:`ertd`. The resampled
#: ``cost`` / ``success`` arrays carry a ``seed x picks`` axis in place of
#: ``realization``, so the peak footprint scales with this batch, not with
#: :data:`ERTD_N_SEEDS`: on a full databank selection (~150 tasks x ~60
#: schedules x 51 targets) the unbatched form allocates tens of GB to
#: produce a sub-GB profile. The batching is exact -- see :func:`ertd`.
ERTD_SEED_BATCH = 5


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
    """
    Compute adaptive worst quantile along a 1D array.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (T,).
    q : float, optional
        Quantile value between 0 and 1, by default 0.99.

    Returns
    -------
    np.ndarray
        Array of worst quantile values computed with adaptive
        window size, shape (T,).

    Notes
    -----
    The window size k grows logarithmically with the index:
    $k = \\lceil \\log(t + 3)^2 / 2 \\rceil$.
    """
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


def min_custom(ds: xr.Dataset) -> xr.DataArray:
    """
    Apply custom log scaling to output_min based on global_min.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'output_min' and 'global_min' variables.

    Returns
    -------
    xr.DataArray
        Min-max scaled log-transformed output values in range [0, 1].

    Notes
    -----
    Applies log10 transformation after shifting by global minimum,
    adding small epsilon to avoid log(0), then min-max scales the
    result.
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


def _ert_for_seeds(ds: xr.Dataset, samples: xr.DataArray) -> xr.DataArray:
    """Expected runtime for one batch of bootstrap seed replicates.

    Resamples the ``realization`` axis with the given draws, sums the cost
    up to (and including) each problem's first success, and normalizes by
    the problem's dimensionality. Replicates without a single success
    become NaN, so the caller's ``mean(SEED)`` skips them.

    This is the memory-heavy inner step: the ``isel`` below replaces the
    store's ``realization`` axis with ``seed x picks``, so the resampled
    ``cost`` / ``success`` arrays are ``n_picks * n_seeds_in_batch /
    n_realizations`` times the footprint of the input. Keeping the batch
    small is what bounds :func:`ertd`'s peak.

    Parameters
    ----------
    ds : xr.Dataset
        Unstacked dataset with ``cost``, ``success`` and ``dims``.
    samples : xr.DataArray
        Bootstrap indices for this batch, dims ``(seed, realization)``.

    Returns
    -------
    xr.DataArray
        ERT for this batch's seeds, with the ``realization`` axis reduced
        away. Materialized (not lazy) so the resampled arrays are released
        before the next batch allocates its own.
    """
    ds_selected = ds.isel({REALIZATION: samples})
    success = ds_selected[SUCCESS]
    any_success = success.any(REALIZATION)

    # Index of the first success, -1 where there is none at all.
    first_true = success.argmax(REALIZATION).where(any_success, other=-1)

    idx = xr.DataArray(
        np.arange(ds_selected.sizes[REALIZATION]), dims=REALIZATION
    )
    mask = (idx <= first_true).fillna(False)

    # Sum cost up to the first success; NaN when nothing was solved.
    ert = ds_selected[COST].where(mask, 0).sum(REALIZATION)
    ert = ert.where(any_success)

    return (ert / ds_selected["dims"].mean(REALIZATION)).compute()


def ertd(
    performance: xr.Dataset,
    dims: xr.DataArray,
    cost_values: np.ndarray = DEFAULT_COST_VALUES,
    seed_batch: int = ERTD_SEED_BATCH,
) -> xr.DataArray:
    """Compute the Expected Runtime Distribution (ERTD).

    Estimates the fraction of problems solved within
    each cost budget, normalized by problem dimensionality,
    using bootstrap sampling over realizations.

    The bootstrap is evaluated in batches of ``seed_batch`` replicates
    (see :func:`_ert_for_seeds`) and the per-batch ERTs are concatenated
    back into one ``seed`` axis before the mean. Batching only changes
    *when* the resampled arrays are allocated, never the arithmetic: each
    ERT element reduces over its own replicate's ``realization`` axis
    alone, and the concatenated axis preserves seed order, so the result
    is independent of ``seed_batch`` (asserted in
    ``tests/unit/test_ertd_separability.py``).

    Parameters
    ----------
    performance : xr.Dataset
        Dataset with ``cost`` and ``success`` variables.
    dims : xr.DataArray
        Problem dimensionalities for normalization.
    cost_values : np.ndarray, optional
        Cost budget thresholds, by default
        ``DEFAULT_COST_VALUES``.
    seed_batch : int, optional
        Bootstrap replicates held in memory at once, by default
        :data:`ERTD_SEED_BATCH`. Trades peak memory against Python-level
        loop overhead; pass :data:`ERTD_N_SEEDS` for the unbatched
        computation.

    Returns
    -------
    xr.DataArray
        ERTD profile indexed by ``cost_per_dim``.
    """
    if seed_batch < 1:
        raise ValueError(f"seed_batch must be >= 1, got {seed_batch}.")

    ds = xr.Dataset(
        {
            COST: performance[COST],
            SUCCESS: performance[SUCCESS].fillna(False),
            "dims": dims,
        }
    )

    ds = ds.unstack(PROBLEM)

    samples = np.stack(
        [
            draw_samples(
                seed=s,
                n_realizations=ds[REALIZATION].size,
                n_picks=ERTD_N_PICKS,
            )
            for s in np.arange(ERTD_N_SEEDS)
        ],
        axis=0,
    )

    samples_da = xr.DataArray(
        samples,
        dims=(SEED, REALIZATION),
    )

    ert = xr.concat(
        [
            _ert_for_seeds(
                ds, samples_da.isel({SEED: slice(lo, lo + seed_batch)})
            )
            for lo in range(0, ERTD_N_SEEDS, seed_batch)
        ],
        dim=SEED,
    )

    # NaN-skipping mean: replicates that solved nothing do not count.
    ert_multi = ert.mean(SEED)

    values = np.unique(cost_values)
    budgets = xr.DataArray(
        values,
        dims="cost_per_dim",
        coords={"cost_per_dim": values},
    )

    return (ert_multi <= budgets).mean([OUTPUT_DIM, TARGET])


def draw_samples(seed: int, n_realizations: int, n_picks: int):
    """
    Draw random samples with replacement for bootstrap sampling.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_realizations : int
        Total number of realizations available.
    n_picks : int
        Number of samples to draw.

    Returns
    -------
    np.ndarray
        Array of sampled indices with shape (n_picks,).

    Notes
    -----
    First draws a random permutation of all realizations, then
    draws additional random samples with replacement if
    n_picks > n_realizations.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.permutation(n_realizations),
            rng.integers(
                0,
                n_realizations,
                size=max(n_picks - n_realizations, 0),
            ),
        ]
    )


def compute_ert_group(
    ds: xr.Dataset,
    samples: xr.DataArray,
) -> xr.DataArray:
    """
    Compute Expected Runtime (ERT) for a group of problems.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'cost', 'success', and 'dims' variables.
    samples : xr.DataArray
        Sample indices for problem selection.

    Returns
    -------
    xr.DataArray
        Mean ERT across seeds, normalized by problem dimensionality.

    Notes
    -----
    Computes ERT as the sum of costs until first success, divided
    by problem dimensionality. Returns NaN if no success achieved.
    """
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
    Compute ERT across multiple random seeds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing cost, success, and dims variables.
    n_seeds : int
        Number of random seeds to use for bootstrap sampling.
    n_picks : int
        Number of problem samples per seed.

    Returns
    -------
    xr.DataArray
        ERT values with dimensions (seed, taskID, ...).

    Notes
    -----
    Performs bootstrap sampling of problems for each seed to
    estimate ERT distribution.
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

    # TODO: is this necessary?
    ert_all = ds.groupby(TASK_ID).map(compute_ert_group, samples=samples_da)

    ert_full = ert_all.sel({TASK_ID: ds[TASK_ID]})

    return ert_full


# =============================================================================


def quality_value(x: xr.DataArray, g_min: float) -> xr.DataArray:
    """
    Compute scaled quality value relative to global minimum.

    Parameters
    ----------
    x : xr.DataArray
        Input data array with objective values.
    g_min : float
        Global minimum value. If NaN, uses minimum from x.

    Returns
    -------
    xr.DataArray
        Scaled quality values where 0 indicates optimal solution.

    Notes
    -----
    Quality value is scaled such that the median initial quality
    equals the maximum target value from DEFAULT_TARGETS.
    """
    if np.isnan(g_min):
        g_min = x.min(skipna=True)
    qv = x - g_min

    # # scaling
    # sf = (
    #     1
    #     / qv.isel({BUDGET: 0})
    #     .quantile(q=0.5, skipna=True)
    #     .drop_vars("quantile")
    # ) * DEFAULT_TARGETS[-1]
    return qv  # * sf


# =============================================================================

HARDEST_TARGET_QV = 1e-8
DEFAULT_EASIEST_TARGET_QV = 1e2
N_TARGETS = 51
DEFAULT_EASIEST_QUANTILE = 0.75


def _fallback_targets(n_targets: int) -> np.ndarray:
    return np.logspace(
        np.log10(HARDEST_TARGET_QV),
        np.log10(DEFAULT_EASIEST_TARGET_QV),
        n_targets,
    )


def shift_to_quality_value(data: xr.Dataset, g_min: float) -> xr.Dataset:
    """Shift ``output_min`` into quality-value space (``g_min == 0``).

    Subtracts the task's global minimum from ``output_min`` so the
    resulting dataset lives in quality-value space, where the optimum
    is 0 and the hardest target ``1e-8`` is meaningful regardless of
    the original scale of the objective. If ``g_min`` is NaN, the
    minimum of ``output_min`` over the data is used as a fallback.

    Parameters
    ----------
    data : xr.Dataset
        Per-task optimization dataset.
    g_min : float
        Task's global minimum (NaN allowed; falls back to data min).

    Returns
    -------
    xr.Dataset
        Copy of ``data`` with ``output_min`` shifted by ``-g_min``.
    """
    if np.isnan(g_min):
        g_min = float(data[OUTPUT_MIN].min(skipna=True))
    return data.assign({OUTPUT_MIN: data[OUTPUT_MIN] - g_min})


def compute_task_targets(
    data_qv: xr.Dataset,
    n_targets: int = N_TARGETS,
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
) -> np.ndarray:
    """Build per-task log-spaced ERT targets in quality-value space.

    Expects ``data_qv`` already shifted so that ``global_min == 0``
    (see :func:`shift_to_quality_value`). The hardest target is
    ``1e-8`` and the easiest is set so that ``easiest_quantile`` of
    the ``randomsearch`` baseline's ``output_min`` values lie above
    (i.e. are worse than) it — equivalently, the
    ``(1 - easiest_quantile)``-quantile of those values. Randomsearch
    values are flattened across all remaining dims (typically
    ``REALIZATION`` x ``BUDGET`` x any ``randomsearch`` schedules).
    Intermediate targets are log-spaced between these endpoints.

    Falls back to ``np.logspace(-8, 2, n_targets)`` if any of the
    following holds:

    * no ``randomsearch`` schedule is present for this task,
    * the easiest quality value is non-finite or not strictly above
      the hardest target.

    Parameters
    ----------
    data_qv : xr.Dataset
        Per-task dataset with ``output_min`` already in quality-value
        space and a ``schedule`` coordinate whose values are
        ``OptimizationStep.name`` strings.
    n_targets : int, optional
        Number of target levels to produce, by default ``51``.
    easiest_quantile : float, optional
        Fraction of ``randomsearch`` ``output_min`` values that
        should lie above (be worse than) the easiest target.
        Must be in ``(0, 1)``. Default ``0.75``.

    Returns
    -------
    np.ndarray
        1-D array of ``n_targets`` log-spaced target quality values,
        sorted from hardest (``1e-8``) to easiest.
    """
    if not 0.0 < easiest_quantile < 1.0:
        raise ValueError(
            f"easiest_quantile must be in (0, 1), got {easiest_quantile}"
        )

    if SCHEDULE not in data_qv.coords or OUTPUT_MIN not in data_qv.data_vars:
        return _fallback_targets(n_targets)

    schedule_names = np.asarray(data_qv[SCHEDULE].values, dtype=str)
    mask = np.array(
        [name.startswith("randomsearch") for name in schedule_names]
    )
    if not mask.any():
        return _fallback_targets(n_targets)

    rs = data_qv[OUTPUT_MIN].isel({SCHEDULE: np.where(mask)[0]})
    easiest_qv = float(rs.quantile(1.0 - easiest_quantile, skipna=True))

    if not np.isfinite(easiest_qv) or easiest_qv <= HARDEST_TARGET_QV:
        return _fallback_targets(n_targets)

    return np.logspace(
        np.log10(HARDEST_TARGET_QV), np.log10(easiest_qv), n_targets
    )


def calculate_ert(
    data: xr.Dataset,
    targets: np.ndarray | None = None,
    global_min: np.ndarray | None = None,
) -> xr.Dataset:
    """Calculate Expected Running Time (ERT) metrics.

    Shifts ``output_min`` into quality-value space (subtracting
    ``global_min``, falling back to the data minimum when it is NaN)
    so the optimum is 0, then counts the first budget step that
    reaches each supplied target. The supplied target quality values
    are echoed as the ``target_value`` data variable so they survive
    concatenation along ``TASK_ID``; the ``target`` dim itself is an
    integer index ``0..len(targets)-1``.

    Parameters
    ----------
    data : xr.Dataset
        Optimization dataset with ``output_min``, ``fevals``,
        and ``iterations`` variables.
    targets : np.ndarray or None, optional
        1-D array of target quality values (in quality-value space,
        i.e. relative to ``global_min``). When ``None`` (default),
        ``_fallback_targets(N_TARGETS) == np.logspace(-8, 2, 51)`` is
        used. Callers that want randomsearch-derived targets should
        compute them via :func:`compute_task_targets` and pass them
        in explicitly.
    global_min : np.ndarray or None, optional
        Task global minimum used to shift ``output_min`` into
        quality-value space. When ``None`` (default), it is read
        from ``data.coords[GLOBAL_MIN]`` and falls back to
        ``np.nan`` if that coordinate is missing; a NaN value in
        turn triggers the per-task data-minimum fallback inside
        :func:`shift_to_quality_value`. When provided, the supplied
        value is used directly.

    Returns
    -------
    xr.Dataset
        Dataset with ``cost``, ``success``, and ``target_value``
        variables across the supplied target quality levels.
    """
    if global_min is None:
        global_min = data.coords.get(GLOBAL_MIN, np.nan)
        if global_min is None:
            global_min = np.nan

    targets_np = (
        targets if targets is not None else _fallback_targets(N_TARGETS)
    )

    data_qv = shift_to_quality_value(data, global_min)

    target_idx = np.arange(targets_np.size)

    qv = output_cummin(data_qv[OUTPUT_MIN]).ffill(BUDGET).fillna(np.inf)

    targets_da = xr.DataArray(
        targets_np, dims=TARGET, coords={TARGET: target_idx}
    )
    solved = qv <= targets_da
    success = solved.any(BUDGET)

    stopping_budget = (qv.argmin(BUDGET) + 1).clip(max=data.sizes[BUDGET] - 1)

    first_solved = (
        solved.argmax(BUDGET).where(success, other=stopping_budget).compute()
    )

    # Extract the scalar values at the 'first_solved' budget step
    fevals = data[FEVALS].isel({BUDGET: first_solved})
    iterations = data[ITERATIONS].isel({BUDGET: first_solved})

    # Combine metrics into a single DataArray named 'COST'
    # with a new dimension 'metric'
    cost_da = xr.concat(
        [fevals, iterations],
        dim=xr.DataArray([FEVALS, ITERATIONS], dims="metric", name="metric"),
    )

    result = xr.Dataset(
        {
            COST: cost_da,
            SUCCESS: success,
            "target_value": targets_da,
        }
    )
    if GLOBAL_MIN in result.coords:
        result = result.drop_vars(GLOBAL_MIN)
    return result
