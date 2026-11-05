"""
Module containing functions that compute the performance of optimizers

Functions
---------

scoring_loss
    The loss series scores are computed from (noise-free when present).
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
expected_runtime
    Per-task expected runtime to each ladder position.
performance_profile
    Dolan-Moré performance profile of a per-task runtime table.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Sequence

# Third-party
import numpy as np
import xarray as xr

# Local
from l2co._src.naming import (
    BUDGET,
    COST,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    METRIC,
    OUTPUT_DIM,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    POSITION,
    PROBLEM,
    RATIO,
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
        # A loss below global_min scores as reaching it (see
        # :func:`shift_to_quality_value`).
        p = (x - g_min).clip(min=0.0)

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
    realizations: Sequence[int] | None = None,
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
    realizations : Sequence[int], optional
        Positional indices of the runs the restart bootstrap may draw
        from, by default None (every run in ``performance``). Passing two
        disjoint index sets yields two independent estimates of the same
        profile, which is how a caller measures the sampling error of the
        ERTD rather than assuming it. Restricting the pool changes the
        estimate -- fewer runs is a noisier bootstrap -- so the halves are
        comparable to each other, not to the full-pool profile.

        Treated as a set: indices are deduplicated and sorted before use,
        so the same pool always gives the same answer however it was
        built. Out-of-range indices raise.

    Returns
    -------
    xr.DataArray
        ERTD profile indexed by ``cost_per_dim``.

    Raises
    ------
    ValueError
        ``seed_batch`` is below 1, or ``realizations`` is empty.
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

    if realizations is not None:
        # A pool, not an ordering: the bootstrap indexes positionally, so
        # without this the same set of runs in a different order would
        # give a different estimate.
        picks = np.unique(np.asarray(realizations, dtype=int))
        if picks.size == 0:
            raise ValueError("realizations must select at least one run.")
        ds = ds.isel({REALIZATION: picks})

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
        Never negative: a value below ``g_min`` scores 0 (see
        :func:`shift_to_quality_value`).

    Notes
    -----
    Quality value is scaled such that the median initial quality
    equals the maximum target value from DEFAULT_TARGETS.
    """
    if np.isnan(g_min):
        g_min = x.min(skipna=True)
    qv = (x - g_min).clip(min=0.0)

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


def scoring_loss(data: xr.Dataset) -> xr.DataArray:
    """The loss series a run is scored on.

    Noisy tasks are scored on their noise-free value (ADR 0024; spec
    decision 12 of the dataset paper's protocol revision). A run on a
    noisy task carries two series: ``output_min``, the lowest *observed*
    loss per step -- what the optimizer saw -- and ``output_min_true``,
    the lowest noise-free loss of the same evaluated population. This
    returns the latter where a task has it and the former otherwise.

    The choice is per task, because one dataset can hold both kinds: a
    databank store with noisy and noise-free tasks reads
    ``output_min_true`` back as NaN on the noise-free tasks' cells, as
    do loaders concatenated from the two kinds. A task none of whose
    runs has the series is scored on ``output_min``; a task with it, on
    it. A task with the series on some runs and observed values alone
    on others raises, since its runs would be scored on different
    series. The dump refuses that within one store
    (:func:`l2co._src.databank.records._check_noise_free_series`), but
    it can arise where data from several sources is combined: an
    overlay read, ``DataLoader.join`` or ``+``. A run with neither
    series (an unwritten pair) counts as neither.

    Tasks are told apart by the ``taskID`` coordinate: a dimension (a
    databank read) or a level of ``problem`` (a ``DataLoader``). Data
    without it, or with a scalar one, is one task.

    Parameters
    ----------
    data : xr.Dataset
        Loss histories with ``output_min`` and, for noisy tasks,
        ``output_min_true`` (same dims).

    Returns
    -------
    xr.DataArray
        ``data["output_min"]`` itself when there is no
        ``output_min_true`` -- so a noise-free task's scores are
        computed exactly as before, and lazy data stays unread --
        else ``output_min_true``, with each run that has no noise-free
        series replaced by its observed one. Telling which runs have
        it reads both series once.

    Raises
    ------
    ValueError
        If a task has the noise-free series on some of its runs and
        observed values alone on others.
    """
    if OUTPUT_MIN_TRUE not in data.data_vars:
        return data[OUTPUT_MIN]
    true = data[OUTPUT_MIN_TRUE]
    observed = data[OUTPUT_MIN]
    # Small (no ``budget``), so computed once and reused below.
    runs = xr.Dataset(
        {"true": _per_run(true), "observed": _per_run(observed)}
    ).compute()
    has_series = runs["true"]
    _refuse_mixed_series(has_series, runs["observed"] & ~has_series)
    return true.where(has_series, observed)


def _per_run(loss: xr.DataArray) -> xr.DataArray:
    """Whether each run has any value of ``loss`` (reduces ``budget``)."""
    present = loss.notnull()
    return present.any(BUDGET) if BUDGET in present.dims else present


#: Tasks a :func:`scoring_loss` error names before summarising the rest.
_MAX_TASKS_SHOWN = 5


def _refuse_mixed_series(
    has_series: xr.DataArray, lacking: xr.DataArray
) -> None:
    """Raise if a task has runs in both ``has_series`` and ``lacking``.

    ``has_series`` flags the runs with the noise-free series, ``lacking``
    those with observed values but no noise-free series; both have the
    same per-run dims (no ``budget``).
    """
    task_ids = has_series.coords.get(TASK_ID)
    if task_ids is None or task_ids.ndim == 0:
        if not (bool(has_series.any()) and bool(lacking.any())):
            return
        mixed = ["(one task)" if task_ids is None else str(task_ids.values)]
    else:
        (task_dim,) = task_ids.dims

        def _tasks_with(flags: xr.DataArray) -> set:
            others = [d for d in flags.dims if d != task_dim]
            if others:
                flags = flags.any(others)
            ids = np.asarray(flags[TASK_ID].values)[flags.values]
            return {str(task_id) for task_id in ids}

        mixed = sorted(_tasks_with(has_series) & _tasks_with(lacking))
        if not mixed:
            return
    shown = mixed[:_MAX_TASKS_SHOWN]
    more = len(mixed) - len(shown)
    raise ValueError(
        f"the noise-free series '{OUTPUT_MIN_TRUE}' is on some of its runs "
        f"only, for task(s) {shown}"
        + (f" and {more} more" if more else "")
        + ": the other runs have observed values alone, so the task's "
        "runs would be scored on different series. The dump refuses this "
        "within one store; here it comes from combining sources (an "
        "overlay read, DataLoader.join or +) where some runs were made "
        "without the series (e.g. rl2co's environment, or init_run_state "
        "without record_noise_free). Re-run those through RolloutWrapper, "
        "or leave them out (l2co ADR 0024)."
    )


def _fallback_targets(n_targets: int) -> np.ndarray:
    return np.logspace(
        np.log10(HARDEST_TARGET_QV),
        np.log10(DEFAULT_EASIEST_TARGET_QV),
        n_targets,
    )


def shift_to_quality_value(data: xr.Dataset, g_min: float) -> xr.Dataset:
    """Shift the scored loss into quality-value space (``g_min == 0``).

    Subtracts the task's global minimum from the series the task is
    scored on (:func:`scoring_loss`: ``output_min``, or on a noisy task
    its noise-free ``output_min_true``) so the resulting dataset lives
    in quality-value space, where the optimum is 0 and the hardest
    target ``1e-8`` is meaningful regardless of the original scale of
    the objective. If ``g_min`` is NaN, the minimum of that series over
    the data is used as a fallback.

    The result holds the shifted series as ``output_min`` and no
    ``output_min_true``, so everything downstream of the shift
    (:func:`compute_task_targets` and its random-search anchor,
    :func:`calculate_ert`) reads one variable whatever the task.

    A loss below ``g_min`` scores 0, as if it had reached ``g_min``.
    That happens when ``g_min`` was set empirically and an optimizer
    finds a lower value; the quality value is never negative, so the
    log scales and ratios downstream stay finite. The dump records each
    such case (:func:`global_min_breaches`) so ``g_min`` can be re-tuned.

    Parameters
    ----------
    data : xr.Dataset
        Per-task optimization dataset.
    g_min : float
        Task's global minimum (NaN allowed; falls back to data min).

    Returns
    -------
    xr.Dataset
        Copy of ``data`` with ``output_min`` replaced by the scored
        series shifted by ``-g_min`` and floored at 0, and
        ``output_min_true`` dropped.
    """
    loss = scoring_loss(data)
    if np.isnan(g_min):
        g_min = float(loss.min(skipna=True))
    if OUTPUT_MIN_TRUE in data.data_vars:
        data = data.drop_vars(OUTPUT_MIN_TRUE)
    return data.assign({OUTPUT_MIN: (loss - g_min).clip(min=0.0)})


#: A loss counts as a ``global_min`` breach below ``global_min - rtol *
#: max(1, |global_min|)``. The same rule and default as the CUTEst floor
#: test in l2co_experiments; the margin absorbs float noise at an exact
#: optimum.
GLOBAL_MIN_BREACH_RTOL = 1e-6


def global_min_breaches(
    data: xr.Dataset,
    g_min: float,
    rtol: float = GLOBAL_MIN_BREACH_RTOL,
) -> dict[str, dict[str, float | int]]:
    """The schedules whose lowest loss on a task is below ``g_min``.

    Parameters
    ----------
    data : xr.Dataset
        One task's raw ``output_min`` over ``schedule`` (and any of
        ``realization`` / ``budget`` / ``output_dim``), before the shift
        into quality-value space. On a noisy task the noise-free
        ``output_min_true`` is judged instead (:func:`scoring_loss`): an
        observed value below ``g_min`` is noise, not a ``g_min`` to
        re-tune.
    g_min : float
        The task's ``global_min``. NaN means there is none to breach.
    rtol : float, optional
        Breach margin, relative to ``max(1, |g_min|)``, by default
        :data:`GLOBAL_MIN_BREACH_RTOL`.

    Returns
    -------
    dict[str, dict[str, float | int]]
        Per breaching schedule: ``best``, its lowest loss; ``margin``,
        how far that is below ``g_min``; and ``realization``, the run
        that reached it (when the data has that axis). Empty when no
        schedule breaches.
    """
    if not np.isfinite(g_min):
        return {}
    floor = g_min - rtol * max(1.0, abs(g_min))
    loss = scoring_loss(data)
    breaches: dict[str, dict[str, float | int]] = {}
    for schedule in loss[SCHEDULE].values:
        values = loss.sel({SCHEDULE: schedule})
        flat = np.asarray(values.values, dtype=np.float64).reshape(-1)
        if np.isnan(flat).all():
            continue
        i = int(np.nanargmin(flat))
        best = float(flat[i])
        if not best < floor:
            continue
        record: dict[str, float | int] = {
            "best": best,
            "margin": g_min - best,
        }
        if REALIZATION in values.dims:
            index = np.unravel_index(i, values.shape)[
                values.dims.index(REALIZATION)
            ]
            record["realization"] = int(values[REALIZATION].values[index])
        breaches[str(schedule)] = record
    return breaches


def compute_task_targets(
    data_qv: xr.Dataset,
    n_targets: int = N_TARGETS,
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
) -> np.ndarray:
    """Build per-task log-spaced ERT targets in quality-value space.

    Expects ``data_qv`` already shifted so that ``global_min == 0``
    (see :func:`shift_to_quality_value`), which also puts a noisy
    task's noise-free series in ``output_min``: the anchor is then
    random search's noise-free losses. The hardest target is
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

    Raises
    ------
    ValueError
        If ``easiest_quantile`` is outside ``(0, 1)``, or ``data_qv``
        still holds ``output_min_true`` -- it was not shifted, and the
        anchor would be read off the observed series.
    """
    if OUTPUT_MIN_TRUE in data_qv.data_vars:
        raise ValueError(
            "data_qv still holds 'output_min_true'; pass it through "
            "shift_to_quality_value first, which puts the noise-free "
            "series the task is scored on in 'output_min'."
        )
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

    Shifts the scored loss -- ``output_min``, or on a noisy task its
    noise-free ``output_min_true`` (:func:`scoring_loss`) -- into
    quality-value space (subtracting ``global_min``, falling back to
    the data minimum when it is NaN) so the optimum is 0, then counts
    the first budget step that reaches each supplied target. The
    supplied target quality values are echoed as the ``target_value``
    data variable so they survive concatenation along ``TASK_ID``; the
    ``target`` dim itself is an integer index ``0..len(targets)-1``.

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


#                                       Performance profiles (docs/adr/0021)
# =============================================================================


def _ladder_index(position: float, n_targets: int) -> int:
    """Stored target index of a ladder position.

    Targets are stored hardest first, so position 1 is index 0 and
    position 0 the last index. A position between two stored targets
    takes the nearer one; exactly halfway, the harder one, so a profile
    never asks for less than it was told to.
    """
    return int(np.ceil((n_targets - 1) * (1.0 - position) - 0.5 - 1e-9))


def expected_runtime(
    performance: xr.Dataset,
    positions: Sequence[float],
    cost_metric: str = FEVALS,
) -> xr.DataArray:
    """Per-task expected runtime to each ladder position.

    For one optimizer on one task: the cost spent over all of its runs,
    divided by the number of runs that reached the target. A failed run
    adds its stored cost to the total but nothing to the divisor. This is
    the closed form of the quantity the ERTD's restart bootstrap
    estimates (:func:`ertd`), so no resampling is needed.

    Parameters
    ----------
    performance : xr.Dataset
        Dataset with ``cost`` (over ``metric``) and ``success``, both
        over ``target`` (stored hardest first) and a ``problem`` axis
        stacking ``realization`` and ``taskID``. Exactly one
        ``output_dim``.
    positions : Sequence[float]
        Ladder positions, 0 (easiest stored target) to 1 (hardest).
    cost_metric : str, optional
        ``"fevals"`` or ``"iterations"``, by default ``"fevals"``.

    Returns
    -------
    xr.DataArray
        Expected runtime over ``taskID`` x ``schedule`` x ``position``,
        with the stored target index each position used as a ``target``
        coordinate along ``position``. NaN where the schedule has no data
        on the task, infinite where no run reached the target.
    """
    n_targets = performance.sizes[TARGET]
    indices = xr.DataArray(
        [_ladder_index(p, n_targets) for p in positions],
        dims=POSITION,
        coords={POSITION: list(positions)},
    )
    ds = (
        performance[[COST, SUCCESS]]
        .sel({METRIC: cost_metric})
        .squeeze(OUTPUT_DIM, drop=True)
        .isel({TARGET: indices})
        .unstack(PROBLEM)
    )
    # With no data both sums are 0 (NaN-skipping), and 0 / 0 is NaN; with
    # data but no success, total / 0 is infinite. xarray keeps both quiet.
    total = ds[COST].sum(REALIZATION)
    solved = ds[SUCCESS].sum(REALIZATION)
    return (total / solved).transpose(TASK_ID, SCHEDULE, POSITION)


#: Right end of a performance profile's ratio axis (docs/adr/0021).
DEFAULT_MAX_RATIO = 1e4

#: Points on the log10 ratio axis: 0.02 decades apart up to 1e4.
DEFAULT_RATIO_POINTS = 201


def performance_profile(
    ert: xr.DataArray,
    max_ratio: float = DEFAULT_MAX_RATIO,
    n_points: int = DEFAULT_RATIO_POINTS,
) -> xr.Dataset:
    """Dolan-Moré performance profile of a per-task runtime table.

    For each task, every schedule's runtime is divided by the cheapest
    schedule's; the profile is the fraction of tasks on which that ratio
    is at most each value on a log10 axis from 1 to ``max_ratio`` (see
    ``docs/adr/0021``). Pure, and reduces only over ``taskID``, so a
    report can read the table once and profile any subset of its tasks.

    Parameters
    ----------
    ert : xr.DataArray
        Expected runtime over ``taskID`` x ``schedule`` (any further
        axes, e.g. ``position``, are kept), as
        :meth:`l2co.DataLoader.expected_runtime` returns.
    max_ratio : float, optional
        Right end of the ratio axis, by default :data:`DEFAULT_MAX_RATIO`.
        A larger ratio still counts as solved, in ``solve_rate``.
    n_points : int, optional
        Points on the ratio axis, by default
        :data:`DEFAULT_RATIO_POINTS`.

    Returns
    -------
    xr.Dataset
        Per schedule:

        - ``rho``: the profile curve over ``ratio``;
        - ``win_rate``: the fraction of tasks on which it is the
          cheapest, an exact tie counting for every tied schedule;
        - ``solve_rate``: the fraction of tasks it solves at all, the
          curve's limit past any cap.

        Over the task set: ``n_tasks``, and ``n_unsolved``, the tasks no
        schedule solves. Those stay in every fraction's denominator as a
        failure for all, so ``solve_rate`` is an absolute solve rate.

        The profile covers only tasks every schedule has data for (NaN
        in ``ert`` is no data). ``n_dropped`` counts the tasks left out,
        and ``n_missing`` per schedule the tasks it has no data on --
        who caused the drops.
    """
    complete = ert.notnull().all(SCHEDULE)
    common = ert.where(complete)
    best = common.min(SCHEDULE)
    # A task nobody solves has no cheapest schedule: inf / inf is NaN,
    # which counts as neither a win nor a solve. A dropped task is NaN
    # throughout and counts in no denominator.
    ratio = common / best
    n_tasks = complete.sum(TASK_ID)
    grid = np.logspace(0.0, np.log10(max_ratio), n_points)
    grid[-1] = max_ratio  # logspace's own end point is rounded
    ratios = xr.DataArray(grid, dims=RATIO, coords={RATIO: grid})
    return xr.Dataset(
        {
            "rho": (ratio <= ratios).sum(TASK_ID) / n_tasks,
            "win_rate": (ratio == 1).sum(TASK_ID) / n_tasks,
            "solve_rate": np.isfinite(ratio).sum(TASK_ID) / n_tasks,
            "n_tasks": n_tasks,
            "n_unsolved": (complete & np.isinf(best)).sum(TASK_ID),
            "n_dropped": (~complete).sum(TASK_ID),
            "n_missing": ert.isnull().sum(TASK_ID),
        }
    )
