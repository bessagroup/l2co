"""Pure helpers for :meth:`DataLoader.report`.

Every function here is a function of NumPy or xarray inputs only — no
:class:`DataLoader` dependency — so each is independently testable.

The log/ratio reductions (``descriptive_stats_log10``, ``pp_auc``,
mean-log-regret) all consume :func:`shifted_final_min`, whose values are
strictly positive by construction. The shift is the per-``taskID``
``global_min`` if known and finite, else the empirical minimum of
``output_min`` across ``(schedule, output_dim, budget, realization)`` for
that task — same convention as
:meth:`DataLoader.to_traj_and_stair`.
"""

from __future__ import annotations

#                                                                       Modules
# =============================================================================
# Standard
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party
import numpy as np
import xarray as xr

# Local
from l2co._src.naming import (
    BUDGET,
    COST,
    FEATURE,
    GLOBAL_MIN,
    OUTPUT_DIM,
    OUTPUT_MIN,
    PROBLEM,
    SCHEDULE,
    SUCCESS,
    TARGET,
    TASK_ID,
)

if TYPE_CHECKING:
    # Only used as a type hint on ``build_report``. DataLoader imports
    # from this module at runtime, so taking the runtime import here would
    # be circular.
    from l2co._src.data.loader import DataLoader

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

EPSILON: float = 1e-30
"""Floor added after subtracting the per-task minimum so that log/ratio
reductions stay strictly positive. Matches the convention used elsewhere
in the L2CO experiment stack."""

_PP_TAU_MAX: float = 100.0
"""Upper bound of the Dolan-Moré performance-profile ratio axis."""

_ERTD_FEVALS_PER_DIM: tuple[int, ...] = (1, 10, 100, 1_000, 10_000, 100_000)
"""Cost-per-dim points sampled in :func:`build_report` for the
``fevals`` cost metric."""

_BASELINE_SCHEDULES: tuple[str, ...] = ("randomsearch",)
"""Schedule names treated as random-search baselines for the
``schedule_worse_than_baseline`` anomaly rule. Only schedules also
present in ``self.data[SCHEDULE]`` trigger the rule."""

_CARDINALITY_CAP: int = 64
"""Task-feature cardinality threshold. Features with cardinality > cap
emit only ``{cardinality: N}`` in ``task_groups`` and are skipped from
``grouped``."""

_TOP_K_TASKS: int = 5
"""Number of entries in ``top_k_hardest_tasks``, ``top_k_easiest_tasks``,
``top_k_tasks_by_variance``."""

_EXCLUDED_TASK_FEATURES: frozenset[str] = frozenset({GLOBAL_MIN})
"""Task-label features that should not be surfaced as categorical
features in the report. ``global_min`` is a per-task numerical floor
used internally by :func:`per_task_shift` to make losses safe-for-log;
exposing it as a ``task_group``/``grouped``/``regime_change`` axis would
emit noise (``{cardinality: N}`` stubs or one-task-per-value tables)
rather than diagnostic signal."""

__all__ = [
    "EPSILON",
    "build_report",
    "descriptive_stats",
    "descriptive_stats_log10",
    "pairwise_win_rates",
    "per_task_shift",
    "pp_auc",
    "sanitize_for_json",
    "shifted_final_min",
    "top_k_by",
]


#                                                          JSON sanitisation
# =============================================================================


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-clean structure.

    NumPy scalars become native Python scalars; NaN and Inf become
    ``None``; :class:`pathlib.Path` becomes ``str``; tuples become
    lists; dict keys are coerced to ``str``.

    Parameters
    ----------
    obj : Any
        Object to sanitize.

    Returns
    -------
    Any
        JSON-serialisable structure.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer, int, np.bool_, bool)):
        # bool must come before int handling in JSON; isinstance covers both.
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


#                                                          Descriptive stats
# =============================================================================


def descriptive_stats(arr: np.ndarray) -> dict[str, float]:
    """NaN-safe ``{mean, median, std, min, max, p10, p90}`` for ``arr``.

    Returns NaN for every field when ``arr`` is empty or all-NaN.

    Parameters
    ----------
    arr : numpy.ndarray
        1-D array of float values.

    Returns
    -------
    dict[str, float]
        Summary statistics.
    """
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0 or np.all(np.isnan(a)):
        return {
            k: float("nan")
            for k in ("mean", "median", "std", "min", "max", "p10", "p90")
        }
    return {
        "mean": float(np.nanmean(a)),
        "median": float(np.nanmedian(a)),
        "std": float(np.nanstd(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "p10": float(np.nanpercentile(a, 10)),
        "p90": float(np.nanpercentile(a, 90)),
    }


def descriptive_stats_log10(shifted: xr.DataArray) -> dict[str, float]:
    """Descriptive stats of ``log10(shifted)``.

    ``shifted`` is expected to be strictly positive (the output of
    :func:`shifted_final_min`); the ``log10`` is therefore always
    well-defined.

    Parameters
    ----------
    shifted : xarray.DataArray
        Strictly-positive values (any shape).

    Returns
    -------
    dict[str, float]
        Summary statistics of the elementwise log10.
    """
    a = np.asarray(shifted.values, dtype=float).ravel()
    return descriptive_stats(np.log10(a))


#                                                Per-task shift / shifted final
# =============================================================================


def per_task_shift(
    data: xr.Dataset, task_labels: xr.DataArray
) -> xr.DataArray:
    """Per-``problem`` floor used to make losses safe-for-log.

    For each ``taskID`` the floor is ``global_min`` (from
    ``task_labels``) when that feature is present *and* finite; otherwise
    it is the empirical minimum of ``output_min`` across
    ``(schedule, output_dim, budget, realization)`` for that ``taskID``.
    The same value is broadcast to every problem sharing that ``taskID``.

    Computed via numpy so the returned DataArray re-uses the original
    ``data[PROBLEM]`` coordinate verbatim, avoiding the MultiIndex /
    positional-index alignment errors that arise from
    ``.unstack(PROBLEM).sel({TASK_ID: data[TASK_ID]})`` round-trips.

    Parameters
    ----------
    data : xarray.Dataset
        DataLoader ``data`` with ``output_min`` and a stacked ``problem``
        coordinate carrying a ``taskID`` level.
    task_labels : xarray.DataArray
        DataLoader ``task_labels`` with dims ``(problem, feature)``.

    Returns
    -------
    xarray.DataArray
        1-D array indexed by ``problem``.
    """
    om = data[OUTPUT_MIN]
    om_vals = np.asarray(om.values, dtype=float)
    problem_axis = om.dims.index(PROBLEM)
    other_axes = tuple(i for i in range(om_vals.ndim) if i != problem_axis)
    if other_axes:
        # All-NaN reductions emit a RuntimeWarning; suppress because
        # we explicitly check the finite mask below.
        with np.errstate(invalid="ignore"):
            om_per_problem = np.nanmin(om_vals, axis=other_axes)
    else:
        om_per_problem = om_vals

    task_ids_per_problem = np.asarray(data[TASK_ID].values)
    unique_tids, inverse = np.unique(task_ids_per_problem, return_inverse=True)
    n_tasks = int(unique_tids.size)

    emp_min_per_task = np.full(n_tasks, np.nan, dtype=float)
    for k in range(n_tasks):
        vals = om_per_problem[inverse == k]
        finite = vals[np.isfinite(vals)]
        if finite.size:
            emp_min_per_task[k] = float(np.min(finite))

    emp_per_problem = emp_min_per_task[inverse]

    if GLOBAL_MIN in task_labels[FEATURE].values.tolist():
        gm_arr = np.asarray(
            task_labels.sel({FEATURE: GLOBAL_MIN}).values, dtype=float
        )
        shift_arr = np.where(np.isfinite(gm_arr), gm_arr, emp_per_problem)
    else:
        shift_arr = emp_per_problem

    return xr.DataArray(
        shift_arr,
        coords={PROBLEM: data[PROBLEM]},
        dims=[PROBLEM],
    )


def shifted_final_min(
    data: xr.Dataset, task_labels: xr.DataArray
) -> xr.DataArray:
    """``(schedule, problem)`` array of safe-for-log final losses.

    Computes ``output_min.min(BUDGET) - per_task_shift + EPSILON`` (with
    ``output_dim`` squeezed away). The ``.min(BUDGET)`` reduction is used
    instead of ``isel(BUDGET=-1)`` because the raw ``output_min`` is the
    *per-iteration* population minimum, which can be NaN at the
    interpolated trajectory boundary even when the optimizer converged
    earlier. Taking the minimum across the budget axis gives the best
    value the optimizer actually achieved, regardless of where in the
    trajectory it occurred. The result is strictly positive by
    construction: a final loss below ``global_min`` (an empirical
    ``global_min`` an optimizer beat) is floored at the shift, so it
    scores ``EPSILON``, as if it had reached ``global_min``.

    Parameters
    ----------
    data : xarray.Dataset
        DataLoader ``data``.
    task_labels : xarray.DataArray
        DataLoader ``task_labels``.

    Returns
    -------
    xarray.DataArray
        Strictly positive values with dims ``(schedule, problem)``.
    """
    final = data[OUTPUT_MIN].min(dim=BUDGET, skipna=True)
    if OUTPUT_DIM in final.dims:
        final = final.isel({OUTPUT_DIM: 0}, drop=True)
    shift = per_task_shift(data=data, task_labels=task_labels)
    return (final - shift).clip(min=0.0) + EPSILON


#                                          Performance-profile (Dolan-Moré) AUC
# =============================================================================


def pp_auc(shifted: xr.DataArray, schedule: str) -> float:
    """Dolan-Moré performance-profile AUC in ``[0, 1]`` for one schedule.

    Builds the ratio ``r_p = shifted[schedule, p] / min_s shifted[s, p]``
    over ``problem``, integrates the empirical CDF ``ρ(τ) = P(r ≤ τ)`` by
    trapezoid in ``log10(τ)`` on ``[1, τ_max]`` (``τ_max = 100``), then
    divides by ``log10(τ_max)`` so the result is bounded in ``[0, 1]``.
    ``1.0`` means the schedule always matches the per-task best;
    ``0.0`` means it is never within ``τ_max``-times the best.

    Parameters
    ----------
    shifted : xarray.DataArray
        Strictly positive ``(schedule, problem)`` array, e.g. the output
        of :func:`shifted_final_min`.
    schedule : str
        Schedule label to score.

    Returns
    -------
    float
        AUC, or ``NaN`` if ``schedule`` is absent or every entry is NaN.
    """
    if schedule not in shifted[SCHEDULE].values.tolist():
        return float("nan")

    sched_vals = shifted.sel({SCHEDULE: schedule}).values.reshape(-1)
    best_vals = shifted.min(dim=SCHEDULE).values.reshape(-1)
    valid = np.isfinite(sched_vals) & np.isfinite(best_vals)
    if not np.any(valid):
        return float("nan")

    rij = sched_vals[valid] / best_vals[valid]
    tau = np.logspace(0, np.log10(_PP_TAU_MAX), 200)
    rho = np.array([np.mean((rij <= t).astype(float)) for t in tau])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201
    return float(trapezoid(rho, x=np.log10(tau))) / np.log10(_PP_TAU_MAX)


#                                                              Win-rate matrix
# =============================================================================


def pairwise_win_rates(
    final_min: xr.DataArray,
) -> dict[str, dict[str, float]]:
    """Pairwise mean ``(final_min[i] < final_min[j])`` over the problem axis.

    Parameters
    ----------
    final_min : xarray.DataArray
        ``(schedule, problem)`` array of final losses. May contain NaNs.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested mapping ``schedule_i -> schedule_j -> fraction``. Self
        comparisons are emitted as ``NaN``.
    """
    schedules = final_min[SCHEDULE].values.tolist()
    vals = final_min.transpose(SCHEDULE, PROBLEM).values  # (n_sched, n_prob)
    out: dict[str, dict[str, float]] = {}
    for i, si in enumerate(schedules):
        row: dict[str, float] = {}
        for j, sj in enumerate(schedules):
            if i == j:
                row[sj] = float("nan")
                continue
            mask = np.isfinite(vals[i]) & np.isfinite(vals[j])
            if not np.any(mask):
                row[sj] = float("nan")
            else:
                row[sj] = float(np.mean(vals[i, mask] < vals[j, mask]))
        out[si] = row
    return out


#                                                                 Top-k picker
# =============================================================================


def top_k_by(values: np.ndarray, k: int, *, ascending: bool) -> list[int]:
    """Return indices of the top-``k`` entries of ``values`` (NaN-safe).

    Parameters
    ----------
    values : numpy.ndarray
        1-D array. NaNs are excluded from the ranking.
    k : int
        Number of indices to return. If fewer finite entries exist, the
        full ordering is returned.
    ascending : bool
        ``True`` selects the smallest entries; ``False`` the largest.

    Returns
    -------
    list[int]
        Indices into the original ``values`` array, ordered from best to
        worst according to ``ascending``.
    """
    a = np.asarray(values, dtype=float).ravel()
    finite = np.where(np.isfinite(a))[0]
    if finite.size == 0:
        return []
    sub = a[finite]
    order = np.argsort(sub) if ascending else np.argsort(-sub)
    take = order[: min(k, sub.size)]
    return [int(finite[i]) for i in take]


#                                                           Full-report builder
# =============================================================================


def build_report(dataloader: DataLoader) -> dict[str, Any]:
    """Compute the AI-friendly JSON-serialisable summary of ``dataloader``.

    This is the implementation of :meth:`DataLoader.report`; it lives
    here so the orchestration logic can be exercised, profiled, and
    extended without touching the :class:`DataLoader` class itself.
    Callers should normally invoke :meth:`DataLoader.report` instead;
    this free function is exposed for advanced use.

    Parameters
    ----------
    dataloader : DataLoader
        The combined loader whose contents to summarise. Read-only. The
        loss-trajectory members (``data``, ``valid_files``,
        ``n_realization``) and the performance members (``ertd_profile``)
        are accessed through its passthroughs.

    Returns
    -------
    dict[str, Any]
        Document conforming to ``schema_version: 2``. Top-level keys:
        ``schema_version``, ``key_findings``, ``dataset``,
        ``task_groups``, ``per_schedule``, ``rankings``, ``grouped``,
        ``observations``, ``observations_summary``. Every observation
        carries a ``magnitude`` in ``[0, 1]`` and observations are
        sorted by severity then magnitude. Every floating-point leaf is
        plain Python ``float`` (with NaN/Inf replaced by ``None``);
        every integer leaf is plain Python ``int``.

    Notes
    -----
    Log and ratio reductions consume :func:`shifted_final_min`, which
    adds ``EPSILON`` to ``final_loss - per_task_shift`` and is therefore
    strictly positive even on tasks whose raw final loss can be negative
    (BBOB and similar shifted-objective benchmarks).
    """
    data = dataloader.data
    task_labels = dataloader.task_labels

    # -- Cube shape -------------------------------------------------------
    schedule_names: list[str] = [
        str(s) for s in data[SCHEDULE].values.tolist()
    ]
    n_schedules = len(schedule_names)
    n_problems = int(data[PROBLEM].size)
    task_id_values = np.asarray(data[TASK_ID].values)
    unique_task_ids, first_idx = np.unique(task_id_values, return_index=True)
    n_tasks = int(unique_task_ids.size)

    has_success_array = SUCCESS in data.data_vars
    has_cost_array = COST in data.data_vars

    task_feature_names: list[str] = [
        str(f)
        for f in task_labels[FEATURE].values.tolist()
        if str(f) not in _EXCLUDED_TASK_FEATURES
    ]
    # One representative row per unique taskID.
    unique_labels = task_labels.isel({PROBLEM: list(first_idx)})

    # -- dataset section --------------------------------------------------
    if TARGET in data.coords and data[TARGET].size > 0:
        target_levels = {
            "n": int(data[TARGET].size),
            "min": float(data[TARGET].min().values),
            "max": float(data[TARGET].max().values),
        }
    else:
        target_levels = {"n": 0, "min": None, "max": None}

    cost_metrics_list = (
        [str(m) for m in data["metric"].values.tolist()]
        if "metric" in data.coords
        else []
    )

    dataset_section: dict[str, Any] = {
        "n_schedules": n_schedules,
        "schedule_names": schedule_names,
        "n_problems": n_problems,
        "n_tasks": n_tasks,
        "n_realizations": int(dataloader.n_realization),
        "budget_axis": {
            "n_steps": int(data[BUDGET].size),
            "min": float(data[BUDGET].min().values),
            "max": float(data[BUDGET].max().values),
        },
        "cost_metrics": cost_metrics_list,
        "target_levels": target_levels,
        "output_dims": (
            [str(o) for o in data[OUTPUT_DIM].values.tolist()]
            if OUTPUT_DIM in data.coords
            else []
        ),
        "task_feature_names": task_feature_names,
        "has_success_array": has_success_array,
        "has_cost_array": has_cost_array,
    }

    # -- task_groups section ----------------------------------------------
    task_groups: dict[str, Any] = {}
    for feat in task_feature_names:
        vals = unique_labels.sel({FEATURE: feat}).values.tolist()
        distinct = set(vals)
        if len(distinct) <= _CARDINALITY_CAP:
            counts: dict[str, int] = {}
            for v in vals:
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
            task_groups[feat] = counts
        else:
            task_groups[feat] = {"cardinality": int(len(distinct))}

    # -- Pre-compute shared scalars / arrays ------------------------------
    shifted = shifted_final_min(
        data=data, task_labels=task_labels
    )  # (schedule, problem), strictly positive
    # Use ``min(BUDGET)`` rather than ``isel(BUDGET=-1)`` because the
    # raw ``output_min`` is the per-iteration population minimum, which
    # is sometimes NaN at the interpolated trajectory boundary even when
    # the optimizer converged earlier.
    final_raw = data[OUTPUT_MIN].min(dim=BUDGET, skipna=True)
    if OUTPUT_DIM in final_raw.dims:
        final_raw = final_raw.isel({OUTPUT_DIM: 0}, drop=True)

    ertd_six = dataloader.ertd_profile(
        cost_values=np.asarray(_ERTD_FEVALS_PER_DIM, dtype=float)
    )
    ertd_six_fevals = ertd_six.sel({"metric": "fevals"}).mean(dim=PROBLEM)

    # Loss-side validity: a (problem, schedule) cell is valid iff its loss
    # curve is not all-NaN over the budget axis. Computed here from
    # ``output_min`` directly rather than via ``DataLoader.valid_files``
    # (which is success-based) so the crash detection stays a function of
    # the loss data, matching this report's historical behaviour.
    valid_per_sched = ~data[OUTPUT_MIN].isnull().all(BUDGET)
    if OUTPUT_DIM in valid_per_sched.dims:
        valid_per_sched = valid_per_sched.isel({OUTPUT_DIM: 0}, drop=True)

    if has_success_array:
        success_mean = data[SUCCESS].mean(dim=PROBLEM)
        if OUTPUT_DIM in success_mean.dims:
            success_mean = success_mean.isel({OUTPUT_DIM: 0}, drop=True)
    target_values = (
        data[TARGET].values.tolist() if TARGET in data.coords else []
    )

    # Per-schedule numpy views (avoid re-indexing the xarray per loop).
    final_vals_by_schedule: dict[str, np.ndarray] = {
        s: np.asarray(final_raw.sel({SCHEDULE: s}).values, dtype=float).ravel()
        for s in schedule_names
    }
    shifted_vals_by_schedule: dict[str, np.ndarray] = {
        s: np.asarray(shifted.sel({SCHEDULE: s}).values, dtype=float).ravel()
        for s in schedule_names
    }

    # -- per_schedule section ---------------------------------------------
    per_schedule: dict[str, dict[str, Any]] = {}
    for s in schedule_names:
        n_valid = int(valid_per_sched.sel({SCHEDULE: s}).sum().values)
        final_vals = final_vals_by_schedule[s]
        shifted_s = shifted.sel({SCHEDULE: s})

        ertd_dict = {
            str(int(c)): float(
                ertd_six_fevals.sel({SCHEDULE: s, "cost_per_dim": c}).values
            )
            for c in _ERTD_FEVALS_PER_DIM
        }

        success_dict: dict[str, float] = {}
        if has_success_array:
            for t in target_values:
                success_dict[f"{float(t):.6g}"] = float(
                    success_mean.sel({SCHEDULE: s, TARGET: t}).values
                )

        per_schedule[s] = {
            "n_valid_problems": n_valid,
            "n_failed_problems": int(n_problems - n_valid),
            "final_loss": descriptive_stats(final_vals),
            "loss_above_optimum_log10": descriptive_stats_log10(shifted_s),
            "ertd_fevals_per_dim": ertd_dict,
            "auc_pp": pp_auc(shifted=shifted, schedule=s),
            "success_fraction_by_target": success_dict,
            "negative_final_loss_fraction": (
                float(np.mean(final_vals < 0))
                if final_vals.size
                else float("nan")
            ),
        }

    # -- rankings section -------------------------------------------------
    # SBS/VBS computed directly from ``final_raw`` so the picks are
    # consistent with the ``min(BUDGET)`` view used elsewhere in the
    # report. ``baselines.calculate_sbs`` uses ``isel(BUDGET=-1)``, which
    # goes NaN at the interpolated trajectory boundary for early-stopping
    # optimizers and would silently disagree.
    final_raw_vals = np.asarray(final_raw.values, dtype=float)

    sched_means_per_task = np.full((n_schedules, n_tasks), np.nan, dtype=float)
    for k, tid in enumerate(unique_task_ids):
        task_mask = task_id_values == tid
        for j in range(n_schedules):
            vals = final_raw_vals[j, task_mask]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                sched_means_per_task[j, k] = float(np.mean(finite))

    sbs_per_task: list[str] = []
    for k in range(n_tasks):
        col = sched_means_per_task[:, k]
        if np.any(np.isfinite(col)):
            sbs_per_task.append(schedule_names[int(np.nanargmin(col))])
        else:
            sbs_per_task.append(schedule_names[0])

    sbs_share: dict[str, float] = {s: 0.0 for s in schedule_names}
    for pick in sbs_per_task:
        sbs_share[pick] += 1.0
    sbs_share = {
        k: (v / float(n_tasks) if n_tasks else float("nan"))
        for k, v in sbs_share.items()
    }

    vbs_share: dict[str, float] = {s: 0.0 for s in schedule_names}
    for p in range(n_problems):
        col = final_raw_vals[:, p]
        if np.any(np.isfinite(col)):
            pick = schedule_names[int(np.nanargmin(col))]
        else:
            pick = schedule_names[0]
        vbs_share[pick] += 1.0
    vbs_share = {
        k: (v / float(n_problems) if n_problems else float("nan"))
        for k, v in vbs_share.items()
    }

    pairwise = pairwise_win_rates(final_raw)

    # Mean log-regret vs per-task SBS, computed in shifted space.
    mean_log_regret: dict[str, float] = {}
    for s in schedule_names:
        s_arr = shifted_vals_by_schedule[s]
        log_regrets: list[float] = []
        for k, tid in enumerate(unique_task_ids):
            task_mask = task_id_values == tid
            if not np.any(task_mask):
                continue
            s_mean = float(np.nanmean(s_arr[task_mask]))
            sbs_choice = sbs_per_task[k]
            sbs_arr = shifted_vals_by_schedule.get(sbs_choice)
            if sbs_arr is None:
                continue
            sbs_mean = float(np.nanmean(sbs_arr[task_mask]))
            if (
                np.isfinite(s_mean)
                and np.isfinite(sbs_mean)
                and s_mean > 0
                and sbs_mean > 0
            ):
                log_regrets.append(float(np.log10(s_mean / sbs_mean)))
        mean_log_regret[s] = (
            float(np.mean(log_regrets)) if log_regrets else float("nan")
        )

    # Per-task SBS scalars used both for top-k lists and observations.
    sbs_shifted_per_task = np.full(n_tasks, np.nan, dtype=float)
    sbs_raw_per_task = np.full(n_tasks, np.nan, dtype=float)
    for k, tid in enumerate(unique_task_ids):
        task_mask = task_id_values == tid
        sbs_choice = sbs_per_task[k]
        sbs_shifted_arr = shifted_vals_by_schedule.get(sbs_choice)
        sbs_raw_arr = final_vals_by_schedule.get(sbs_choice)
        if sbs_shifted_arr is None or sbs_raw_arr is None:
            continue
        sbs_shifted_per_task[k] = float(np.nanmean(sbs_shifted_arr[task_mask]))
        sbs_raw_per_task[k] = float(np.nanmean(sbs_raw_arr[task_mask]))

    def _task_entry(idx: int) -> dict[str, Any]:
        tid = unique_task_ids[idx]
        rep_problem = int(first_idx[idx])
        labels = task_labels.isel({PROBLEM: rep_problem})
        entry: dict[str, Any] = {"task_id": str(tid)}
        for feat in ("fn_name", "dimensionality"):
            if feat in task_feature_names:
                entry[feat] = labels.sel({FEATURE: feat}).item()
        entry["sbs"] = sbs_per_task[idx]
        entry["sbs_final_loss"] = float(sbs_raw_per_task[idx])
        entry["sbs_loss_above_optimum"] = float(sbs_shifted_per_task[idx])
        return entry

    hardest_idx = top_k_by(sbs_shifted_per_task, _TOP_K_TASKS, ascending=False)
    easiest_idx = top_k_by(sbs_shifted_per_task, _TOP_K_TASKS, ascending=True)
    top_hardest = [_task_entry(i) for i in hardest_idx]
    top_easiest = [_task_entry(i) for i in easiest_idx]

    # Per-task schedule spread = std over schedules of log10(mean shifted).
    log_means = np.full((n_tasks, n_schedules), np.nan, dtype=float)
    for k, tid in enumerate(unique_task_ids):
        task_mask = task_id_values == tid
        for j, s in enumerate(schedule_names):
            m = float(np.nanmean(shifted_vals_by_schedule[s][task_mask]))
            if np.isfinite(m) and m > 0:
                log_means[k, j] = np.log10(m)
    spread_per_task = np.nanstd(log_means, axis=1)
    variance_idx = top_k_by(spread_per_task, _TOP_K_TASKS, ascending=False)

    top_variance: list[dict[str, Any]] = []
    for i in variance_idx:
        rep_problem = int(first_idx[i])
        labels = task_labels.isel({PROBLEM: rep_problem})
        entry: dict[str, Any] = {"task_id": str(unique_task_ids[i])}
        for feat in ("fn_name", "dimensionality"):
            if feat in task_feature_names:
                entry[feat] = labels.sel({FEATURE: feat}).item()
        entry["schedule_spread_log10"] = float(spread_per_task[i])
        top_variance.append(entry)

    rankings_section: dict[str, Any] = {
        "sbs_share": sbs_share,
        "vbs_share": vbs_share,
        "pairwise_win_rates": pairwise,
        "mean_log_regret_vs_sbs": mean_log_regret,
        "top_k_hardest_tasks": top_hardest,
        "top_k_easiest_tasks": top_easiest,
        "top_k_tasks_by_variance": top_variance,
    }

    # -- grouped section --------------------------------------------------
    # Track per-feature SBS picks (best schedule by mean ERTD over the cost
    # grid) and the per-(feature, value, schedule) mean-ERTD score. These
    # feed the per-feature observation rules below (regime_change,
    # hard_subset, easy_subset).
    grouped: dict[str, Any] = {}
    per_feature_sbs: dict[str, dict[str, str]] = {}
    per_feature_mean_ertd: dict[str, dict[str, dict[str, float]]] = {}
    for feat in task_feature_names:
        vals_unique = unique_labels.sel({FEATURE: feat}).values.tolist()
        distinct = set(vals_unique)
        if len(distinct) > _CARDINALITY_CAP:
            continue
        feat_block: dict[str, Any] = {}
        feat_sbs: dict[str, str] = {}
        feat_mean_ertd: dict[str, dict[str, float]] = {}
        for v in distinct:
            try:
                sub = dataloader.select_feature({feat: v})
            except Exception:
                continue
            if int(sub.data[PROBLEM].size) == 0:
                continue
            sub_ertd = sub.ertd_profile(
                cost_values=np.asarray(_ERTD_FEVALS_PER_DIM, dtype=float)
            )
            sub_ertd_fevals = sub_ertd.sel({"metric": "fevals"}).mean(
                dim=PROBLEM
            )
            per_s: dict[str, Any] = {}
            sched_mean_ertd: dict[str, float] = {}
            for s in sub.data[SCHEDULE].values.tolist():
                pts = {
                    str(int(c)): float(
                        sub_ertd_fevals.sel(
                            {SCHEDULE: s, "cost_per_dim": c}
                        ).values
                    )
                    for c in _ERTD_FEVALS_PER_DIM
                }
                per_s[str(s)] = {"ertd_fevals_per_dim": pts}
                arr = np.asarray(list(pts.values()), dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    sched_mean_ertd[str(s)] = float(np.mean(finite))
            if sched_mean_ertd:
                feat_sbs[str(v)] = max(
                    sched_mean_ertd, key=lambda k: sched_mean_ertd[k]
                )
                feat_mean_ertd[str(v)] = sched_mean_ertd
            feat_block[str(v)] = {
                "n_problems_in_group": int(sub.data[PROBLEM].size),
                "per_schedule": per_s,
            }
        grouped[feat] = feat_block
        if feat_sbs:
            per_feature_sbs[feat] = feat_sbs
            per_feature_mean_ertd[feat] = feat_mean_ertd

    # -- observations section ---------------------------------------------
    # Each observation has a stable ``code``, a graduated ``severity``
    # (info/warning/error), a ``magnitude`` in [0, 1] used for sorting
    # within a severity bucket, a templated ``message``, and rule-specific
    # extra fields. Severity reflects *how bad* a finding is, not just
    # whether it fired.
    observations: list[dict[str, Any]] = []

    # Mean shifted loss per schedule (positive by construction) — used by
    # ``schedule_worse_than_baseline`` to grade severity in decades rather
    # than against raw sign-sensitive means.
    shifted_mean_by_schedule: dict[str, float] = {
        s: (
            float(np.nanmean(shifted_vals_by_schedule[s]))
            if shifted_vals_by_schedule[s].size
            else float("nan")
        )
        for s in schedule_names
    }

    def _grade(value: float, *, low: float, high: float) -> str:
        """Map a magnitude to ``"info"`` / ``"warning"`` / ``"error"``."""
        if value >= high:
            return "error"
        if value >= low:
            return "warning"
        return "info"

    # --- Rule: optimizer_crashed (graduated by failure fraction) -----
    for s in schedule_names:
        n_failed = int(per_schedule[s]["n_failed_problems"])
        if n_failed <= 0:
            continue
        frac = n_failed / float(n_problems) if n_problems else 0.0
        severity = _grade(frac, low=0.05, high=0.5)
        observations.append(
            {
                "severity": severity,
                "code": "optimizer_crashed",
                "schedule": s,
                "n_problems_affected": n_failed,
                "failure_fraction": float(frac),
                "magnitude": float(frac),
                "message": (
                    f"Schedule {s!r} produced all-NaN loss curves on "
                    f"{n_failed} problem(s) ({frac:.1%})."
                ),
            }
        )

    # --- Rule: schedule_worse_than_baseline (graduated in shifted log-space)
    present_baselines = [b for b in _BASELINE_SCHEDULES if b in schedule_names]
    for b in present_baselines:
        b_sm = shifted_mean_by_schedule.get(b, float("nan"))
        for s in schedule_names:
            if s in _BASELINE_SCHEDULES:
                continue
            s_sm = shifted_mean_by_schedule.get(s, float("nan"))
            if not (
                np.isfinite(s_sm)
                and np.isfinite(b_sm)
                and b_sm > 0
                and s_sm > 0
            ):
                continue
            log_r = math.log10(s_sm / b_sm)
            if log_r <= 0:
                continue  # schedule is no worse than baseline
            severity = _grade(log_r, low=0.5, high=2.0)
            magnitude = float(min(log_r / 6.0, 1.0))
            s_raw = (
                float(np.nanmean(final_vals_by_schedule[s]))
                if final_vals_by_schedule[s].size
                else float("nan")
            )
            b_raw = (
                float(np.nanmean(final_vals_by_schedule[b]))
                if final_vals_by_schedule[b].size
                else float("nan")
            )
            observations.append(
                {
                    "severity": severity,
                    "code": "schedule_worse_than_baseline",
                    "schedule": s,
                    "baseline": b,
                    "schedule_mean": s_raw,
                    "baseline_mean": b_raw,
                    "shifted_log10_ratio": float(log_r),
                    "magnitude": magnitude,
                    "message": (
                        f"Schedule {s!r} mean is {10**log_r:.3g}× "
                        f"worse than baseline {b!r} in shifted space "
                        f"({log_r:.1f} decades)."
                    ),
                }
            )

    # --- Rule: no_schedule_reaches_target (graduated by fraction) ---
    if has_success_array and TARGET in data.coords:
        sf_full = data[SUCCESS].sum(dim=(SCHEDULE, PROBLEM))
        if OUTPUT_DIM in sf_full.dims:
            sf_full = sf_full.isel({OUTPUT_DIM: 0}, drop=True)
        sf_vals = np.asarray(sf_full.values)
        target_arr = np.asarray(data[TARGET].values, dtype=float)
        unreached_mask = sf_vals == 0
        n_unreached = int(np.sum(unreached_mask))
        n_total_targets = int(target_arr.size)
        if n_unreached > 0:
            frac = (
                n_unreached / float(n_total_targets)
                if n_total_targets
                else 0.0
            )
            severity = _grade(frac, low=0.5, high=0.95)
            deepest = float(np.min(target_arr[unreached_mask]))
            observations.append(
                {
                    "severity": severity,
                    "code": "no_schedule_reaches_target",
                    "target": deepest,
                    "n_targets_affected": n_unreached,
                    "fraction_of_targets": float(frac),
                    "magnitude": float(frac),
                    "message": (
                        f"No schedule reached any of {n_unreached} "
                        f"target level(s) ({frac:.0%} of grid); "
                        f"deepest unreached: {deepest:.3g}."
                    ),
                }
            )

    # --- Rule: noisy_schedule (single-row, replaces high_realization_variance)
    median_cov_by_schedule: dict[str, float] = {}
    for s in schedule_names:
        s_final = final_vals_by_schedule[s]
        cov_per_task: list[float] = []
        for tid in unique_task_ids:
            task_mask = task_id_values == tid
            vals = s_final[task_mask]
            if not vals.size:
                continue
            m = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
            if np.isfinite(m) and np.isfinite(sd) and abs(m) > 0:
                cov_per_task.append(sd / abs(m))
        if cov_per_task:
            median_cov_by_schedule[s] = float(np.nanmedian(cov_per_task))

    if median_cov_by_schedule:
        noisiest = max(
            median_cov_by_schedule, key=lambda k: median_cov_by_schedule[k]
        )
        nm = median_cov_by_schedule[noisiest]
        if nm > 1.0:
            observations.append(
                {
                    "severity": "info",
                    "code": "noisy_schedule",
                    "schedule": noisiest,
                    "median_cov": float(nm),
                    "magnitude": float(min(nm / 10.0, 1.0)),
                    "message": (
                        f"Schedule {noisiest!r} is the noisiest "
                        f"(median std/|mean| = {nm:.2f} across tasks)."
                    ),
                }
            )

    # --- Rule: dominant_schedule (with sbs_gap_to_runner_up) ----------
    sorted_log_regret = sorted(
        ((s, v) for s, v in mean_log_regret.items() if np.isfinite(v)),
        key=lambda kv: kv[1],
    )
    sbs_top_lr = sorted_log_regret[0][1] if sorted_log_regret else float("nan")
    runner_up_pair: tuple[str | None, float] = (
        sorted_log_regret[1]
        if len(sorted_log_regret) >= 2
        else (None, float("nan"))
    )
    runner_up_gap = (
        runner_up_pair[1] - sbs_top_lr
        if np.isfinite(runner_up_pair[1]) and np.isfinite(sbs_top_lr)
        else float("nan")
    )

    for s, share in sbs_share.items():
        if not (np.isfinite(share) and share > 0.5):
            continue
        if np.isfinite(runner_up_gap):
            if runner_up_gap > 2.0:
                flavor = "decisive"
            elif runner_up_gap > 0.5:
                flavor = "clear"
            else:
                flavor = "narrow"
            msg = (
                f"Schedule {s!r} is the SBS on {share:.0%} of tasks "
                f"({flavor} win: runner-up {runner_up_pair[0]!r} is "
                f"{runner_up_gap:.1f} decades behind)."
            )
        else:
            msg = f"Schedule {s!r} is the SBS on {share:.0%} of tasks."
        observations.append(
            {
                "severity": "info",
                "code": "dominant_schedule",
                "schedule": s,
                "sbs_share": float(share),
                "runner_up": runner_up_pair[0],
                "sbs_gap_log10": (
                    float(runner_up_gap)
                    if np.isfinite(runner_up_gap)
                    else float("nan")
                ),
                "magnitude": float(share),
                "message": msg,
            }
        )

    # --- Rule: pairwise_dominance (new) -------------------------------
    _PAIRWISE_DOMINANCE_THRESHOLD = 0.85
    dominance_pairs: list[tuple[str, str, float]] = []
    for s1, rates in pairwise.items():
        for s2, rate in rates.items():
            if s1 == s2 or rate is None:
                continue
            if rate > _PAIRWISE_DOMINANCE_THRESHOLD:
                dominance_pairs.append((s1, s2, float(rate)))
    dominance_pairs.sort(key=lambda t: -t[2])
    for s1, s2, rate in dominance_pairs[:5]:
        observations.append(
            {
                "severity": "info",
                "code": "pairwise_dominance",
                "schedule": s1,
                "vs_schedule": s2,
                "win_rate": float(rate),
                "magnitude": float(rate),
                "message": (
                    f"Schedule {s1!r} beats {s2!r} on {rate:.0%} of cells."
                ),
            }
        )

    # --- Per-feature rules: regime_change / hard_subset / easy_subset --
    _HARD_THRESHOLD = 0.2
    _EASY_THRESHOLD = 0.95
    for feat, sbs_by_value in per_feature_sbs.items():
        distinct_sbs = set(sbs_by_value.values())
        n_values = len(sbs_by_value)
        if n_values >= 2 and len(distinct_sbs) >= 2:
            observations.append(
                {
                    "severity": "info",
                    "code": "regime_change",
                    "feature": feat,
                    "sbs_by_value": dict(sbs_by_value),
                    "n_distinct_sbs": int(len(distinct_sbs)),
                    "magnitude": float(len(distinct_sbs) / n_values),
                    "message": (
                        f"SBS varies across {feat}: "
                        + ", ".join(
                            f"{v}→{s!r}" for v, s in sbs_by_value.items()
                        )
                        + "."
                    ),
                }
            )

    for feat, ertd_by_value in per_feature_mean_ertd.items():
        for v, sched_scores in ertd_by_value.items():
            if not sched_scores:
                continue
            scores = list(sched_scores.values())
            m_max = max(scores)
            m_min = min(scores)
            if m_max < _HARD_THRESHOLD:
                observations.append(
                    {
                        "severity": "info",
                        "code": "hard_subset",
                        "feature": feat,
                        "value": v,
                        "best_mean_ertd": float(m_max),
                        "magnitude": float(1.0 - m_max),
                        "message": (
                            f"All schedules struggle at {feat}={v}: "
                            f"best mean ERTD across the cost grid is "
                            f"{m_max:.2f}."
                        ),
                    }
                )
            if m_min > _EASY_THRESHOLD:
                observations.append(
                    {
                        "severity": "info",
                        "code": "easy_subset",
                        "feature": feat,
                        "value": v,
                        "worst_mean_ertd": float(m_min),
                        "magnitude": float(m_min),
                        "message": (
                            f"All schedules solve {feat}={v} easily: "
                            f"worst mean ERTD is {m_min:.2f}."
                        ),
                    }
                )

    # --- Sort: severity DESC, then magnitude DESC ---------------------
    _SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}
    observations.sort(
        key=lambda o: (
            -_SEVERITY_ORDER.get(o.get("severity", "info"), 0),
            -float(o.get("magnitude") or 0.0),
        )
    )

    sev_counter = Counter(o["severity"] for o in observations)
    observations_summary = {
        "error": int(sev_counter.get("error", 0)),
        "warning": int(sev_counter.get("warning", 0)),
        "info": int(sev_counter.get("info", 0)),
    }

    # -- key_findings (top-level TL;DR derived from observations) -----
    key_findings: list[str] = []

    if sorted_log_regret:
        top_schedule, _ = sorted_log_regret[0]
        auc = per_schedule.get(top_schedule, {}).get("auc_pp")
        auc_str = (
            f", AUC-PP {auc:.2f}"
            if isinstance(auc, (int, float)) and np.isfinite(auc)
            else ""
        )
        share = sbs_share.get(top_schedule, 0.0)
        if runner_up_pair[0] is not None and np.isfinite(runner_up_gap):
            gap_str = (
                f"; runner-up {runner_up_pair[0]!r} is "
                f"{runner_up_gap:.1f} decades behind"
            )
        else:
            gap_str = ""
        key_findings.append(
            f"{top_schedule!r} is the strongest schedule (SBS on "
            f"{share:.0%} of tasks{auc_str}{gap_str})."
        )

    failures = [
        o
        for o in observations
        if o["code"] in ("optimizer_crashed", "schedule_worse_than_baseline")
        and o["severity"] in ("warning", "error")
    ]
    if failures:
        f = failures[0]
        if f["code"] == "optimizer_crashed":
            key_findings.append(
                f"{f['schedule']!r} is broken: "
                f"{f['failure_fraction']:.0%} of cells failed."
            )
        else:
            key_findings.append(
                f"{f['schedule']!r} is {f['shifted_log10_ratio']:.1f} "
                f"decades worse than baseline {f['baseline']!r} "
                f"(mean shifted)."
            )

    regime_obs = [o for o in observations if o["code"] == "regime_change"]
    if regime_obs:
        f = max(regime_obs, key=lambda o: o["magnitude"])
        key_findings.append(
            f"Optimizer choice depends on {f['feature']}: "
            f"{f['n_distinct_sbs']} different schedules win across its "
            f"{len(f['sbs_by_value'])} values."
        )

    hard_obs = [o for o in observations if o["code"] == "hard_subset"]
    if hard_obs:
        f = max(hard_obs, key=lambda o: o["magnitude"])
        key_findings.append(
            f"All schedules struggle at {f['feature']}={f['value']}: "
            f"best mean ERTD is {f['best_mean_ertd']:.2f}."
        )

    pd_obs = [o for o in observations if o["code"] == "pairwise_dominance"]
    if pd_obs and len(key_findings) < 5:
        f = pd_obs[0]
        key_findings.append(
            f"{f['schedule']!r} beats {f['vs_schedule']!r} on "
            f"{f['win_rate']:.0%} of cells."
        )

    # -- Final report assembly ----------------------------------------
    # Append the shifted-objective flag now that per_schedule has been
    # populated. ``negative_final_loss_fraction`` is the source of truth.
    any_neg = False
    for blk in per_schedule.values():
        fr = blk.get("negative_final_loss_fraction")
        if isinstance(fr, (int, float)) and np.isfinite(fr) and fr > 0:
            any_neg = True
            break
    dataset_section["uses_shifted_objective"] = any_neg

    report: dict[str, Any] = {
        "schema_version": 2,
        "key_findings": key_findings,
        "dataset": dataset_section,
        "task_groups": task_groups,
        "per_schedule": per_schedule,
        "rankings": rankings_section,
        "grouped": grouped,
        "observations": observations,
        "observations_summary": observations_summary,
    }
    return sanitize_for_json(report)
