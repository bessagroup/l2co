"""Shared record-building pipeline for the databank.

Backend-agnostic helpers that turn finished f3dasm optimization runs into
the per-record xarray payloads the writer stores, and resample a raw loss
history onto the canonical budget grid for reading:

* :func:`_iter_task_runs` -- pivot finished runs by ``task x optimizer``
  and yield the on-disk loss-history path for each present pair;
* :func:`_build_task_records` -- concatenate one task's per-optimizer raw
  loss histories along ``schedule``, derive randomsearch-based ERT targets
  and performance, and split the result into per-record datasets;
* :func:`loss_preprocess` -- resample one raw ``loss_history`` onto a
  common budget grid (integer counters restored).

These are reused by both the writer (creation / append) and the reader
(``load_loss_raw``), so they live in their own module rather than in
either.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import f3dasm

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from l2co_optimizers import OptimizationStep
from l2co_tasks import Task

# Local
from l2co._src.data_validation import _ert_from_quantile, loss_column_name
from l2co._src.databank.provenance import POLICY_OPTIMIZERS
from l2co._src.performance import global_min_breaches
from l2co._src.validation import normalize_budget_grid
from l2co.naming import (
    BUDGET,
    EVALUATION_BUDGET,
    F_STAR,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OPTIMIZATIONSTEP,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    PROVENANCE,
    SCHEDULE,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
    VALUE_KIND,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Payload attribute carrying a task's ``global_min`` breaches from the
# per-task dump step to finalize, which writes them to the dataset's
# ``global_min_breaches.json`` (see ``writer._write_global_min_breaches``).
BREACHES_ATTR = "global_min_breaches"

# Integer counters that nearest-neighbour interpolation promotes to
# float and that must be cast back so dtypes stay consistent with the
# rest of L2CO (see CLAUDE.md "Default float dtype").
_INT_LOSS_VARS = (FEVALS, ITERATIONS, UPDATE_STEP)

#: The loss variables stored as gaps to f* (ADR 0027). ``output_std`` and
#: the counters are not shifted.
GAP_VARS = (OUTPUT_MIN, OUTPUT_MIN_TRUE, OUTPUT_MEAN)

#: ``value_kind`` of a raw file holding gaps / losses (no finite f*).
GAP = "gap"
RAW = "raw"


def f_star_of(task: Task) -> float:
    """The task's f* as float64: its ``global_min``, NaN when it has none."""
    gmin = getattr(task, "global_min", None)
    if gmin is None:
        return float("nan")
    return float(np.asarray(gmin, dtype=np.float64))


def to_gap(ds: xr.Dataset, f_star: float) -> xr.Dataset:
    """Subtract ``f_star`` from the gap variables, in float64.

    A non-finite ``f_star`` leaves the values as they are (stored raw).
    Either way the result is tagged with the ``f_star`` and
    ``value_kind`` attributes the raw netCDF files carry.
    """
    finite = bool(np.isfinite(f_star))
    shifted = {
        name: (ds[name].astype(np.float64) - f_star) if finite else ds[name]
        for name in GAP_VARS
        if name in ds.data_vars
    }
    out = ds.assign(shifted)
    out.attrs = {
        **ds.attrs,
        F_STAR: f_star,
        VALUE_KIND: GAP if finite else RAW,
    }
    return out


def _cell_budget_problem(ds: xr.Dataset, budget: int) -> str | None:
    """Why one cell does not match the evaluation budget, or ``None``.

    ``n`` is the cell's number of iterations (the ``iterations``
    counter's maximum) and its cost per iteration is read from the
    ``fevals`` increments. A cell whose every iteration costs the same
    ``lambda`` evaluations must have ``n * lambda <= B < (n + 1) *
    lambda`` (``n == B`` for one evaluation per iteration); a cell whose
    cost varies (L-BFGS, Nelder-Mead) is budgeted at one evaluation per
    iteration and must have ``n == B`` (ADR 0026, ADR 0027).
    """
    n_values = np.unique(ds[ITERATIONS].max(BUDGET).values)
    if n_values.size != 1 or not np.isfinite(n_values[0]):
        return (
            "its runs made different numbers of iterations "
            f"({n_values.tolist()})"
        )
    n = int(n_values[0])
    # Evaluations per iteration: the fevals increments, the first one
    # from 0 (every step is recorded, so fevals starts at its cost).
    fevals = ds[FEVALS].astype(np.float64)
    steps = fevals.diff(BUDGET, label="upper")
    per_iteration = np.unique(
        np.concatenate(
            [
                np.ravel(fevals.isel({BUDGET: 0}).values),
                np.ravel(steps.values),
            ]
        )
    )
    per_iteration = per_iteration[np.isfinite(per_iteration)]
    if per_iteration.size == 1:
        lam = int(per_iteration[0])
        if lam >= 1 and n * lam <= budget < (n + 1) * lam:
            return None
        return (
            f"it made {n} iteration(s) at {lam} evaluations per iteration "
            f"({n * lam} evaluations), but an evaluation budget of {budget} "
            f"needs {budget // max(lam, 1)} iteration(s)"
        )
    if n == budget:
        return None
    return (
        f"its cost per iteration varies ({int(per_iteration.min())} to "
        f"{int(per_iteration.max())} evaluations), so it is budgeted at one "
        f"evaluation per iteration and needs {budget} iteration(s), but it "
        f"made {n}"
    )


def _task_budget_problem(
    task: Task,
    ds_by_step: dict[OptimizationStep, xr.Dataset],
    budget: int,
) -> str | None:
    """Why a task's cells do not match its budget B, or ``None``."""
    problems = {
        step.name: problem
        for step, ds in ds_by_step.items()
        if (problem := _cell_budget_problem(ds, budget)) is not None
    }
    if not problems:
        return None
    detail = "; ".join(f"'{k}': {v}" for k, v in problems.items())
    return (
        f"task {task.hash}: cells do not match the task's evaluation "
        f"budget B = {budget} -- {detail}"
    )


_BUDGET_REMEDY = (
    "Every run of a task gets the same B (l2co ADR 0026); re-run these "
    "cells with the budget function's iterations, or pass the B they "
    "were run with (l2co ADR 0027)."
)


def _align_cell_lengths(
    ds_by_step: dict[OptimizationStep, xr.Dataset],
) -> dict[OptimizationStep, xr.Dataset]:
    """A task's cells at one common length, for scoring them together.

    Under the evaluation budget a task's cells record different numbers
    of steps (B for random search, about B / lambda for a population
    method), and the run loop's histories carry no ``budget``
    coordinate, so they cannot be combined as they are. Each cell gets a
    ``budget`` index and is extended to the longest cell by repeating
    its last recorded step: a run that has ended stays at its final
    state (its evaluations, its best value), which leaves its ERT, its
    successes and its breaches unchanged. Values inside a run are not
    touched. Only the combined scoring sees the extension; every cell
    is stored, and its raw history written, as it ran.
    """
    longest = max(ds.sizes[BUDGET] for ds in ds_by_step.values())
    out = {}
    for step, ds in ds_by_step.items():
        n = ds.sizes[BUDGET]
        indexed = ds.assign_coords({BUDGET: np.arange(n)})
        if n < longest:
            indexed = indexed.reindex(
                {BUDGET: np.arange(longest)}, method="ffill"
            )
        out[step] = indexed
    return out


def _check_cell_budgets(
    task: Task,
    ds_by_step: dict[OptimizationStep, xr.Dataset],
    budget: int,
) -> None:
    """Refuse a task with any cell that does not match its budget B.

    Raises
    ------
    ValueError
        Naming the task, B and every mismatching schedule.
    """
    problem = _task_budget_problem(task, ds_by_step, budget)
    if problem is not None:
        raise ValueError(f"{problem}. {_BUDGET_REMEDY}")


def _budget_counters(path: Path) -> xr.Dataset:
    """The two counters the budget check reads, from one raw loss file.

    Only ``iterations`` and ``fevals`` are loaded, not the whole history,
    so checking every cell of a dump stays cheap.
    """
    with xr.open_dataset(path) as ds:
        return ds[[ITERATIONS, FEVALS]].load()


def check_run_budgets(runs: list, budgets: dict[str, int]) -> None:
    """Refuse a dump with any cell that does not match its task's B.

    Run by the template step, before anything is written, deleted or
    marked, so a mismatch leaves the store as it was (ADR 0027). Every
    task is checked and every mismatch is reported at once.

    Parameters
    ----------
    runs : list
        ``[(task, {step: loss_path})]``, as :func:`_iter_task_runs`
        yields.
    budgets : dict[str, int]
        The evaluation budget B of each task, by task hash.

    Raises
    ------
    ValueError
        Naming every task, its B and every mismatching schedule.
    """
    problems = []
    for task, step_paths in runs:
        counters = {
            step: _budget_counters(path) for step, path in step_paths.items()
        }
        problem = _task_budget_problem(task, counters, budgets[task.hash])
        if problem is not None:
            problems.append(problem)
    if problems:
        raise ValueError(
            "; ".join(problems) + f". {_BUDGET_REMEDY} Nothing was written."
        )


def loss_preprocess(ds: xr.Dataset, budget_steps: int) -> xr.Dataset:
    """Resample one raw ``loss_history`` file onto the canonical grid.

    Delegates to :func:`l2co.validation.normalize_budget_grid` so the
    ``[-step, 0, step', ...]`` slot-0 ("pre-optimization") convention
    used throughout L2CO is preserved -- the same normalization the
    legacy write-time pipeline applied via ``validation_fn``. Nearest
    interpolation keeps the recorded 0th and final iterations exact, but
    promotes the integer counters (``fevals``, ``iterations``,
    ``update_step``) to float; they are cast back to ``int`` here.

    The cast goes through ``.round()`` first: although ``method="nearest"``
    only copies existing sample values (so the integer is conceptually
    exact), the float64 round-trip can leave a count at e.g. ``3.9999999``,
    and a bare ``.astype(int)`` truncates toward zero -- silently dropping
    it to ``3``. Rounding first restores the intended integer before the
    truncation.

    Parameters
    ----------
    ds : xr.Dataset
        A single ``loss_history`` dataset with a raw ``budget`` dim.
    budget_steps : int
        Number of points on the common budget grid (``>= 3``).

    Returns
    -------
    xr.Dataset
        ``ds`` resampled to ``budget_steps`` points with integer
        counters restored.
    """
    ds = normalize_budget_grid(ds, budget_steps)
    for var in _INT_LOSS_VARS:
        if var in ds:
            ds[var] = ds[var].round().astype(int)
    return ds


def _iter_task_runs(
    experiment_data: f3dasm.ExperimentData,
    tasks_experiment_data: f3dasm.ExperimentData,
    schedules_experiment_data: f3dasm.ExperimentData,
) -> Iterator[tuple[Any, dict[OptimizationStep, Path]]]:
    """Yield ``(task, {optimizer_step: loss_history_path})`` per task.

    Mirrors the data-loading preamble of
    :func:`l2co.create_combined_odata`: select finished runs, resolve the
    on-disk loss-history reference for each, and pivot by
    ``task x optimizer``. Missing ``(task, optimizer)`` combinations are
    omitted from the per-task mapping (they become unwritten cells -- read
    back as ``NaN`` -- rather than placeholders).
    """
    experiment_data = experiment_data.select_with_status("finished")
    experiment_data = experiment_data.reset_index()

    df_input, df_output = experiment_data.to_pandas(keep_references=True)
    df_output = df_output.map(
        lambda row: (
            Path(experiment_data.project_dir)
            / "experiment_data"
            / row.reference
        ),
        na_action="ignore",
    )

    tasks = [es.output_data.get("task") for _, es in tasks_experiment_data]
    schedules = [
        es.input_data["optimizer"] for _, es in schedules_experiment_data
    ]

    temp_df = pd.DataFrame(
        {
            TASK_ID: df_input["task_idx"].apply(lambda i: tasks[int(i)]),
            SCHEDULE: df_input["optimizer_idx"].apply(
                lambda i: schedules[int(i)]
            ),
            loss_column_name: df_output[loss_column_name],
        }
    )
    pivot_table = temp_df.pivot(
        index=TASK_ID, columns=SCHEDULE, values=loss_column_name
    )

    for task, loss_history in pivot_table.iterrows():
        present = loss_history.dropna()
        yield task, {step: path for step, path in present.items()}


def _task_dataset(
    task: Any,
    target_value: xr.DataArray,
    evaluation_budget: int | None = None,
) -> xr.Dataset:
    """Build the per-task payload.

    Carries ``task_labels`` (the task tag plus ``global_min``, stringified
    the same way :func:`l2co.create_combined_odata` does), the per-task
    ERT ``target_value`` echoed by
    :func:`l2co._src.performance.calculate_ert`, the float64 ``f_star``
    the stored losses are gaps to (NaN: stored raw; ADR 0027) and, when
    given, the task's ``evaluation_budget``.
    """
    row = {**task.tag, GLOBAL_MIN: task.global_min}
    features = list(row.keys())
    values = np.array([[str(row[f]) for f in features]], dtype=object)
    task_labels = xr.DataArray(
        values,
        coords={TASK_ID: [task.hash], FEATURE: features},
        dims=[TASK_ID, FEATURE],
    ).astype(str)
    tv = target_value.expand_dims({TASK_ID: [task.hash]})
    out = xr.Dataset(
        {
            TASK_LABELS: task_labels,
            TARGET_VALUE: tv,
            F_STAR: xr.DataArray(
                np.array([f_star_of(task)], dtype=np.float64),
                coords={TASK_ID: [task.hash]},
                dims=[TASK_ID],
            ),
        }
    )
    if evaluation_budget is not None:
        out[EVALUATION_BUDGET] = xr.DataArray(
            np.array([evaluation_budget], dtype=np.int64),
            coords={TASK_ID: [task.hash]},
            dims=[TASK_ID],
        )
    return out


def _optimizer_dataset(
    step: OptimizationStep, provenance: str = ""
) -> xr.Dataset:
    """Build the per-optimizer payload (recipe + provenance record).

    Stores the full :class:`OptimizationStep` recipe as a JSON string on
    the ``schedule`` axis (the human-readable name is the coord value),
    plus the serialized provenance record for policy schedules (an
    empty string for static optimizers — see
    :mod:`l2co._src.databank.provenance`).
    """

    def _da(value: str) -> xr.DataArray:
        return xr.DataArray(
            [value],
            coords={SCHEDULE: [step.name]},
            dims=[SCHEDULE],
        )

    return xr.Dataset(
        {
            OPTIMIZATIONSTEP: _da(json.dumps(step.to_dict())),
            PROVENANCE: _da(provenance),
        }
    )


def _check_noise_free_series(
    task: Task, ds_by_step: dict[OptimizationStep, xr.Dataset]
) -> None:
    """Refuse a task whose runs would be scored on mixed series.

    A noisy task is scored on its noise-free series
    (:func:`l2co._src.performance.scoring_loss`; ADR 0024). Both checks
    below guard against a run that lacks it being scored, silently, on
    its observed values instead:

    * either every schedule of the task carries ``output_min_true`` or
      none does -- concatenated, a schedule without it would read back
      as NaN there and score on its observed series next to the others;
    * a task with a noise-free loss (``task.loss_fn_true``) must carry
      it. Runs made before the series existed, or through a path that
      does not record it (rl2co's environment; a direct
      ``init_run_state`` without ``record_noise_free``), are refused.

    A third keeps the store's schema a function of its tasks: a task
    *without* a noise-free loss must not carry the series. The writer
    sizes the store from the first task with one
    (:func:`l2co._src.databank.writer._template_sample_run`), so a
    series on any other task could meet a store without the variable.

    Raises
    ------
    ValueError
        On any violation, naming the schedules concerned.
    """
    carrying = [
        step.name
        for step, ds in ds_by_step.items()
        if OUTPUT_MIN_TRUE in ds.data_vars
    ]
    if carrying and len(carrying) != len(ds_by_step):
        lacking = [
            step.name for step in ds_by_step if step.name not in carrying
        ]
        raise ValueError(
            f"task {task.hash}: the noise-free series "
            f"'{OUTPUT_MIN_TRUE}' is on schedules {sorted(carrying)} but "
            f"not on {sorted(lacking)}. A noisy task is scored on it, so "
            "every schedule of the task must carry it; re-run the "
            "schedules without it (see l2co ADR 0024)."
        )
    noisy = getattr(task, "loss_fn_true", None) is not None
    if carrying and not noisy:
        raise ValueError(
            f"task {task.hash} has no loss_fn_true but its schedules "
            f"{sorted(carrying)} carry the noise-free series "
            f"'{OUTPUT_MIN_TRUE}'. A store holds the series for its noisy "
            "tasks only, those with a loss_fn_true; see l2co ADR 0024."
        )
    if not carrying and noisy:
        raise ValueError(
            f"task {task.hash} is noisy (it has a loss_fn_true) but none "
            f"of its runs carries the noise-free series "
            f"'{OUTPUT_MIN_TRUE}', which noisy tasks are scored on. Run "
            "it through RolloutWrapper (or init_run_state with "
            "record_noise_free=True); see l2co ADR 0024."
        )


def _build_task_records(
    task: Task,
    ds_by_step: dict[OptimizationStep, xr.Dataset],
    easiest_quantile: float,
    existing_targets: np.ndarray | None = None,
    evaluation_budget: int | None = None,
) -> tuple[
    xr.Dataset,
    dict[int, xr.Dataset],
    dict[int, xr.Dataset],
    dict[int, xr.Dataset],
]:
    """Decompose one task's runs into per-record datasets.

    Pure (no filesystem / f3dasm): concatenates the task's per-optimizer
    raw loss histories along ``schedule`` to derive randomsearch-based
    ERT targets and performance, then splits everything into the four
    per-record payloads.

    Parameters
    ----------
    task : Task
        Task object exposing ``.hash``, ``.tag`` and ``.global_min``.
    ds_by_step : dict[OptimizationStep, xr.Dataset]
        Per-optimizer raw loss histories (dims
        ``(realization, budget, output_dim)``).
    easiest_quantile : float
        Forwarded to the per-task ERT target computation (only used
        when ``existing_targets`` is ``None``).
    existing_targets : np.ndarray or None, optional
        ERT target quality values read from an already-present task entry.
        When given, performance is scored against these targets instead of
        recomputing them from this run's randomsearch baseline, so
        optimizers appended to an existing task stay consistent with the
        records written earlier. When ``None`` (a new task) the
        randomsearch-derived target is computed as before.
    evaluation_budget : int or None, optional
        The task's evaluation budget B (ADR 0027). When given, every
        cell is checked against it (:func:`_cell_budget_problem`), the
        scores ignore points past it
        (:func:`l2co._src.performance.calculate_ert`) and ``task_ds``
        carries it. The breach check still reads every point. ``None``
        (default): no check, no cut, no variable.

    Returns
    -------
    tuple
        ``(task_ds, loss_by_hash, perf_by_hash, opt_by_hash)`` where the
        three dicts are keyed by ``OptimizationStep.hash``. When
        ``existing_targets`` was supplied, ``task_ds`` simply echoes
        those targets. When a schedule's lowest loss is below the
        task's ``global_min`` (:func:`l2co._src.performance.
        global_min_breaches`), ``task_ds.attrs[BREACHES_ATTR]`` holds
        a JSON record: the task's ``tag``, its ``global_min`` and the
        breaching ``schedules``. Those losses still score as reaching
        ``global_min``. On a noisy task, targets, performance and
        breaches are those of the noise-free ``output_min_true``
        (ADR 0024); the loss payloads keep both series.

        The loss payloads hold gaps: once the scores, targets and
        breaches above are computed from the losses, f* (the task's
        ``global_min``) is subtracted in float64 from ``output_min``,
        ``output_min_true`` and ``output_mean`` (:func:`to_gap`; a task
        without a finite f* keeps its losses). Each payload carries
        ``f_star`` and ``value_kind`` attributes, and ``task_ds`` an
        ``f_star`` variable (ADR 0027).

    Raises
    ------
    ValueError
        A policy schedule (``optimizer`` in
        :data:`~l2co._src.databank.provenance.POLICY_OPTIMIZERS`)
        arrives without a provenance record on its loss history — the
        dump-time admissibility backstop (the evaluation pipeline
        should have failed fast much earlier). Also when the task's
        runs would be scored on mixed series (see
        :func:`_check_noise_free_series`), and when a cell does not
        match ``evaluation_budget``.
    """
    for step, ds in ds_by_step.items():
        if step.optimizer in POLICY_OPTIMIZERS and not str(
            ds.attrs.get(PROVENANCE, "")
        ):
            raise ValueError(
                f"Schedule '{step.name}' (optimizer='{step.optimizer}') "
                "is a policy artifact but its loss history carries no "
                f"'{PROVENANCE}' record: not admissible to the databank. "
                "Evaluate through a pipeline that loads the checkpoint's "
                "provenance manifest and attaches the record (there is "
                "no bypass)."
            )

    _check_noise_free_series(task, ds_by_step)
    if evaluation_budget is not None:
        _check_cell_budgets(task, ds_by_step, int(evaluation_budget))

    gmin = task.global_min if task.global_min is not None else np.nan
    aligned = _align_cell_lengths(ds_by_step)
    b = xr.concat(
        [
            ds.expand_dims({SCHEDULE: [step.name]})
            for step, ds in aligned.items()
        ],
        dim=SCHEDULE,
        coords="minimal",
        compat="override",
        join="outer",
    ).assign_coords({GLOBAL_MIN: gmin})

    perf_all = _ert_from_quantile(
        b,
        easiest_quantile=easiest_quantile,
        targets=existing_targets,
        evaluation_budget=evaluation_budget,
    )
    task_ds = _task_dataset(task, perf_all[TARGET_VALUE], evaluation_budget)
    breaches = global_min_breaches(b, gmin)
    if breaches:
        task_ds.attrs[BREACHES_ATTR] = json.dumps(
            {"task": task.tag, GLOBAL_MIN: gmin, "schedules": breaches},
            default=str,
        )
    perf_vars = perf_all.drop_vars(TARGET_VALUE)
    # Only now, with every score taken from the losses: store gaps.
    f_star = f_star_of(task)

    loss_by_hash: dict[int, xr.Dataset] = {}
    perf_by_hash: dict[int, xr.Dataset] = {}
    opt_by_hash: dict[int, xr.Dataset] = {}
    for step, ds in ds_by_step.items():
        loss_by_hash[step.hash] = to_gap(ds, f_star).expand_dims(
            {TASK_ID: [task.hash], SCHEDULE: [step.name]}
        )
        perf_by_hash[step.hash] = perf_vars.sel(
            {SCHEDULE: [step.name]}
        ).expand_dims({TASK_ID: [task.hash]})
        opt_by_hash[step.hash] = _optimizer_dataset(
            step, provenance=str(ds.attrs.get(PROVENANCE, ""))
        )
    return task_ds, loss_by_hash, perf_by_hash, opt_by_hash
