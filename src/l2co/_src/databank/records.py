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
from l2co_tasks import Task

# Local
from l2co._src.data_validation import _ert_from_quantile, loss_column_name
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.validation import normalize_budget_grid
from l2co.naming import (
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OPTIMIZATIONSTEP,
    PROVENANCE,
    SCHEDULE,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)

from .provenance import POLICY_OPTIMIZERS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Integer counters that nearest-neighbour interpolation promotes to
# float and that must be cast back so dtypes stay consistent with the
# rest of L2CO (see CLAUDE.md "Default float dtype").
_INT_LOSS_VARS = (FEVALS, ITERATIONS, UPDATE_STEP)


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

    tasks = [
        es.output_data.get("task") for _, es in tasks_experiment_data
    ]
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


def _task_dataset(task: Any, target_value: xr.DataArray) -> xr.Dataset:
    """Build the per-task payload (``task_labels`` + ``target_value``).

    Carries ``task_labels`` (the task tag plus ``global_min``, stringified
    the same way :func:`l2co.create_combined_odata` does) and the per-task
    ERT ``target_value`` echoed by
    :func:`l2co._src.performance.calculate_ert`.
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
    return xr.Dataset({TASK_LABELS: task_labels, TARGET_VALUE: tv})


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


def _build_task_records(
    task: Task,
    ds_by_step: dict[OptimizationStep, xr.Dataset],
    easiest_quantile: float,
    existing_targets: np.ndarray | None = None,
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

    Returns
    -------
    tuple
        ``(task_ds, loss_by_hash, perf_by_hash, opt_by_hash)`` where the
        three dicts are keyed by ``OptimizationStep.hash``. When
        ``existing_targets`` was supplied, ``task_ds`` simply echoes
        those targets.

    Raises
    ------
    ValueError
        A policy schedule (``optimizer`` in
        :data:`~l2co._src.databank.provenance.POLICY_OPTIMIZERS`)
        arrives without a provenance record on its loss history — the
        dump-time admissibility backstop (the evaluation pipeline
        should have failed fast much earlier).
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

    gmin = task.global_min if task.global_min is not None else np.nan
    b = xr.concat(
        [
            ds.expand_dims({SCHEDULE: [step.name]})
            for step, ds in ds_by_step.items()
        ],
        dim=SCHEDULE,
        coords="minimal",
        compat="override",
        join="outer",
    ).assign_coords({GLOBAL_MIN: gmin})

    perf_all = _ert_from_quantile(
        b, easiest_quantile=easiest_quantile, targets=existing_targets
    )
    task_ds = _task_dataset(task, perf_all[TARGET_VALUE])
    perf_vars = perf_all.drop_vars(TARGET_VALUE)

    loss_by_hash: dict[int, xr.Dataset] = {}
    perf_by_hash: dict[int, xr.Dataset] = {}
    opt_by_hash: dict[int, xr.Dataset] = {}
    for step, ds in ds_by_step.items():
        loss_by_hash[step.hash] = ds.expand_dims(
            {TASK_ID: [task.hash], SCHEDULE: [step.name]}
        )
        perf_by_hash[step.hash] = perf_vars.sel(
            {SCHEDULE: [step.name]}
        ).expand_dims({TASK_ID: [task.hash]})
        opt_by_hash[step.hash] = _optimizer_dataset(
            step, provenance=str(ds.attrs.get(PROVENANCE, ""))
        )
    return task_ds, loss_by_hash, perf_by_hash, opt_by_hash
