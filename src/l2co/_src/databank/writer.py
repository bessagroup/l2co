"""Write a per-record databank directly from finished optimization runs.

Replaces the monolithic ``entire_dataset.odata`` step for the new layout:
instead of merging every record into one NetCDF, this writes one file per
task, per optimizer, and per ``(task, optimizer)`` pair into the four
typed subfolders (see :mod:`l2co._src.databank.layout`).

The per-task processing reuses the same pipeline as
:func:`l2co.create_combined_odata`: the runs are pivoted by
``task x optimizer``, each task's per-optimizer loss histories are
concatenated along ``schedule`` to derive randomsearch-based ERT targets
and performance via :func:`l2co._src.data_validation._ert_from_quantile`.
The results are then split back into per-record files. ``loss_history``
files keep the *raw* budget axis (resampled at read time), so no budget
interpolation happens here.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import f3dasm

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from l2co_tasks import Task
from tqdm import tqdm

# Local
from l2co._src.data_validation import _ert_from_quantile, loss_column_name
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.performance import DEFAULT_EASIEST_QUANTILE
from l2co.naming import (
    FEATURE,
    GLOBAL_MIN,
    OPTIMIZATIONSTEP,
    SCHEDULE,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
)

from . import layout
from .tasksets import save_taskset

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)


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
    omitted from the per-task mapping (they become "missing" files on
    disk rather than NaN placeholders).
    """
    experiment_data = experiment_data.select_with_status("finished")
    experiment_data = experiment_data.reset_index()

    df_input, df_output = experiment_data.to_pandas(keep_references=True)
    df_output = df_output.map(
        lambda row: Path(experiment_data.project_dir)
        / "experiment_data"
        / row.reference,
        na_action="ignore",
    )

    tasks = [es.output_data["task"] for _, es in tasks_experiment_data]
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
    """Build the per-task ``task/<taskID>.nc`` payload.

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


def _optimizer_dataset(step: OptimizationStep) -> xr.Dataset:
    """Build the per-optimizer ``optimizer/<hash>.nc`` payload.

    Stores the full :class:`OptimizationStep` recipe as a JSON string on
    the ``schedule`` axis (the human-readable name is the coord value).
    """
    recipe = xr.DataArray(
        [json.dumps(step.to_dict())],
        coords={SCHEDULE: [step.name]},
        dims=[SCHEDULE],
    )
    return xr.Dataset({OPTIMIZATIONSTEP: recipe})


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
        ERT target quality values read from an already-present
        ``task/<taskID>.nc``. When given, performance is scored against
        these targets instead of recomputing them from this run's
        randomsearch baseline, so optimizers appended to an existing
        task stay consistent with the records written earlier. When
        ``None`` (a new task) the randomsearch-derived target is
        computed as before.

    Returns
    -------
    tuple
        ``(task_ds, loss_by_hash, perf_by_hash, opt_by_hash)`` where the
        three dicts are keyed by ``OptimizationStep.hash``. When
        ``existing_targets`` was supplied, ``task_ds`` simply echoes
        those targets (the caller leaves the stored file untouched).
    """
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
        opt_by_hash[step.hash] = _optimizer_dataset(step)
    return task_ds, loss_by_hash, perf_by_hash, opt_by_hash


def _should_write(path: Path, mode: str) -> bool:
    """Return whether ``path`` should be (over)written under ``mode``."""
    if mode == "append" and path.exists():
        logger.warning(f"dump_databank: skipping existing {path} (append).")
        return False
    return True


def _existing_target_value(task_p: Path, task_hash: str) -> np.ndarray | None:
    """Return the ERT ``target_value`` stored in ``task_p``, or ``None``.

    A ``task/<taskID>.nc`` written by an earlier dump is authoritative
    for its per-task ERT targets: optimizers appended later must be
    scored against the *same* targets, not a fresh randomsearch-derived
    one (which would differ whenever the appending run's randomsearch
    baseline differs, or is absent -- see
    :func:`l2co._src.performance.compute_task_targets`). Returns the
    stored 1-D target quality values when the file exists, or ``None``
    when the task is new so :func:`_build_task_records` falls back to
    the randomsearch derivation.
    """
    if not task_p.exists():
        return None
    with xr.load_dataset(task_p) as ds:
        return ds[TARGET_VALUE].sel({TASK_ID: task_hash}).values


def dump_databank(
    experiment_data: f3dasm.ExperimentData,
    tasks_experiment_data: f3dasm.ExperimentData,
    schedules_experiment_data: f3dasm.ExperimentData,
    location: Literal["local", "brown"],
    dataset_name: str,
    *,
    mode: Literal["error", "replace", "append"] = "error",
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
    taskset: str | None = None,
) -> Path:
    """Split finished optimization runs into a per-record databank.

    For each task, the per-optimizer loss histories are concatenated
    along ``schedule`` to derive randomsearch-based ERT targets and
    performance (via
    :func:`l2co._src.data_validation._ert_from_quantile`). The result is
    written as one ``task/<taskID>.nc`` (labels + ``target_value``), one
    ``loss_history/loss_<taskID>_<optimizer_hash>.nc`` and one
    ``performance/perf_<taskID>_<optimizer_hash>.nc`` per present
    ``(task, optimizer)`` pair, and one ``optimizer/<optimizer_hash>.nc``
    per distinct optimizer. ``loss_history`` keeps the raw budget axis.

    Parameters
    ----------
    experiment_data : f3dasm.ExperimentData
        Finished optimization runs (one row per ``(task, optimizer)``),
        carrying integer ``task_idx`` / ``optimizer_idx`` that index into
        the two source ExperimentData below.
    tasks_experiment_data : f3dasm.ExperimentData
        Source whose ``task`` output column is indexed by
        ``experiment_data["task_idx"]``.
    schedules_experiment_data : f3dasm.ExperimentData
        Source whose ``optimizer`` input column is indexed by
        ``experiment_data["optimizer_idx"]``.
    location : {"local", "brown"}
        Which databank to write to.
    dataset_name : str
        Name of the dataset folder to create.
    mode : {"error", "replace", "append"}, optional
        How to handle an existing dataset folder. ``"error"`` (default)
        raises. ``"replace"`` wipes and rewrites. ``"append"`` keeps
        existing files and writes only missing ones (skip-with-warning
        per file). Under ``"append"`` a task whose ``task/<taskID>.nc``
        already exists keeps that file's per-task ERT ``target_value``:
        the newly appended optimizers are scored against the stored
        targets rather than a target recomputed from this run's
        randomsearch baseline, so old and new performance files stay
        comparable. A task that is new to the databank computes and
        writes its randomsearch-derived target as usual.
    easiest_quantile : float, optional
        Forwarded to the per-task ERT target computation, by default
        :data:`~l2co._src.performance.DEFAULT_EASIEST_QUANTILE`.
    taskset : str | None, optional
        If given, also write ``tasksets/<taskset>.txt`` listing every
        task hash written.

    Returns
    -------
    pathlib.Path
        Absolute path to the written dataset folder.

    Raises
    ------
    FileExistsError
        The dataset folder exists and ``mode="error"``.
    ValueError
        ``location`` is unknown.
    """
    rootdir = layout.databank_root(location, dataset_name)
    if rootdir.exists():
        if mode == "error":
            raise FileExistsError(
                f"Dataset folder {rootdir} already exists; pass "
                "mode='replace' to wipe and rewrite, or mode='append' "
                "to extend it."
            )
        if mode == "replace":
            logger.info(f"dump_databank: removing existing folder {rootdir}")
            shutil.rmtree(rootdir)
    for sub in (
        layout.TASK_DIR,
        layout.OPTIMIZER_DIR,
        layout.LOSS_DIR,
        layout.PERF_DIR,
    ):
        (rootdir / sub).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"dump_databank: writing dataset '{dataset_name}' to {rootdir} "
        f"(mode={mode})"
    )

    written_tasks: list[str] = []
    written_optimizers: set[int] = set()

    runs = list(
        _iter_task_runs(
            experiment_data,
            tasks_experiment_data,
            schedules_experiment_data,
        )
    )

    for task, step_paths in tqdm(runs, total=len(runs)):
        if not step_paths:
            continue

        # --- Load each present run eagerly and close the handle ---
        ds_by_step = {
            step: xr.load_dataset(path) for step, path in step_paths.items()
        }

        # An already-present task file is authoritative for its ERT
        # targets; reuse them so appended optimizers are scored against
        # the same targets as the records written earlier.
        task_p = layout.task_path(rootdir, task.hash)
        existing_targets = _existing_target_value(task_p, task.hash)

        task_ds, loss_by_hash, perf_by_hash, opt_by_hash = _build_task_records(
            task,
            ds_by_step,
            easiest_quantile,
            existing_targets=existing_targets,
        )

        # --- task/<taskID>.nc (labels + per-task target_value) ---
        # Only written for a new task; an existing file is left untouched
        # (its target_value was just reused above).
        if existing_targets is None:
            task_ds.to_netcdf(task_p, engine="netcdf4")
        else:
            logger.info(
                f"dump_databank: reusing stored target_value for task "
                f"{task.hash} ({task_p.name} already present)."
            )
        written_tasks.append(task.hash)

        # --- per (task, optimizer): loss_history + performance ---
        for opt_hash in loss_by_hash:
            loss_p = layout.loss_path(rootdir, task.hash, opt_hash)
            if _should_write(loss_p, mode):
                loss_by_hash[opt_hash].to_netcdf(loss_p, engine="netcdf4")

            perf_p = layout.perf_path(rootdir, task.hash, opt_hash)
            if _should_write(perf_p, mode):
                perf_by_hash[opt_hash].to_netcdf(perf_p, engine="netcdf4")

            # --- optimizer/<hash>.nc (once per distinct optimizer) ---
            if opt_hash not in written_optimizers:
                opt_p = layout.optimizer_path(rootdir, opt_hash)
                if _should_write(opt_p, mode):
                    opt_by_hash[opt_hash].to_netcdf(opt_p, engine="netcdf4")
                written_optimizers.add(opt_hash)

    if taskset is not None:
        save_taskset(rootdir, taskset, written_tasks)

    logger.info(
        f"dump_databank: done -- {len(written_tasks)} task(s), "
        f"{len(written_optimizers)} optimizer(s) -> {rootdir}"
    )
    return rootdir.resolve()
