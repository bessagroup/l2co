"""Write a databank from finished optimization runs.

Reuses the shared f3dasm preamble and per-task ERT pipeline
(:func:`l2co._src.databank.records._iter_task_runs`,
:func:`l2co._src.databank.records._build_task_records`) and then writes
**two** things per ``(task, optimizer)`` pair:

#. the *raw* loss history (original variable-length ``budget``) to a
   per-pair netCDF under ``loss_history_raw/`` -- the source of truth for
   regeneration; and
#. the loss history *interpolated* to a fixed ``budget_steps`` grid,
   region-written into the consolidated zarr store alongside the
   performance, task and optimizer arrays.

Creation (``mode="error"`` / ``"replace"``) streams one ``(task,
optimizer)`` cell at a time, so the full dense array is never held in
memory. ``mode="append"`` rebuilds the store over the union of old and
new ``taskID`` / ``schedule`` coordinates into a temporary store and
swaps it in atomically, copying untouched cells from the old store one
at a time (also memory-light, and it sidesteps the fixed-length-string
truncation that an in-place coordinate resize would cause).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import logging
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

# Third-party
import xarray as xr
import zarr
from l2co_tasks import Task
from tqdm import tqdm

# Local
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.performance import DEFAULT_EASIEST_QUANTILE
from l2co._src.validation import normalize_budget_grid
from l2co.naming import (
    FEATURE,
    OPTIMIZATIONSTEP,
    SCHEDULE,
    SUCCESS,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
)

from . import layout
from .records import _build_task_records, _iter_task_runs
from .schema import big_template, downcast_floats, to_region_cell
from .tasksets import save_taskset

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)


@contextmanager
def silence_zarr_warnings():
    """Suppress zarr-v3 maturity warnings around store I/O.

    zarr 3 emits :class:`UserWarning` / ``UnstableSpecificationWarning``
    for consolidated metadata and fixed-length string dtypes (both used
    deliberately here). They are noisy and not actionable for the
    databank, so they are silenced narrowly around our reads/writes.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated.*")
        warnings.filterwarnings("ignore", message=".*Zarr V3 specification.*")
        yield


def _interp_loss(loss_raw: xr.Dataset, budget_steps: int) -> xr.Dataset:
    """Resample a raw loss cell onto the fixed ``budget_steps`` grid.

    Unlike :func:`l2co._src.databank.records.loss_preprocess`, the integer
    counters are **not** cast back to ``int``: the consolidated store
    keeps every numerical variable as ``float`` (a missing cell must be
    representable as ``NaN``).
    """
    return normalize_budget_grid(loss_raw, budget_steps)


def _write_pair(
    store: Path,
    loss_interp: xr.Dataset,
    perf: xr.Dataset,
    ti: int,
    sj: int,
) -> None:
    """Region-write one ``(task, optimizer)`` cell into the store.

    ``ti`` / ``sj`` are the integer positions of the task / schedule in
    the store's coordinates. Each variable is transposed to canonical
    ``(taskID, schedule, ...)`` order, cast to float and stripped of
    coordinates before the region write (see
    :func:`l2co._src.databank.schema.to_region_cell`).
    """
    for ds in (loss_interp, perf):
        cell = to_region_cell(ds)
        region: dict[str, slice] = {
            TASK_ID: slice(ti, ti + 1),
            SCHEDULE: slice(sj, sj + 1),
        }
        for d in cell.dims:
            if d not in (TASK_ID, SCHEDULE):
                region[d] = slice(0, cell.sizes[d])
        with silence_zarr_warnings():
            cell.to_zarr(store, region=region, consolidated=False)


def _finalize_small(
    store: Path,
    labels: dict[str, xr.DataArray],
    targets: dict[str, xr.DataArray],
    recipes: dict[str, xr.DataArray],
    taskids: list[str],
    schedules: list[str],
) -> None:
    """Write the small string / target arrays and consolidate metadata.

    ``task_labels`` (taskID, feature), ``target_value`` (taskID, target)
    and ``optimizationstep`` (schedule,) are tiny, so they are assembled
    in full and written with ``mode="a"`` (which also creates the new
    ``feature`` coordinate). Metadata is consolidated last so the reader
    opens the store with a single metadata read.

    The two string variables are stored as **variable-length** (``object``
    dtype) strings. Fixed-width unicode would cap every entry at the first
    contributor's width and silently truncate longer ones -- e.g. a longer
    ``optimizationstep`` JSON recipe would be cut mid-string and fail to
    parse on read. Each contributor is cast to ``object`` *before* the
    concat so the full string is preserved.
    """
    labels_da = xr.concat(
        [labels[t].astype(object) for t in taskids], dim=TASK_ID
    )
    recipes_da = xr.concat(
        [recipes[s].astype(object) for s in schedules], dim=SCHEDULE
    )
    small = xr.Dataset(
        {
            TASK_LABELS: labels_da,
            TARGET_VALUE: xr.concat(
                [targets[t] for t in taskids], dim=TASK_ID
            ),
            OPTIMIZATIONSTEP: recipes_da,
        }
    )
    with silence_zarr_warnings():
        small.to_zarr(store, mode="a", consolidated=False)
        zarr.consolidate_metadata(store)


def _accumulate_task(
    task_ds: xr.Dataset,
    task_hash: str,
    labels: dict[str, xr.DataArray],
    targets: dict[str, xr.DataArray],
) -> None:
    """Record one task's labels / target_value, asserting feature order.

    The consolidated ``task_labels`` is a single ``(taskID, feature)``
    array, so every task must expose the same ``feature`` ordering.
    """
    feats = list(task_ds[FEATURE].values)
    if labels:
        first = next(iter(labels.values()))
        if list(first[FEATURE].values) != feats:
            raise ValueError(
                "Inconsistent task 'feature' axis across tasks: "
                f"{feats} != {list(first[FEATURE].values)}. The "
                "consolidated store requires one shared feature ordering."
            )
    labels[task_hash] = task_ds[TASK_LABELS]
    targets[task_hash] = task_ds[TARGET_VALUE]


def dump_databank(
    experiment_data,
    tasks_experiment_data,
    schedules_experiment_data,
    root: Path,
    dataset_name: str,
    *,
    mode: Literal["error", "replace", "append"] = "error",
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
    taskset: str | None = None,
    budget_steps: int = layout.BUDGET_STEPS,
) -> Path:
    """Write finished optimization runs to a consolidated-zarr databank.

    Stores the interpolated ``loss_history`` plus ``performance`` /
    ``task`` / ``optimizer`` in one zarr store (one chunk per
    ``(task, optimizer)``) and keeps the raw loss histories as per-pair
    netCDF under ``loss_history_raw/``.

    Parameters
    ----------
    experiment_data : f3dasm.ExperimentData
        Finished optimization runs (one row per ``(task, optimizer)``).
    tasks_experiment_data : f3dasm.ExperimentData
        Source whose ``task`` output column is indexed by
        ``experiment_data["task_idx"]``.
    schedules_experiment_data : f3dasm.ExperimentData
        Source whose ``optimizer`` input column is indexed by
        ``experiment_data["optimizer_idx"]``.
    root : pathlib.Path
        The databank base directory (the application supplies it).
    dataset_name : str
        Name of the dataset.
    mode : {"error", "replace", "append"}, optional
        How to handle an existing store. ``"error"`` (default) raises,
        ``"replace"`` wipes and rewrites, ``"append"`` extends the store
        with new tasks / optimizers (existing tasks keep their stored
        ERT ``target_value``). Append reuses the existing store's
        ``budget`` grid, ignoring ``budget_steps``.
    easiest_quantile : float, optional
        Forwarded to the per-task ERT target computation.
    taskset : str | None, optional
        If given, also write ``tasksets/<taskset>.txt`` (under the
        per-dataset folder) listing every task hash written.
    budget_steps : int, optional
        Number of points on the fixed budget grid the zarr
        ``loss_history`` is interpolated onto, by default
        :data:`l2co._src.databank.layout.BUDGET_STEPS`.

    Returns
    -------
    pathlib.Path
        Absolute path to the written zarr store.

    Raises
    ------
    FileExistsError
        The store exists and ``mode="error"``.
    ValueError
        No runs are present, or a task's ``feature`` axis is
        inconsistent with the others.
    """
    store = layout.zarr_store_path(root, dataset_name)
    rawroot = layout.raw_root(root, dataset_name)
    exists = store.exists()

    if exists and mode == "error":
        raise FileExistsError(
            f"Zarr store {store} already exists; pass mode='replace' to "
            "wipe and rewrite, or mode='append' to extend it."
        )
    if mode == "replace":
        if store.exists():
            shutil.rmtree(store)
        if rawroot.exists():
            shutil.rmtree(rawroot)
        exists = False

    runs = [
        (task, step_paths)
        for task, step_paths in _iter_task_runs(
            experiment_data,
            tasks_experiment_data,
            schedules_experiment_data,
        )
        if step_paths
    ]
    if not runs:
        raise ValueError("No finished (task, optimizer) runs to write.")

    rawroot.mkdir(parents=True, exist_ok=True)

    if mode == "append" and exists:
        result = _dump_append(store, rawroot, runs, easiest_quantile)
    else:
        result = _dump_create(
            store, rawroot, runs, easiest_quantile, budget_steps
        )

    if taskset is not None:
        save_taskset(
            layout.databank_root(root, dataset_name),
            taskset,
            [task.hash for task, _ in runs],
        )

    logger.info(
        f"dump_databank: done -- {result} task(s) -> {store} (mode={mode})"
    )
    return store.resolve()


def _records_for_task(
    task: Task,
    step_paths: dict[OptimizationStep, Path],
    easiest_quantile: float,
    existing_targets,
    budget_steps: int,
    rawroot: Path,
) -> tuple[xr.Dataset, dict[int, OptimizationStep], dict, dict, dict]:
    """Build one task's records and write its raw loss histories.

    Returns ``(task_ds, hash_to_step, loss_interp, perf, opt)`` where the
    three dicts are keyed by ``OptimizationStep.hash``. The raw loss
    histories are written to ``loss_history_raw/`` as a side effect, with
    their float variables downcast to :data:`~l2co._src.databank.schema.
    STORE_DTYPE` (integer counters keep their dtype; see
    :func:`l2co._src.databank.schema.downcast_floats`).
    """
    ds_by_step = {
        step: xr.load_dataset(path) for step, path in step_paths.items()
    }
    hash_to_step = {step.hash: step for step in step_paths}

    task_ds, loss_raw, perf, opt = _build_task_records(
        task, ds_by_step, easiest_quantile, existing_targets=existing_targets
    )

    loss_interp: dict[int, xr.Dataset] = {}
    for h, raw in loss_raw.items():
        downcast_floats(raw).to_netcdf(
            layout.loss_raw_path(rawroot, task.hash, h), engine="netcdf4"
        )
        loss_interp[h] = _interp_loss(raw, budget_steps)
    return task_ds, hash_to_step, loss_interp, perf, opt


def _dump_create(
    store: Path,
    rawroot: Path,
    runs: list,
    easiest_quantile: float,
    budget_steps: int,
) -> int:
    """Stream a fresh store, one ``(task, optimizer)`` cell at a time."""
    taskids = [task.hash for task, _ in runs]
    schedules: list[str] = []
    for _, step_paths in runs:
        for step in step_paths:
            if step.name not in schedules:
                schedules.append(step.name)

    labels: dict[str, xr.DataArray] = {}
    targets: dict[str, xr.DataArray] = {}
    recipes: dict[str, xr.DataArray] = {}
    created = False

    for task, step_paths in tqdm(runs, total=len(runs)):
        task_ds, hash_to_step, loss_interp, perf, opt = _records_for_task(
            task, step_paths, easiest_quantile, None, budget_steps, rawroot
        )

        if not created:
            h0 = next(iter(loss_interp))
            template, encoding = big_template(
                loss_interp[h0], perf[h0], taskids, schedules
            )
            with silence_zarr_warnings():
                template.to_zarr(
                    store,
                    mode="w",
                    compute=False,
                    encoding=encoding,
                    zarr_format=3,
                    consolidated=False,
                )
            created = True

        ti = taskids.index(task.hash)
        for h, step in hash_to_step.items():
            sj = schedules.index(step.name)
            _write_pair(store, loss_interp[h], perf[h], ti, sj)
            recipes.setdefault(step.name, opt[h][OPTIMIZATIONSTEP])
        _accumulate_task(task_ds, task.hash, labels, targets)

    _finalize_small(store, labels, targets, recipes, taskids, schedules)
    return len(taskids)


def _populated_mask(old: xr.Dataset) -> xr.DataArray:
    """Return a ``(taskID, schedule)`` bool mask of written cells.

    A cell counts as populated when its ``success`` is non-null for any
    interior index -- i.e. the ``(task, optimizer)`` pair was written.
    """
    mask = old[SUCCESS].notnull()
    for d in list(mask.dims):
        if d not in (TASK_ID, SCHEDULE):
            mask = mask.any(d)
    return mask.load()


def _dump_append(
    store: Path,
    rawroot: Path,
    runs: list,
    easiest_quantile: float,
) -> int:
    """Rebuild the store over the union of old and new coordinates.

    New cells come from ``runs`` and are written first; old cells are then
    copied from the old store one at a time. A ``(task, optimizer)`` pair
    present in **both** is *rewritten* from ``runs`` (the copy step skips
    it), so re-running a pair -- e.g. at a higher iteration budget --
    overwrites its stored cell instead of being clobbered by the stale
    copy. Existing tasks keep their stored ERT targets so appended (and
    re-run) optimizers stay comparable. The rebuild lands in a temporary
    store that is swapped in atomically.
    """
    with silence_zarr_warnings():
        old = xr.open_zarr(store, consolidated=True)
    budget_steps = int(old.sizes["budget"])
    old_taskids = [str(t) for t in old[TASK_ID].values]
    old_scheds = [str(s) for s in old[SCHEDULE].values]
    old_targets = old[TARGET_VALUE].load()
    old_labels = old[TASK_LABELS].load()
    old_recipes = old[OPTIMIZATIONSTEP].load()
    mask = _populated_mask(old)
    loss_vars = [v for v in old.data_vars if "budget" in old[v].dims]
    perf_vars = [
        v
        for v in old.data_vars
        if "target" in old[v].dims and v != TARGET_VALUE
    ]

    # Union coordinates: old first (stable), then the new arrivals.
    taskids = list(old_taskids)
    schedules = list(old_scheds)
    for task, step_paths in runs:
        if task.hash not in taskids:
            taskids.append(task.hash)
        for step in step_paths:
            if step.name not in schedules:
                schedules.append(step.name)

    tmp = store.with_name(store.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)

    labels: dict[str, xr.DataArray] = {}
    targets: dict[str, xr.DataArray] = {}
    recipes: dict[str, xr.DataArray] = {}
    # (taskID, schedule) cells written from runs -- the copy step below
    # must skip these so a re-run pair overwrites rather than gets clobbered.
    written: set[tuple[str, str]] = set()

    # New runs first: gives the template sample and (re)writes raw files.
    for task, step_paths in tqdm(runs, total=len(runs)):
        existing = (
            old_targets.sel({TASK_ID: task.hash}).values
            if task.hash in old_taskids
            else None
        )
        task_ds, hash_to_step, loss_interp, perf, opt = _records_for_task(
            task,
            step_paths,
            easiest_quantile,
            existing,
            budget_steps,
            rawroot,
        )
        if not tmp.exists():
            h0 = next(iter(loss_interp))
            template, encoding = big_template(
                loss_interp[h0], perf[h0], taskids, schedules
            )
            with silence_zarr_warnings():
                template.to_zarr(
                    tmp,
                    mode="w",
                    compute=False,
                    encoding=encoding,
                    zarr_format=3,
                    consolidated=False,
                )
        ti = taskids.index(task.hash)
        for h, step in hash_to_step.items():
            sj = schedules.index(step.name)
            _write_pair(tmp, loss_interp[h], perf[h], ti, sj)
            recipes.setdefault(step.name, opt[h][OPTIMIZATIONSTEP])
            written.add((task.hash, step.name))
        _accumulate_task(task_ds, task.hash, labels, targets)

    # Copy untouched old cells (one at a time) into the rebuilt store.
    for oi, t in enumerate(old_taskids):
        ti = taskids.index(t)
        for oj, s in enumerate(old_scheds):
            if not bool(mask.isel({TASK_ID: oi, SCHEDULE: oj})):
                continue
            if (t, s) in written:
                # A new run already (re)wrote this cell -- don't clobber it.
                continue
            sj = schedules.index(s)
            with silence_zarr_warnings():
                loss_cell = (
                    old[loss_vars].isel({TASK_ID: [oi], SCHEDULE: [oj]}).load()
                )
                perf_cell = (
                    old[perf_vars].isel({TASK_ID: [oi], SCHEDULE: [oj]}).load()
                )
            _write_pair(tmp, loss_cell, perf_cell, ti, sj)

    # Small arrays: old entries fall back to the stored values.
    for t in taskids:
        if t not in labels:
            labels[t] = old_labels.sel({TASK_ID: [t]})
            targets[t] = old_targets.sel({TASK_ID: [t]})
    for s in schedules:
        if s not in recipes:
            recipes[s] = old_recipes.sel({SCHEDULE: [s]})

    old.close()
    _finalize_small(tmp, labels, targets, recipes, taskids, schedules)

    shutil.rmtree(store)
    tmp.rename(store)
    return len(taskids)
