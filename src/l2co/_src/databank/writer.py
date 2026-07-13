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

import json
import logging
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

# Third-party
import numpy as np
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
    IN_TRAINING,
    OPTIMIZATIONSTEP,
    PROVENANCE,
    SCHEDULE,
    SUCCESS,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
)

from . import layout
from .provenance import (
    _weights_files,
    manifest_path,
    parse_provenance,
    policy_identity_hash,
    record_task_hashes,
    record_weights_hash,
    weights_hash,
)
from .records import _build_task_records, _iter_task_runs
from .schema import STORE_DTYPE, big_template, downcast_floats, to_region_cell
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


def _in_training_mask(
    provenances: dict[str, xr.DataArray],
    taskids: list[str],
    schedules: list[str],
) -> xr.DataArray:
    """Materialize the ``(taskID, schedule)`` in-training mask.

    Derived from two inputs frozen at dump time — each policy
    schedule's recorded training task hashes and the store's task
    axis — so the mask cannot drift from the provenance it summarizes.
    Policy schedules get ``1.0`` / ``0.0`` per task; static optimizers
    (empty provenance) get ``NaN``.
    """
    mask = np.full((len(taskids), len(schedules)), np.nan, dtype=STORE_DTYPE)
    for j, s in enumerate(schedules):
        record = parse_provenance(str(provenances[s].values.flat[0]))
        if record is None:
            continue
        trained = record_task_hashes(record)
        mask[:, j] = [float(t in trained) for t in taskids]
    return xr.DataArray(
        mask,
        coords={TASK_ID: taskids, SCHEDULE: schedules},
        dims=[TASK_ID, SCHEDULE],
    )


def _check_alias_consistency(name: str, existing: str, incoming: str) -> None:
    """Refuse to mix two different policy artifacts under one alias.

    Schedule names are short human aliases; identity lives in the
    provenance record. Reusing a name for a *different* artifact (or
    for a static optimizer where a policy lived, and vice versa) would
    silently mix rollouts from different weights under one coordinate,
    so it is a hard error. Records of the *same* artifact may differ in
    their evaluation section (timestamps); only the combined weights
    hash decides.
    """
    old = parse_provenance(existing)
    new = parse_provenance(incoming)
    if old is None and new is None:
        return
    if (old is None) != (new is None):
        raise ValueError(
            f"Schedule name '{name}' is already used by a "
            f"{'static optimizer' if old is None else 'policy artifact'} "
            f"but the incoming run is a "
            f"{'static optimizer' if new is None else 'policy artifact'}: "
            "pick a different schedule name."
        )
    old_hash = record_weights_hash(old)
    new_hash = record_weights_hash(new)
    if old_hash != new_hash:
        raise ValueError(
            f"Schedule name '{name}' already refers to policy artifact "
            f"{old_hash[:12]}… but the incoming run evaluated "
            f"{new_hash[:12]}…: one alias must always mean one artifact "
            "— pick a different schedule name for the new checkpoint."
        )


def _finalize_small(
    store: Path,
    labels: dict[str, xr.DataArray],
    targets: dict[str, xr.DataArray],
    recipes: dict[str, xr.DataArray],
    provenances: dict[str, xr.DataArray],
    taskids: list[str],
    schedules: list[str],
) -> None:
    """Write the small string / target arrays and consolidate metadata.

    ``task_labels`` (taskID, feature), ``target_value`` (taskID, target),
    ``optimizationstep`` / ``provenance`` (schedule,) and the derived
    ``in_training`` (taskID, schedule) mask are tiny, so they are
    assembled in full and written with ``mode="a"`` (which also creates
    the new ``feature`` coordinate). Metadata is consolidated last so
    the reader opens the store with a single metadata read.

    The string variables are stored as **variable-length** (``object``
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
    provenance_da = xr.concat(
        [provenances[s].astype(object) for s in schedules], dim=SCHEDULE
    )
    small = xr.Dataset(
        {
            TASK_LABELS: labels_da,
            TARGET_VALUE: xr.concat(
                [targets[t] for t in taskids], dim=TASK_ID
            ),
            OPTIMIZATIONSTEP: recipes_da,
            PROVENANCE: provenance_da,
            IN_TRAINING: _in_training_mask(provenances, taskids, schedules),
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

    Learned optimizers (rl2co / l2co policy schedules) are stored in the
    exact same array format as static optimizers, plus three additive
    provenance surfaces (see :mod:`l2co._src.databank.provenance`): the
    per-schedule ``provenance`` record (read from the loss history's
    ``provenance`` attribute; a policy schedule without one is refused),
    the derived ``in_training`` mask, and an archived copy of the
    artifact files under ``policy_artifacts/<schedule>/``.

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
        n_tasks, policy_provs = _dump_append(
            store, rawroot, runs, easiest_quantile
        )
    else:
        n_tasks, policy_provs = _dump_create(
            store, rawroot, runs, easiest_quantile, budget_steps
        )

    _archive_policy_artifacts(root, dataset_name, policy_provs)

    if taskset is not None:
        save_taskset(
            layout.databank_root(root, dataset_name),
            taskset,
            [task.hash for task, _ in runs],
        )

    logger.info(
        f"dump_databank: done -- {n_tasks} task(s) -> {store} (mode={mode})"
    )
    return store.resolve()


def _safe_dirname(schedule_name: str) -> str:
    """Filesystem-safe folder name for one schedule."""
    return schedule_name.replace("/", "_")


def _archive_policy_artifacts(
    root: Path,
    dataset_name: str,
    provenances: dict[str, str],
) -> None:
    """Copy each policy schedule's artifact files into the dataset.

    Training run directories live on purged scratch, so the dataset
    archives every policy artifact it stores results for:
    ``policy_artifacts/<schedule>/`` receives the checkpoint file(s)
    listed under the record's ``evaluation.artifacts`` map, their
    sibling manifests, and a ``provenance.json`` with the full record.
    This keeps the dataset directory self-sufficient — the rollouts,
    their provenance, *and* the means to reproduce or extend them.

    Fails loudly (no partial silence): a policy record without an
    ``artifacts`` map, a missing artifact file, or a file whose bytes
    no longer match the expected ``weights_hash`` each raise.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.
    provenances : dict[str, str]
        Serialized provenance record per policy schedule written in
        this dump (static schedules are not present).
    """
    for schedule_name, prov_str in provenances.items():
        record = parse_provenance(prov_str)
        if record is None:  # pragma: no cover - callers pre-filter
            continue
        artifacts = record.get("evaluation", {}).get("artifacts", {})
        if not artifacts:
            raise ValueError(
                f"Policy schedule '{schedule_name}' carries no "
                "'evaluation.artifacts' map in its provenance record, so "
                "its checkpoint cannot be archived into the dataset. "
                "List every artifact file (path + expected weights_hash) "
                "when assembling the record."
            )
        dest = layout.policy_artifacts_root(
            root, dataset_name
        ) / _safe_dirname(schedule_name)
        dest.mkdir(parents=True, exist_ok=True)
        for label, spec in artifacts.items():
            src = Path(spec["path"])
            try:
                files = _weights_files(src)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Cannot archive artifact '{label}' of schedule "
                    f"'{schedule_name}': {e} Re-evaluate the policy from "
                    "an existing checkpoint (the run directory may have "
                    "been purged between evaluation and dump)."
                ) from e
            expected = spec.get("weights_hash")
            if expected is not None and weights_hash(src) != expected:
                raise ValueError(
                    f"Artifact '{label}' of schedule '{schedule_name}' at "
                    f"{src} no longer matches its recorded weights_hash: "
                    "the file was overwritten since evaluation; refusing "
                    "to archive the wrong weights."
                )
            for f in files:
                shutil.copy2(f, dest / f.name)
            mpath = manifest_path(src)
            if mpath.is_file():
                shutil.copy2(mpath, dest / mpath.name)
        (dest / "provenance.json").write_text(
            json.dumps(record, indent=2, default=str)
        )
        logger.info(
            f"dump_databank: archived policy artifact(s) of "
            f"'{schedule_name}' -> {dest}"
        )


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

    Raw filenames are keyed by ``OptimizationStep.hash`` for static
    optimizers and by :func:`~l2co._src.databank.provenance.
    policy_identity_hash` for policy schedules (the recipe hash is
    path-derived for those — neither stable nor unique across
    checkpoints).
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
        record = parse_provenance(str(raw.attrs.get(PROVENANCE, "")))
        file_hash = (
            policy_identity_hash(record, hash_to_step[h])
            if record is not None
            else h
        )
        downcast_floats(raw).to_netcdf(
            layout.loss_raw_path(rawroot, task.hash, file_hash),
            engine="netcdf4",
        )
        loss_interp[h] = _interp_loss(raw, budget_steps)
    return task_ds, hash_to_step, loss_interp, perf, opt


def _accumulate_schedule(
    step_name: str,
    opt_ds: xr.Dataset,
    recipes: dict[str, xr.DataArray],
    provenances: dict[str, xr.DataArray],
) -> None:
    """Record one schedule's recipe + provenance, guarding the alias.

    First contributor wins (matching the recipe behaviour); later
    contributions of the same name must refer to the same policy
    artifact (see :func:`_check_alias_consistency`).
    """
    incoming = str(opt_ds[PROVENANCE].values.flat[0])
    if step_name in provenances:
        _check_alias_consistency(
            step_name,
            str(provenances[step_name].values.flat[0]),
            incoming,
        )
        return
    recipes[step_name] = opt_ds[OPTIMIZATIONSTEP]
    provenances[step_name] = opt_ds[PROVENANCE]


def _dump_create(
    store: Path,
    rawroot: Path,
    runs: list,
    easiest_quantile: float,
    budget_steps: int,
) -> tuple[int, dict[str, str]]:
    """Stream a fresh store, one ``(task, optimizer)`` cell at a time.

    Returns ``(n_tasks, provenance_by_schedule)`` where the dict holds
    the serialized provenance record of every *policy* schedule written
    (used by :func:`dump_databank` to archive the artifacts).
    """
    taskids = [task.hash for task, _ in runs]
    schedules: list[str] = []
    for _, step_paths in runs:
        for step in step_paths:
            if step.name not in schedules:
                schedules.append(step.name)

    labels: dict[str, xr.DataArray] = {}
    targets: dict[str, xr.DataArray] = {}
    recipes: dict[str, xr.DataArray] = {}
    provenances: dict[str, xr.DataArray] = {}
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
            _accumulate_schedule(step.name, opt[h], recipes, provenances)
        _accumulate_task(task_ds, task.hash, labels, targets)

    _finalize_small(
        store, labels, targets, recipes, provenances, taskids, schedules
    )
    return len(taskids), _policy_provenances(provenances)


def _policy_provenances(
    provenances: dict[str, xr.DataArray],
) -> dict[str, str]:
    """Extract the non-empty (policy) provenance strings per schedule."""
    out: dict[str, str] = {}
    for name, da in provenances.items():
        s = str(da.values.flat[0])
        if s:
            out[name] = s
    return out


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
) -> tuple[int, dict[str, str]]:
    """Rebuild the store over the union of old and new coordinates.

    New cells come from ``runs`` and are written first; old cells are then
    copied from the old store one at a time. A ``(task, optimizer)`` pair
    present in **both** is *rewritten* from ``runs`` (the copy step skips
    it), so re-running a pair -- e.g. at a higher iteration budget --
    overwrites its stored cell instead of being clobbered by the stale
    copy. Existing tasks keep their stored ERT targets so appended (and
    re-run) optimizers stay comparable. The rebuild lands in a temporary
    store that is swapped in atomically.

    Returns ``(n_tasks, provenance_by_schedule)`` for the policy
    schedules written from ``runs`` (see :func:`_dump_create`).
    Schedule names arriving for an already-present name must refer to
    the same policy artifact (:func:`_check_alias_consistency`); stores
    written before the provenance schema read back as all-static
    (empty provenance) and the derived ``in_training`` mask is
    recomputed over the union.
    """
    with silence_zarr_warnings():
        old = xr.open_zarr(store, consolidated=True)
    budget_steps = int(old.sizes["budget"])
    old_taskids = [str(t) for t in old[TASK_ID].values]
    old_scheds = [str(s) for s in old[SCHEDULE].values]
    old_targets = old[TARGET_VALUE].load()
    old_labels = old[TASK_LABELS].load()
    old_recipes = old[OPTIMIZATIONSTEP].load()
    if PROVENANCE in old:
        old_provenances = old[PROVENANCE].astype(object).load()
    else:
        # Pre-provenance store: every existing schedule is static.
        old_provenances = xr.DataArray(
            np.array([""] * len(old_scheds), dtype=object),
            coords={SCHEDULE: old_scheds},
            dims=[SCHEDULE],
        )
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
    provenances: dict[str, xr.DataArray] = {}
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
            if step.name in old_scheds:
                # An alias may only ever mean one artifact -- also
                # across appends into an existing store.
                _check_alias_consistency(
                    step.name,
                    str(old_provenances.sel({SCHEDULE: step.name}).values),
                    str(opt[h][PROVENANCE].values.flat[0]),
                )
            _accumulate_schedule(step.name, opt[h], recipes, provenances)
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
            provenances[s] = old_provenances.sel({SCHEDULE: [s]})

    old.close()
    new_policy_provs = _policy_provenances(
        {s: provenances[s] for s in {name for _, name in written}}
    )
    _finalize_small(
        tmp, labels, targets, recipes, provenances, taskids, schedules
    )

    shutil.rmtree(store)
    tmp.rename(store)
    return len(taskids), new_policy_provs
