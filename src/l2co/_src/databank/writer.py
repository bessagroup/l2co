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

Every mode is driven through the same three phases (:func:`dump_databank`
is a thin serial wrapper over them; see the "Parallel dump" section
below): a **template** step lays out the store, a **task** step
region-writes each ``(task, optimizer)`` cell, and a **finalize** step
writes the small string / target arrays and consolidates.

Creation (``mode="error"`` / ``"replace"``) builds a fresh **deferred**
zarr template (``compute=False``) over the full ``(taskID, schedule)``
grid, so the full dense array is never held in memory -- only the cells
the task step actually writes touch disk.

``mode="append"`` **grows the existing store in place** rather than
rebuilding it: :meth:`zarr.Array.resize` on these sharded arrays is
metadata-only, so every untouched old cell keeps its file key and its
bytes and a new ``(taskID, schedule)`` position simply reads back as the
fill value. The ``taskID`` / ``schedule`` coordinate arrays are the one
exception -- their fixed-length string dtype would truncate a longer new
entry -- so they are deleted and rewritten fresh at the union values
(see :func:`_grow_store_for_append`). The upshot: append cost scales
with the *incoming* runs, not with however many cells the store already
holds.

Growing in place means a crashed append can leave the store
mid-mutation, so :func:`_template_append` writes an
:data:`l2co._src.databank.layout.APPEND_MARKER_NAME` sidecar *inside*
the store before touching anything (the reader's :func:`_open_store`
choke point in :mod:`l2co._src.databank.reader` refuses to load a
marked store) and stages an :func:`_snapshot_for_append` of every path
the grow is about to mutate -- root/array metadata, the two coordinate
arrays, the five small arrays, and the chunk file of any *incoming* pair
that already has a stored cell (a re-run pair overwrites it). The
snapshot is hard-linked, so it costs disk proportional to the incoming
runs, not to the store. :func:`dump_databank_finalize` removes both the
marker and the snapshot on success; a leftover snapshot found by the
*next* append is auto-restored (:func:`_restore_rollback`) before that
append proceeds.

All cell writes go through zarr array handles opened once per dump
(:func:`_open_big_arrays`) rather than per-cell ``Dataset.to_zarr``
calls, which would re-read the store metadata on every call.

A store holds the noise-free series ``output_min_true`` (ADR 0024)
exactly when one of its tasks is noisy (has a ``loss_fn_true``): the
template takes its sample cell from the first such task
(:func:`_template_sample_run`), so the variable's presence follows
every task, not the first. A noise-free task's cells of that variable
are never written: no chunk file, read back as NaN, and scored on the
observed series. Append adds noise-free runs to such a store freely,
but refuses noisy runs into a store without the variable.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# Third-party
import numpy as np
import xarray as xr
import zarr

# Local
from l2co_optimizers import OptimizationStep
from l2co_tasks import Task

from l2co._src.databank import layout
from l2co._src.databank.provenance import (
    _weights_files,
    manifest_path,
    parse_provenance,
    policy_identity_hash,
    record_task_hashes,
    record_weights_hash,
    weights_hash,
)
from l2co._src.databank.records import (
    BREACHES_ATTR,
    _build_task_records,
    _iter_task_runs,
)
from l2co._src.databank.schema import (
    STORE_DTYPE,
    big_template,
    canonical_dims,
    downcast_floats,
    to_region_cell,
)
from l2co._src.databank.tasksets import save_taskset
from l2co._src.performance import DEFAULT_EASIEST_QUANTILE
from l2co._src.validation import normalize_budget_grid
from l2co.naming import (
    FEATURE,
    GLOBAL_MIN,
    IN_TRAINING,
    OPTIMIZATIONSTEP,
    OUTPUT_MIN_TRUE,
    PROVENANCE,
    SCHEDULE,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# task_labels fill-in for a feature column a given task's tag doesn't carry
# (e.g. bbob_embedded's "intrinsic_dimensionality" on a plain bbob task in
# the same dump). Chosen to parse cleanly via float("nan") -- the same
# "present and finite" convention l2co._src.data.report_helpers already
# uses for an absent-per-task global_min -- rather than a marker only a
# human reader would recognize.
MISSING_FEATURE = "nan"


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


def _open_big_arrays(
    store: Path, names: Iterable[str]
) -> dict[str, zarr.Array]:
    """Open the store's big per-pair arrays once for repeated cell writes.

    A per-cell ``Dataset.to_zarr(region=...)`` call re-opens the group
    and re-reads every array's metadata, which dominates the write cost
    of the small per-pair cells; the returned handles amortize that to
    a single open per dump.

    Opens with ``use_consolidated=False``: an append's template step
    just grew these arrays via :meth:`zarr.Array.resize`, which leaves
    *consolidated* metadata stale (still reporting the pre-grow shape)
    until the next explicit ``consolidate_metadata`` -- reading the
    live per-array metadata directly sidesteps that regardless of
    whether the template step's re-consolidate has landed on disk yet.
    """
    with silence_zarr_warnings():
        group = zarr.open_group(store, mode="r+", use_consolidated=False)
        return {name: group[name] for name in names}


def _write_pair(
    arrays: dict[str, zarr.Array],
    ds: xr.Dataset,
    ti: int,
    sj: int,
) -> None:
    """Region-write one ``(task, optimizer)`` cell through open handles.

    ``ti`` / ``sj`` are the integer positions of the task / schedule in
    the store's coordinates; ``arrays`` holds the store's arrays keyed
    by variable name (see :func:`_open_big_arrays`). Each variable is
    transposed to canonical ``(taskID, schedule, ...)`` order, cast to
    float and stripped of coordinates before the write (see
    :func:`l2co._src.databank.schema.to_region_cell`); an interior dim
    smaller than the store's leaves the remainder at the NaN fill.
    """
    cell = to_region_cell(ds)
    for name, var in cell.data_vars.items():
        index = (
            slice(ti, ti + 1),
            slice(sj, sj + 1),
            *(slice(0, size) for size in var.shape[2:]),
        )
        arrays[name][index] = var.values


def _cell_file(
    store: Path, name: str, arr: zarr.Array, i: int, j: int
) -> Path:
    """Path of the single chunk file holding cell ``(i, j)`` of ``arr``."""
    key = arr.metadata.chunk_key_encoding.encode_chunk_key(
        (i, j, *([0] * (arr.ndim - 2)))
    )
    return store / name / key


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``, copying when linking is unsupported.

    Used both to snapshot a store path before an append mutates it
    (:func:`_snapshot_for_append`) and to restore one after (source and
    destination swap roles, via :func:`_restore_rollback`'s ``os.replace``
    instead). The snapshot sits next to the store (same filesystem), so a
    link is the common case; the copy fallback covers filesystems without
    hard links. Linking a chunk file for a snapshot is safe because zarr's
    local store writes via a temp file + rename -- overwriting a cell
    allocates a fresh inode rather than mutating the linked one in place.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


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

    Per-task ``feature`` columns need not match: ``task_labels`` is
    concatenated with an explicit outer join, so a task missing a
    feature some other task in the same dump carries (mixed-family
    composite task sets -- see :func:`_accumulate_task`) gets
    :data:`MISSING_FEATURE` in that column rather than aborting the
    dump. ``join`` is passed explicitly because xarray's own default
    for ``concat`` is changing from ``"outer"`` to ``"exact"``.

    An appended dump can also widen the ``feature`` axis itself (a new
    task family contributing a longer feature name than the store has
    ever stored, e.g. ``intrinsic_dimensionality``). ``feature`` is a
    coordinate, so it is cast to ``object`` too, but unlike
    ``task_labels``' values, xarray still checks an *existing*
    on-disk coordinate's fixed-width dtype against the incoming one
    on ``to_zarr(mode="a")`` and refuses a mismatch outright -- there
    is no outer-join escape hatch for a coordinate the way there is
    for the values. ``task_labels`` and ``feature`` are therefore
    dropped from the store first when present, the same "delete and
    rewrite fresh at the union values" treatment
    :func:`_grow_store_for_append` already gives ``taskID`` /
    ``schedule`` for the identical reason; the rewrite lands the full
    union either way, so this is a no-op for a variable that was
    already going to be fully overwritten and simply lets a first-ever
    dump skip it (nothing to drop yet).
    """
    labels_da = xr.concat(
        [labels[t].astype(object) for t in taskids],
        dim=TASK_ID,
        join="outer",
        fill_value=MISSING_FEATURE,
    )
    labels_da = labels_da.assign_coords(
        {FEATURE: labels_da[FEATURE].astype(object)}
    )
    with silence_zarr_warnings():
        for name in (TASK_LABELS, FEATURE):
            path = store / name
            if path.exists():
                shutil.rmtree(path)
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
    """Record one task's labels / target_value.

    Tasks are not required to share a ``feature`` ordering, or even the
    same *set* of features: a composite task suite that row-concatenates
    multiple task families in one dump (e.g. ``bbob_headroom_train``,
    which mixes plain ``bbob`` with ``bbob_embedded`` and
    ``bbob_noisy`` tasks -- see the ``parts:`` config in
    ``l2co_tasks``) legitimately produces per-task tags of different
    widths, since only ``bbob_embedded`` tags carry ``embedded`` /
    ``intrinsic_dimensionality`` / ``bulk_scale``
    (``l2co_tasks.create_embedded_bbob_task``). :func:`_finalize_small`
    reconciles the per-task rows collected here into one ``(taskID,
    feature)`` array by unioning every feature column and filling the
    ones a task doesn't have with :data:`MISSING_FEATURE`.
    """
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

    A thin serial driver over the same three phases the SLURM-array
    parallel dump uses (:func:`dump_databank_template` ->
    :func:`dump_databank_task` -> :func:`dump_databank_finalize`; see the
    module docstring), run here in one process. ``mode="append"`` grows
    the existing store in place, so its cost scales with the incoming
    runs rather than with the store's existing size.

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
        ``budget`` grid, ignoring ``budget_steps``, and requires the
        incoming runs' variable set and interior shapes (e.g.
        ``realization`` count) to match the store's exactly.
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
        No runs are present, or (``mode="append"``) the incoming runs'
        schema does not match the existing store's.
    """
    runs = [
        (task, step_paths)
        for task, step_paths in _iter_task_runs(
            experiment_data,
            tasks_experiment_data,
            schedules_experiment_data,
        )
        if step_paths
    ]
    return dump_runs(
        runs,
        root,
        dataset_name,
        mode=mode,
        easiest_quantile=easiest_quantile,
        taskset=taskset,
        budget_steps=budget_steps,
    )


def dump_runs(
    runs: list,
    root: Path,
    dataset_name: str,
    *,
    mode: Literal["error", "replace", "append"] = "error",
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
    taskset: str | None = None,
    budget_steps: int = layout.BUDGET_STEPS,
) -> Path:
    """Run all three dump phases over already-resolved runs, serially.

    :func:`dump_databank` is this plus the f3dasm bookkeeping that turns
    three ``ExperimentData`` tables into the ``runs`` list. Everything
    below that seam works on runs alone, which is what the parallel
    SLURM-array driver dispatches on too -- so this is the entry point
    for a caller that already holds runs and does not want to
    reconstruct the ``ExperimentData`` triple to get at the writer.

    Parameters
    ----------
    runs : list
        ``(task, step_paths)`` pairs: a task and the netCDF path of each
        of its schedules' loss histories. Empty ``step_paths`` entries
        must already be filtered out.
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.
    mode : {"error", "replace", "append"}, optional
        As :func:`dump_databank`.
    easiest_quantile : float, optional
        Forwarded to the per-task ERT target computation.
    taskset : str | None, optional
        Optional ``tasksets/<name>.txt`` to write.
    budget_steps : int, optional
        Budget grid to interpolate onto; ignored for ``"append"``, which
        reuses the existing store's grid.

    Returns
    -------
    pathlib.Path
        Absolute path to the written zarr store.
    """
    plan = _dump_template_runs(
        runs,
        root,
        dataset_name,
        mode=mode,
        easiest_quantile=easiest_quantile,
        taskset=taskset,
        budget_steps=budget_steps,
    )
    task_smalls = _dump_task_runs(runs, plan)
    return dump_databank_finalize(task_smalls, plan)


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


def _minimal_sample_steps(
    step_paths: dict[OptimizationStep, Path],
) -> dict[OptimizationStep, Path]:
    """Smallest optimizer subset that yields a shape-correct template cell.

    The store template only needs *one* interpolated loss cell + one
    performance cell to learn the interior dims/coords (budget grid,
    realizations, ERT target/metric axes) -- and those interior shapes do
    not depend on how many optimizers a task ran. So instead of loading a
    task's full optimizer set (~50 x 60 MB, the template step's OOM
    source), take just the ``randomsearch`` ERT-calibration baseline
    (structurally always present -- see
    :func:`l2co_experiments.create_experimentdata`) plus one representative
    optimizer. ``_ert_from_quantile`` needs the randomsearch baseline to
    derive targets when none are supplied; the representative optimizer
    gives a non-baseline performance cell.
    """
    items = list(step_paths.items())
    rs = [(s, p) for s, p in items if s.optimizer == "randomsearch"]
    other = [(s, p) for s, p in items if s.optimizer != "randomsearch"]
    chosen = (rs[:1] + other[:1]) or items[:1]
    return dict(chosen)


def _template_sample(
    task: Task,
    step_paths: dict[OptimizationStep, Path],
    easiest_quantile: float,
    budget_steps: int,
    existing_targets=None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Cheaply derive one ``(loss_interp, perf)`` sample for the template.

    Loads only :func:`_minimal_sample_steps` of the task (1-2 files) and
    builds its records in memory -- no raw sidecar write, no full-optimizer
    concat -- so the template step's memory stays flat regardless of
    optimizers-per-task. The returned interior dims/coords are identical to
    what the full per-task build produces (interior shape is
    optimizer-count-independent), so :func:`big_template` lays out the same
    store.
    """
    subset = _minimal_sample_steps(step_paths)
    ds_by_step = {step: xr.load_dataset(path) for step, path in subset.items()}
    _, loss_raw, perf, _ = _build_task_records(
        task, ds_by_step, easiest_quantile, existing_targets=existing_targets
    )
    h = next(iter(loss_raw))
    return _interp_loss(loss_raw[h], budget_steps), perf[h]


def _template_sample_run(runs: list) -> tuple[Task, dict]:
    """The ``(task, step_paths)`` run the store template is sized from.

    The template's variables are those of its one sample cell
    (:func:`l2co._src.databank.schema.big_template`), so the sample must
    carry the noise-free series whenever any run does (ADR 0024): the
    first noisy task -- one with a ``loss_fn_true`` -- else the first
    run. Told from the task, so choosing opens no file. That a noisy
    task's runs carry the series and no other task's do is
    :func:`l2co._src.databank.records._check_noise_free_series`, which
    the sample's own records run first; so a store has the variable
    exactly when one of its tasks has a ``loss_fn_true``.
    """
    for task, step_paths in runs:
        if getattr(task, "loss_fn_true", None) is not None:
            return task, step_paths
    return runs[0]


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


#                          Parallel dump: template / per-task / finalize
# =============================================================================
# The three functions below split a dump into a shape that a SLURM-array
# pipeline can fan out -- a single-process *template* that lays out the
# store (creating it fresh, or growing an existing one in place for
# append), an embarrassingly parallel *per-task* step that region-writes
# each task's disjoint cells (one shard file per ``(task, schedule)`` --
# so separate processes never touch the same file), and a single-process
# *finalize* that writes the small string / target arrays and
# consolidates. :func:`dump_databank` (the serial entry point) is a thin
# wrapper running the same three phases in one process, so the two paths
# share one implementation.


@dataclass
class DumpPlan:
    """Serializable hand-off between the three parallel-dump phases.

    Every field is JSON-native so the template step can persist it (e.g.
    ``json.dumps(dataclasses.asdict(plan))``) for the array + finalize
    steps to read back (``DumpPlan(**json.loads(...))``).

    Attributes
    ----------
    store : str
        The store the per-task step region-writes into. Append grows
        this store *in place* (no separate rebuild target), so this
        always equals ``final_store``.
    final_store : str
        Where the store lands. Always equals ``store`` (kept as a
        separate field for compatibility with existing serialized
        plans / call sites, not because the two paths can differ).
    rawroot : str
        ``loss_history_raw/`` directory the per-task step writes sidecars
        to.
    taskids, schedules : list[str]
        Full, ordered store coordinates (the old ∪ new union for append).
        The per-task step maps a cell to ``(taskids.index(hash),
        schedules.index(name))``.
    run_taskids : list[str]
        Task hashes contributed by *this* dump's runs (== ``taskids`` for
        create; the new subset for append). Used for the ``taskset``.
    budget_steps : int
        Fixed budget grid the loss history is interpolated onto (for
        append this is the existing store's grid).
    mode : {"create", "append"}
        Which finalize behaviour to run.
    easiest_quantile : float
        Forwarded to the per-task ERT computation.
    var_names : list[str]
        The big per-pair array names to open for region writes.
    root, dataset_name : str
        Databank base directory and dataset name (for artifact archival /
        taskset / fallbacks).
    existing_targets : dict[str, list[float]]
        For append: stored ERT ``target_value`` per already-present task,
        so appended optimizers score against them (mirrors the serial
        ``existing_targets=`` path).
    taskset : str | None
        Optional ``tasksets/<name>.txt`` to write in finalize.
    old_schedules : list[str]
        For append: the schedule names present in the store *before*
        this dump grew it (empty for create). The store's own
        ``schedule`` coordinate is the union by the time finalize runs
        an in-place append, so this is what finalize's alias-consistency
        check must test "already present" against -- reading it off the
        (already-grown) store there would treat every brand-new schedule
        as pre-existing.
    """

    store: str
    final_store: str
    rawroot: str
    taskids: list[str]
    schedules: list[str]
    run_taskids: list[str]
    budget_steps: int
    mode: str
    easiest_quantile: float
    var_names: list[str]
    root: str
    dataset_name: str
    existing_targets: dict[str, list[float]] = field(default_factory=dict)
    taskset: str | None = None
    old_schedules: list[str] = field(default_factory=list)


def dump_databank_template(
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
) -> DumpPlan:
    """Phase A: lay out the union store; return a :class:`DumpPlan`.

    Single-process, cheap. For ``create`` it builds the deferred (NaN,
    ``compute=False``) zarr template over the full ``(taskID, schedule)``
    grid. For ``append`` it *grows the existing store in place* over the
    old ∪ new union (see :func:`_template_append`) -- metadata-only for
    every array, so the cost scales with the incoming runs rather than
    with the store's existing size. Neither path writes any per-task
    loss history here (the one task loaded to size a fresh template's
    sample cell uses ``write_raw=False``).

    Takes the same three source ExperimentData as :func:`dump_databank`.
    Raises the same ``FileExistsError`` / ``ValueError`` as
    :func:`dump_databank` for ``mode="error"`` on an existing store, for
    an empty run set, and (``mode="append"``) for a schema mismatch
    between the incoming runs and the store.
    """
    runs = [
        (task, step_paths)
        for task, step_paths in _iter_task_runs(
            experiment_data,
            tasks_experiment_data,
            schedules_experiment_data,
        )
        if step_paths
    ]
    return _dump_template_runs(
        runs,
        root,
        dataset_name,
        mode=mode,
        easiest_quantile=easiest_quantile,
        taskset=taskset,
        budget_steps=budget_steps,
    )


def _dump_template_runs(
    runs: list,
    root: Path,
    dataset_name: str,
    *,
    mode: Literal["error", "replace", "append"],
    easiest_quantile: float,
    taskset: str | None,
    budget_steps: int,
) -> DumpPlan:
    """Runs-level template core (f3dasm-free; shared with the tests).

    ``runs`` is the ``[(task, {step: loss_path})]`` list
    :func:`_iter_task_runs` yields; the public
    :func:`dump_databank_template` is the ExperimentData-facing wrapper.
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

    if not runs:
        raise ValueError("No finished (task, optimizer) runs to write.")

    rawroot.mkdir(parents=True, exist_ok=True)

    if mode == "append" and exists:
        return _template_append(
            store,
            rawroot,
            runs,
            root,
            dataset_name,
            easiest_quantile,
            taskset,
        )
    return _template_create(
        store,
        rawroot,
        runs,
        root,
        dataset_name,
        easiest_quantile,
        budget_steps,
        taskset,
    )


def _template_create(
    store: Path,
    rawroot: Path,
    runs: list,
    root: Path,
    dataset_name: str,
    easiest_quantile: float,
    budget_steps: int,
    taskset: str | None,
) -> DumpPlan:
    """Lay out the deferred zarr template for a fresh store."""
    taskids = [task.hash for task, _ in runs]
    schedules: list[str] = []
    for _, step_paths in runs:
        for step in step_paths:
            if step.name not in schedules:
                schedules.append(step.name)

    task0, steps0 = _template_sample_run(runs)
    sample_loss, sample_perf = _template_sample(
        task0, steps0, easiest_quantile, budget_steps
    )
    template, encoding = big_template(
        sample_loss, sample_perf, taskids, schedules
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

    return DumpPlan(
        store=str(store),
        final_store=str(store),
        rawroot=str(rawroot),
        taskids=taskids,
        schedules=schedules,
        run_taskids=taskids,
        budget_steps=budget_steps,
        mode="create",
        easiest_quantile=easiest_quantile,
        var_names=list(template.data_vars),
        root=str(root),
        dataset_name=dataset_name,
        taskset=taskset,
    )


def _validate_append_schema(
    old: xr.Dataset,
    sample_loss: xr.Dataset,
    sample_perf: xr.Dataset,
    loss_vars: list[str],
    perf_vars: list[str],
) -> None:
    """Refuse an append whose incoming runs don't match the store's schema.

    Two things must match exactly: the *set* of big per-pair variable
    names (a variable dropped from the incoming runs would otherwise
    silently vanish from the grown store; an extra one would crash deep
    in a parallel array job with a bare ``KeyError``), and each
    variable's *interior* shape -- the ``{dim name: size}`` beyond
    ``(taskID, schedule)``, e.g. ``realization`` or ``target`` (growing
    one of those would need every already-stored cell re-encoded, which
    in-place growth cannot do; the store's ``budget`` grid is already
    authoritative on append the same way, see :func:`_template_append`).
    Compared by name, not physical dim order: the incoming sample's
    natural dim order need not match the store's -- :func:`_write_pair`
    transposes explicitly at write time either way. Both raise, naming
    the old vs. incoming shapes/names and pointing at ``mode="replace"``
    as the remedy.

    The noise-free series ``output_min_true`` (ADR 0024) is the one
    variable allowed to differ, in one direction: noise-free runs may
    join a store that has it (their cells of it stay unwritten, as in a
    fresh mixed store), but noisy runs may not join a store without it.
    That store's targets, and the scores of any noisy task already in
    it, were computed from observed values, and growing it would score
    one dataset two ways. The sample is taken from a noisy task whenever
    the incoming runs hold one (:func:`_template_sample_run`), so the
    check sees every incoming run.
    """
    incoming_shape: dict[str, dict[str, int]] = {}
    for sample in (sample_loss, sample_perf):
        for name, var in sample.data_vars.items():
            interior = canonical_dims(var.dims)[2:]
            incoming_shape[name] = {d: sample.sizes[d] for d in interior}

    store_vars = set(loss_vars) | set(perf_vars)
    incoming_vars = set(incoming_shape)
    if OUTPUT_MIN_TRUE in incoming_vars - store_vars:
        raise ValueError(
            "cannot append to the store: the incoming runs include a "
            f"noisy task, whose noise-free series '{OUTPUT_MIN_TRUE}' the "
            "store has no variable for. Its targets and scores were "
            "computed from observed values; re-dump everything with "
            "mode='replace' (see l2co ADR 0024)."
        )
    if OUTPUT_MIN_TRUE in store_vars - incoming_vars:
        # Noise-free runs into a store with noisy tasks: allowed.
        incoming_vars = incoming_vars | {OUTPUT_MIN_TRUE}
        incoming_shape[OUTPUT_MIN_TRUE] = {
            d: old.sizes[d]
            for d in canonical_dims(old[OUTPUT_MIN_TRUE].dims)[2:]
        }
    if store_vars != incoming_vars:
        raise ValueError(
            "cannot append to the store: its per-pair variables do not "
            "match the incoming runs' -- variables only in the store: "
            f"{sorted(store_vars - incoming_vars)}; variables only in "
            f"the incoming runs: {sorted(incoming_vars - store_vars)}. "
            "Append requires an identical schema; re-dump with "
            "mode='replace' if the schema has genuinely changed."
        )
    for name in sorted(store_vars):
        old_interior = {
            d: old.sizes[d] for d in canonical_dims(old[name].dims)[2:]
        }
        if old_interior != incoming_shape[name]:
            raise ValueError(
                f"cannot append to the store: variable '{name}' has "
                f"interior shape {old_interior} in the store but "
                f"{incoming_shape[name]} in the incoming runs. Growing "
                "an interior axis would require re-encoding every "
                "stored cell; re-dump with mode='replace' instead."
            )


def _snapshot_for_append(
    store: Path,
    rollback: Path,
    *,
    big_vars: list[str],
    small_vars: list[str],
    incoming_pairs: set[tuple[str, str]],
    old_taskids: list[str],
    old_scheds: list[str],
) -> None:
    """Snapshot every path an in-place append is about to mutate.

    Growing the store touches: the root ``zarr.json``, each grown
    array's own ``zarr.json`` (rewritten by ``resize`` /
    ``consolidate_metadata``), the ``taskID`` / ``schedule`` coordinate
    directories (deleted and recreated -- their fixed-length string
    dtype cannot simply grow), and the single chunk file of any
    *incoming* ``(task, schedule)`` pair that already has a stored cell
    (a re-run pair overwrites it). Untouched old cells are never
    written to and need no snapshot.

    The coordinate and small-array directories are snapshotted whole --
    they hold O(1) chunks regardless of store size -- while the big
    per-pair arrays are snapshotted cell-by-cell for only the incoming
    pairs, so the whole call costs disk proportional to the incoming
    runs, not to the existing store. Every path is hard-linked (falling
    back to a copy across filesystems; see :func:`_link_or_copy`).
    """
    if rollback.exists():
        shutil.rmtree(rollback)
    rollback.mkdir(parents=True)

    _link_or_copy(store / "zarr.json", rollback / "zarr.json")

    for name in (TASK_ID, SCHEDULE, *small_vars):
        src_dir = store / name
        for f in src_dir.rglob("*"):
            if f.is_file():
                _link_or_copy(f, rollback / name / f.relative_to(src_dir))

    with silence_zarr_warnings():
        group = zarr.open_group(store, mode="r", use_consolidated=False)
    old_ti = {t: i for i, t in enumerate(old_taskids)}
    old_sj = {s: j for j, s in enumerate(old_scheds)}
    rerun_positions = {
        (old_ti[t], old_sj[s])
        for t, s in incoming_pairs
        if t in old_ti and s in old_sj
    }
    for name in big_vars:
        _link_or_copy(
            store / name / "zarr.json", rollback / name / "zarr.json"
        )
        arr = group[name]
        for i, j in rerun_positions:
            src = _cell_file(store, name, arr, i, j)
            if src.is_file():
                _link_or_copy(src, _cell_file(rollback, name, arr, i, j))


def _restore_rollback(store: Path, rollback: Path) -> None:
    """Undo an aborted append by replaying its rollback snapshot.

    The mirror of :func:`_snapshot_for_append`: every file the snapshot
    captured is moved back over its live counterpart -- same filesystem,
    so a rename -- returning the store to exactly its pre-append state.
    Paths the snapshot doesn't cover were never touched by the abort and
    are left alone. Called by :func:`_template_append` when it finds a
    leftover snapshot from a crashed append, before that append proceeds.
    """
    for f in rollback.rglob("*"):
        if f.is_file():
            rel = f.relative_to(rollback)
            dst = store / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, dst)
    shutil.rmtree(rollback, ignore_errors=True)


def _grow_store_for_append(
    store: Path,
    taskids: list[str],
    schedules: list[str],
    grow_vars: list[str],
) -> None:
    """Grow every array in ``store`` to the union ``(taskID, schedule)`` shape.

    Metadata-only for both the big region-written arrays and the small
    wholesale-written ones: :meth:`zarr.Array.resize` changes only the
    recorded shape, so every existing cell keeps its file key and bytes,
    and a position beyond the old shape simply reads back as the fill
    value. The ``taskID`` / ``schedule`` coordinate arrays are the one
    exception -- their fixed-length string dtype would truncate a
    longer new entry -- so they are deleted and rewritten fresh at the
    union values (creating them at union shape is what lets a variable
    the old store never had, e.g. ``provenance`` on a pre-provenance
    store, come into existence here with no special-casing: it simply
    isn't in ``grow_vars`` and gets created wholesale later by
    :func:`_finalize_small`). Re-consolidates at the end -- a resize
    alone leaves *consolidated* metadata stale (see
    :func:`_open_big_arrays`), so a plain ``xr.open_zarr`` right after
    this call would still report the pre-grow shape without it.
    """
    n_task, n_sched = len(taskids), len(schedules)
    with silence_zarr_warnings():
        group = zarr.open_group(store, mode="r+", use_consolidated=False)
        for name in grow_vars:
            arr = group[name]
            dim_names = arr.metadata.dimension_names
            new_shape = tuple(
                n_task if d == TASK_ID else (n_sched if d == SCHEDULE else s)
                for d, s in zip(dim_names, arr.shape, strict=True)
            )
            if new_shape != arr.shape:
                arr.resize(new_shape)

        shutil.rmtree(store / TASK_ID)
        shutil.rmtree(store / SCHEDULE)
        xr.Dataset(coords={TASK_ID: taskids, SCHEDULE: schedules}).to_zarr(
            store, mode="a", consolidated=False
        )
        zarr.consolidate_metadata(store)


def _template_append(
    store: Path,
    rawroot: Path,
    runs: list,
    root: Path,
    dataset_name: str,
    easiest_quantile: float,
    taskset: str | None,
) -> DumpPlan:
    """Grow the existing store in place over the old ∪ new union.

    Recovers first if a previous append crashed mid-flight -- a leftover
    rollback snapshot is restored (:func:`_restore_rollback`) before
    anything else happens. Then validates the incoming runs' schema
    against the store's (:func:`_validate_append_schema`), snapshots
    every path the grow is about to mutate and marks the store
    mid-append (:func:`_snapshot_for_append`), and finally grows every
    array to the union shape (:func:`_grow_store_for_append`) -- cost
    proportional to the incoming runs, not to the store's existing
    size. The incoming ``(task, schedule)`` cells read back as the fill
    value until the per-task step writes them; every untouched old cell
    is never written to, so its bytes are exactly what they were
    before this call.
    """
    rollback = layout.rollback_path(root, dataset_name)
    marker = layout.append_marker_path(root, dataset_name)
    if rollback.exists():
        logger.info(
            f"dump_databank: found a leftover rollback snapshot for "
            f"'{dataset_name}' from an aborted append; restoring it "
            "before proceeding."
        )
        _restore_rollback(store, rollback)
        marker.unlink(missing_ok=True)

    with silence_zarr_warnings():
        old = xr.open_zarr(store, consolidated=True)
    budget_steps = int(old.sizes["budget"])
    old_taskids = [str(t) for t in old[TASK_ID].values]
    old_scheds = [str(s) for s in old[SCHEDULE].values]
    old_targets = old[TARGET_VALUE].load()
    loss_vars = [v for v in old.data_vars if "budget" in old[v].dims]
    perf_vars = [
        v
        for v in old.data_vars
        if "target" in old[v].dims and v != TARGET_VALUE
    ]
    small_vars = [
        v
        for v in (
            TASK_LABELS,
            TARGET_VALUE,
            OPTIMIZATIONSTEP,
            PROVENANCE,
            IN_TRAINING,
        )
        if v in old.data_vars
    ]

    taskids = list(old_taskids)
    schedules = list(old_scheds)
    for task, step_paths in runs:
        if task.hash not in taskids:
            taskids.append(task.hash)
        for step in step_paths:
            if step.name not in schedules:
                schedules.append(step.name)

    task0, steps0 = _template_sample_run(runs)
    existing0 = (
        old_targets.sel({TASK_ID: task0.hash}).values
        if task0.hash in old_taskids
        else None
    )
    sample_loss, sample_perf = _template_sample(
        task0,
        steps0,
        easiest_quantile,
        budget_steps,
        existing_targets=existing0,
    )
    _validate_append_schema(
        old, sample_loss, sample_perf, loss_vars, perf_vars
    )

    big_vars = [*loss_vars, *perf_vars]
    incoming_pairs = {
        (task.hash, step.name)
        for task, step_paths in runs
        for step in step_paths
    }
    _snapshot_for_append(
        store,
        rollback,
        big_vars=big_vars,
        small_vars=small_vars,
        incoming_pairs=incoming_pairs,
        old_taskids=old_taskids,
        old_scheds=old_scheds,
    )
    marker.write_text(
        json.dumps(
            {
                "started": datetime.now(UTC).isoformat(),
                "old_tasks": len(old_taskids),
                "old_schedules": len(old_scheds),
                "incoming_tasks": len(taskids) - len(old_taskids),
                "incoming_schedules": len(schedules) - len(old_scheds),
            }
        )
    )

    existing_targets = {
        t: [
            float(x)
            for x in np.asarray(old_targets.sel({TASK_ID: t}).values).ravel()
        ]
        for t in old_taskids
    }
    run_taskids = [task.hash for task, _ in runs]
    old.close()

    _grow_store_for_append(store, taskids, schedules, [*big_vars, *small_vars])

    return DumpPlan(
        store=str(store),
        final_store=str(store),
        rawroot=str(rawroot),
        taskids=taskids,
        schedules=schedules,
        run_taskids=run_taskids,
        budget_steps=budget_steps,
        mode="append",
        easiest_quantile=easiest_quantile,
        var_names=big_vars,
        root=str(root),
        dataset_name=dataset_name,
        existing_targets=existing_targets,
        taskset=taskset,
        old_schedules=old_scheds,
    )


def dump_databank_task(
    experiment_data,
    tasks_experiment_data,
    schedules_experiment_data,
    plan: DumpPlan,
    *,
    only_task_hashes: set[str] | None = None,
) -> list[xr.Dataset]:
    """Phase B: region-write tasks from ``experiment_data`` into the store.

    Embarrassingly parallel: a caller fans this out with one task per
    SLURM-array process by passing the *pristine* combined
    ExperimentData together with ``only_task_hashes={that task's hash}``.
    The task subsetting is done here at the ``runs`` level -- **after**
    :func:`_iter_task_runs` has run its ``select_with_status`` /
    ``reset_index`` preamble on the unmodified ED -- rather than by
    pre-slicing the ED, which would leave its status index inconsistent
    and break the preamble's re-filter.

    Each present ``(task, optimizer)`` cell is its own shard file keyed by
    ``(taskID, schedule)`` position, so disjoint tasks in separate
    processes never write the same file. Writes the raw loss sidecars as a
    side effect and returns one small payload :class:`~xarray.Dataset` per
    task (``task_labels`` + ``target_value``, with the schedules' recipe /
    provenance packed into ``.attrs["schedules_json"]`` and the task hash
    in ``.attrs["task_hash"]``) for the finalize step to assemble.

    Parameters
    ----------
    only_task_hashes : set[str] or None, optional
        When given, only tasks whose ``hash`` is in the set are written
        (others skipped). ``None`` writes every task present.
    """
    runs = [
        (task, step_paths)
        for task, step_paths in _iter_task_runs(
            experiment_data,
            tasks_experiment_data,
            schedules_experiment_data,
        )
        if step_paths
        and (only_task_hashes is None or task.hash in only_task_hashes)
    ]
    return _dump_task_runs(runs, plan)


def _dump_task_runs(runs: list, plan: DumpPlan) -> list[xr.Dataset]:
    """Runs-level per-task write core (f3dasm-free; shared with the tests).

    Region-writes every ``(task, {step: loss_path})`` in ``runs`` into
    ``plan.store`` and returns the small payloads; the public
    :func:`dump_databank_task` is the ExperimentData-facing wrapper.
    """
    rawroot = Path(plan.rawroot)
    arrays = _open_big_arrays(Path(plan.store), plan.var_names)

    payloads: list[xr.Dataset] = []
    for task, step_paths in runs:
        existing = (
            np.asarray(plan.existing_targets[task.hash])
            if task.hash in plan.existing_targets
            else None
        )
        task_ds, hash_to_step, loss_interp, perf, opt = _records_for_task(
            task,
            step_paths,
            plan.easiest_quantile,
            existing,
            plan.budget_steps,
            rawroot,
        )
        ti = plan.taskids.index(task.hash)
        sched_payload: dict[str, dict[str, str]] = {}
        for h, step in hash_to_step.items():
            sj = plan.schedules.index(step.name)
            _write_pair(arrays, loss_interp[h], ti, sj)
            _write_pair(arrays, perf[h], ti, sj)
            sched_payload[step.name] = {
                "recipe": str(opt[h][OPTIMIZATIONSTEP].values.flat[0]),
                "provenance": str(opt[h][PROVENANCE].values.flat[0]),
            }
        payload = task_ds.copy()
        payload.attrs["schedules_json"] = json.dumps(sched_payload)
        payload.attrs["task_hash"] = task.hash
        payloads.append(payload)
    return payloads


def _accumulate_schedule_payload(
    name: str,
    recipe: str,
    provenance: str,
    recipes: dict[str, xr.DataArray],
    provenances: dict[str, xr.DataArray],
) -> None:
    """First-contributor-wins accumulate from the per-task JSON payload.

    The parallel analogue of :func:`_accumulate_schedule`: rebuilds the
    recipe / provenance ``schedule``-axis DataArrays from the strings the
    per-task step packed, guarding the alias with
    :func:`_check_alias_consistency` for policy schedules.
    """
    if name in provenances:
        _check_alias_consistency(
            name, str(provenances[name].values.flat[0]), provenance
        )
        return
    recipes[name] = xr.DataArray(
        [recipe], coords={SCHEDULE: [name]}, dims=[SCHEDULE]
    )
    provenances[name] = xr.DataArray(
        [provenance], coords={SCHEDULE: [name]}, dims=[SCHEDULE]
    )


def _write_global_min_breaches(
    task_smalls: list[xr.Dataset], plan: DumpPlan
) -> None:
    """Record the tasks on which a run beat ``global_min``.

    An empirical ``global_min`` (e.g. CUTEst's) can sit above the true
    minimum. The dump scores such a run as reaching ``global_min`` (see
    :func:`l2co._src.performance.shift_to_quality_value`) and records it
    here, in :func:`~l2co._src.databank.layout.global_min_breaches_path`,
    so the value can be re-tuned. The file maps each breached ``taskID``
    to its ``task`` tag, ``global_min``, the lowest loss any schedule
    reached (``best``), how far below ``global_min`` that is
    (``margin``), and each breaching schedule's own ``best`` /
    ``margin`` / ``realization``.

    A fresh dump writes the file from scratch. An append merges into it:
    a schedule this dump wrote for a task replaces that schedule's old
    entry, or removes it if the new runs no longer breach.
    """
    path = layout.global_min_breaches_path(Path(plan.root), plan.dataset_name)
    tasks: dict[str, dict] = {}
    if plan.mode == "append" and path.is_file():
        tasks = json.loads(path.read_text())

    n_breached = 0
    for ds in task_smalls:
        task_hash = str(ds.attrs["task_hash"])
        written = set(json.loads(ds.attrs["schedules_json"]))
        old = tasks.pop(task_hash, None)
        new = (
            json.loads(ds.attrs[BREACHES_ATTR])
            if BREACHES_ATTR in ds.attrs
            else None
        )
        schedules = {
            name: record
            for name, record in (old or {}).get("schedules", {}).items()
            if name not in written
        }
        if new is not None:
            n_breached += 1
            schedules.update(new["schedules"])
        if not schedules:
            continue
        head = new if new is not None else old
        best = min(record["best"] for record in schedules.values())
        tasks[task_hash] = {
            "task": head["task"],
            GLOBAL_MIN: head[GLOBAL_MIN],
            "best": best,
            "margin": head[GLOBAL_MIN] - best,
            "schedules": dict(sorted(schedules.items())),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(tasks.items())), indent=2))
    if n_breached:
        logger.warning(
            f"dump_databank: on {n_breached} task(s), a run reached a loss "
            "below the task's global_min. Those runs score as reaching "
            f"global_min. Re-tune global_min from {path}."
        )


def dump_databank_finalize(
    task_smalls: list[xr.Dataset],
    plan: DumpPlan,
) -> Path:
    """Phase C: write small arrays, consolidate, archive, clear append state.

    Refuses first if the per-task step didn't cover every task this
    dump's runs should have written (a SLURM array row that died, e.g.
    under ``dump_collect``'s ``afterany`` dependency): the store is left
    exactly as the template step grew it, marker and rollback snapshot
    still in place, so re-submitting the dump chain always converges.

    Otherwise reassembles ``task_labels`` / ``target_value`` /
    ``optimizationstep`` / ``provenance`` from the per-task payloads
    (:func:`_accumulate_task` collects each task's row as-is; mismatched
    ``feature`` sets across tasks are reconciled -- not asserted equal
    -- by :func:`_finalize_small`'s outer-join concat; dedup schedules,
    first-contributor-wins with the same alias-consistency guard as the
    serial path), fills
    old-store fallbacks for append (using :attr:`DumpPlan.old_schedules`
    for the alias check -- the store's own ``schedule`` coordinate is
    already the union by this point, so it can't tell old from new),
    writes the small arrays + derived ``in_training`` mask via
    :func:`_finalize_small`, archives every *new* policy schedule's
    artifacts, optionally writes the taskset, and (append) removes the
    append marker and rollback snapshot -- the store is grown in place,
    so there is no swap.
    """
    store = Path(plan.store)

    covered = {str(ds.attrs["task_hash"]) for ds in task_smalls}
    missing = [t for t in plan.run_taskids if t not in covered]
    if missing:
        note = (
            " The store is left marked mid-append (rollback snapshot "
            "intact); re-submit the dump to complete it."
            if plan.mode == "append"
            else " Re-submit the dump to complete it."
        )
        raise RuntimeError(
            f"dump_databank_finalize: {len(missing)} of "
            f"{len(plan.run_taskids)} task(s) this dump should have "
            f"written produced no payload: {missing}." + note
        )

    labels: dict[str, xr.DataArray] = {}
    targets: dict[str, xr.DataArray] = {}
    recipes: dict[str, xr.DataArray] = {}
    provenances: dict[str, xr.DataArray] = {}

    for ds in task_smalls:
        th = str(ds.attrs["task_hash"])
        _accumulate_task(ds, th, labels, targets)
        for name, rp in json.loads(ds.attrs["schedules_json"]).items():
            _accumulate_schedule_payload(
                name, rp["recipe"], rp["provenance"], recipes, provenances
            )

    # Policy artifacts are archived only for schedules written by this
    # dump's runs -- capture the new names before the append fallback adds
    # old ones.
    new_schedule_names = set(recipes)

    if plan.mode == "append":
        with silence_zarr_warnings():
            old = xr.open_zarr(store, consolidated=True)
        old_labels = old[TASK_LABELS].load()
        old_targets = old[TARGET_VALUE].load()
        old_recipes = old[OPTIMIZATIONSTEP].load()
        if PROVENANCE in old:
            old_provenances = old[PROVENANCE].astype(object).load()
        else:
            old_provenances = xr.DataArray(
                np.array([""] * len(plan.old_schedules), dtype=object),
                coords={SCHEDULE: plan.old_schedules},
                dims=[SCHEDULE],
            )
        # An alias may only ever mean one artifact -- also across appends.
        # plan.old_schedules (captured before the grow) -- not the store's
        # own SCHEDULE coordinate, already the union -- says what's old.
        for name in list(provenances):
            if name in plan.old_schedules:
                _check_alias_consistency(
                    name,
                    str(old_provenances.sel({SCHEDULE: name}).values),
                    str(provenances[name].values.flat[0]),
                )
        for t in plan.taskids:
            if t not in labels:
                labels[t] = old_labels.sel({TASK_ID: [t]})
                targets[t] = old_targets.sel({TASK_ID: [t]})
        for s in plan.schedules:
            if s not in recipes:
                recipes[s] = old_recipes.sel({SCHEDULE: [s]})
                provenances[s] = old_provenances.sel({SCHEDULE: [s]})
        old.close()

    _finalize_small(
        store,
        labels,
        targets,
        recipes,
        provenances,
        plan.taskids,
        plan.schedules,
    )

    _write_global_min_breaches(task_smalls, plan)

    _archive_policy_artifacts(
        Path(plan.root),
        plan.dataset_name,
        _policy_provenances({s: provenances[s] for s in new_schedule_names}),
    )

    if plan.taskset is not None:
        save_taskset(
            layout.databank_root(Path(plan.root), plan.dataset_name),
            plan.taskset,
            plan.run_taskids,
        )

    if plan.mode == "append":
        layout.append_marker_path(Path(plan.root), plan.dataset_name).unlink(
            missing_ok=True
        )
        rollback = layout.rollback_path(Path(plan.root), plan.dataset_name)
        if rollback.exists():
            shutil.rmtree(rollback)

    logger.info(
        f"dump_databank: finalized {len(plan.taskids)} task(s) -> "
        f"{plan.final_store} (mode={plan.mode})"
    )
    return Path(plan.final_store).resolve()
