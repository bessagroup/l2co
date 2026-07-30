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
swaps it in atomically (also memory-light, and it sidesteps the
fixed-length-string truncation that an in-place coordinate resize would
cause). Untouched cells are carried into the rebuilt store as hard
links (or copies) of their single chunk file -- the store keeps one
chunk per ``(task, optimizer)`` cell and the union coordinates preserve
every old index, so the file key is unchanged -- with a decode /
re-encode fallback for variables whose cell layout differs. All cell
writes go through zarr array handles opened once per dump
(:func:`_open_big_arrays`) rather than per-cell ``Dataset.to_zarr``
calls, which would re-read the store metadata on every call.
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


def _open_big_arrays(
    store: Path, names: Iterable[str]
) -> dict[str, zarr.Array]:
    """Open the store's big per-pair arrays once for repeated cell writes.

    A per-cell ``Dataset.to_zarr(region=...)`` call re-opens the group
    and re-reads every array's metadata, which dominates the write cost
    of the small per-pair cells; the returned handles amortize that to
    a single open per dump.
    """
    with silence_zarr_warnings():
        group = zarr.open_group(store, mode="r+")
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


def _cell_layout_key(arr: zarr.Array) -> str:
    """Serialized array metadata minus ``shape`` -- the cell layout.

    Two arrays with equal keys (dtype, chunk grid, codecs, fill value,
    ...) store a ``(taskID, schedule)`` cell as a byte-identical single
    chunk file, so a cell can be carried between them by copying that
    file. Serialized via JSON so the NaN fill value compares equal to
    itself (``nan != nan`` under ``dict`` equality).
    """
    meta = {k: v for k, v in arr.metadata.to_dict().items() if k != "shape"}
    return json.dumps(meta, sort_keys=True, default=str)


def _file_copyable(src: zarr.Array, dst: zarr.Array) -> bool:
    """Whether ``src`` cells can be carried into ``dst`` as file copies.

    Requires one chunk file per ``(taskID, schedule)`` cell (leading
    chunk axes of 1, interior chunk == interior shape) and an identical
    cell layout on both sides; anything else (e.g. an old store with a
    different realization count) falls back to the decode / re-encode
    copy.
    """
    chunk_shape = tuple(src.metadata.chunk_grid.chunk_shape)
    return chunk_shape == (1, 1, *src.shape[2:]) and _cell_layout_key(
        src
    ) == _cell_layout_key(dst)


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

    The rebuilt store sits next to the old one (same filesystem), so a
    link is the common case; the copy fallback covers filesystems
    without hard links. Linking is safe because the databank never
    modifies a written chunk file in place -- rewrites always land in a
    fresh temporary store that is swapped in whole.
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
    arrays: dict[str, zarr.Array] = {}

    for task, step_paths in tqdm(runs, total=len(runs)):
        task_ds, hash_to_step, loss_interp, perf, opt = _records_for_task(
            task, step_paths, easiest_quantile, None, budget_steps, rawroot
        )

        if not arrays:
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
            arrays = _open_big_arrays(store, list(template.data_vars))

        ti = taskids.index(task.hash)
        for h, step in hash_to_step.items():
            sj = schedules.index(step.name)
            _write_pair(arrays, loss_interp[h], ti, sj)
            _write_pair(arrays, perf[h], ti, sj)
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


def _carry_old_cells(
    *,
    old_store: Path,
    tmp: Path,
    old: xr.Dataset,
    mask: xr.DataArray,
    arrays: dict[str, zarr.Array],
    old_taskids: list[str],
    old_scheds: list[str],
    taskids: list[str],
    schedules: list[str],
    written: set[tuple[str, str]],
    loss_vars: list[str],
    perf_vars: list[str],
) -> None:
    """Carry untouched old cells from ``old_store`` into rebuilt ``tmp``.

    Variables whose cell layout matches the new template are carried as
    hard links / copies of their single chunk file (a missing source file
    stays absent -- it reads back as the NaN fill either way); the rest
    fall back to the per-cell decode / re-encode copy. A ``(task,
    schedule)`` pair in ``written`` (rewritten by the new runs) is skipped
    so the fresh cell is not clobbered by the stale copy.

    Shared by the serial :func:`_dump_append` and the parallel
    :func:`dump_databank_template`: in the parallel path the new-run cells
    are written by the array step *after* this carry-over, so ``written``
    holds every incoming ``(task, schedule)`` and those cells are left
    absent here for the array step to fill.
    """
    with silence_zarr_warnings():
        old_group = zarr.open_group(old_store, mode="r")
    all_vars = [*loss_vars, *perf_vars]
    fast_vars = [
        name
        for name in all_vars
        if _file_copyable(old_group[name], arrays[name])
    ]
    slow_vars = [name for name in all_vars if name not in fast_vars]
    if slow_vars:
        logger.info(
            f"dump_databank: cell layout of {slow_vars} differs from the "
            "existing store; carrying their cells via decode/re-encode."
        )
    for oi, t in enumerate(old_taskids):
        ti = taskids.index(t)
        for oj, s in enumerate(old_scheds):
            if not bool(mask.isel({TASK_ID: oi, SCHEDULE: oj})):
                continue
            if (t, s) in written:
                # A new run already (re)wrote this cell -- don't clobber it.
                continue
            sj = schedules.index(s)
            for name in fast_vars:
                src = _cell_file(old_store, name, old_group[name], oi, oj)
                if src.is_file():
                    _link_or_copy(
                        src, _cell_file(tmp, name, arrays[name], ti, sj)
                    )
            if slow_vars:
                with silence_zarr_warnings():
                    cell = (
                        old[slow_vars]
                        .isel({TASK_ID: [oi], SCHEDULE: [oj]})
                        .load()
                    )
                _write_pair(arrays, cell, ti, sj)


def _dump_append(
    store: Path,
    rawroot: Path,
    runs: list,
    easiest_quantile: float,
) -> tuple[int, dict[str, str]]:
    """Rebuild the store over the union of old and new coordinates.

    New cells come from ``runs`` and are written first; untouched old
    cells are then carried over as hard links (or copies) of their
    single chunk file -- the union coordinates keep every old index, so
    a cell's file key is unchanged -- falling back to a decode /
    re-encode copy for variables whose cell layout differs from the
    rebuilt template (see :func:`_file_copyable`). A ``(task,
    optimizer)`` pair present in **both** is *rewritten* from ``runs``
    (the carry-over step skips it), so re-running a pair -- e.g. at a
    higher iteration budget -- overwrites its stored cell instead of
    being clobbered by the stale copy. Existing tasks keep their stored
    ERT targets so appended (and re-run) optimizers stay comparable.
    The rebuild lands in a temporary store that is swapped in
    atomically.

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
    arrays: dict[str, zarr.Array] = {}
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
        if not arrays:
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
            arrays = _open_big_arrays(tmp, list(template.data_vars))
        ti = taskids.index(task.hash)
        for h, step in hash_to_step.items():
            sj = schedules.index(step.name)
            _write_pair(arrays, loss_interp[h], ti, sj)
            _write_pair(arrays, perf[h], ti, sj)
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

    # Carry untouched old cells into the rebuilt store.
    _carry_old_cells(
        old_store=store,
        tmp=tmp,
        old=old,
        mask=mask,
        arrays=arrays,
        old_taskids=old_taskids,
        old_scheds=old_scheds,
        taskids=taskids,
        schedules=schedules,
        written=written,
        loss_vars=loss_vars,
        perf_vars=perf_vars,
    )

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


#                          Parallel dump: template / per-task / finalize
# =============================================================================
# The three functions below decompose the serial dump into a shape that a
# SLURM-array pipeline can fan out: a single-process *template* that lays
# out the empty union store, an embarrassingly parallel *per-task* step
# that region-writes each task's disjoint cells (one shard file per
# ``(task, schedule)`` -- so separate processes never touch the same
# file), and a single-process *finalize* that writes the small string /
# target arrays, consolidates, archives policy artifacts and (append)
# swaps the rebuilt store in. They reuse the exact helpers the serial path
# uses, so the two paths stay behaviourally identical.


@dataclass
class DumpPlan:
    """Serializable hand-off between the three parallel-dump phases.

    Every field is JSON-native so the template step can persist it (e.g.
    ``json.dumps(dataclasses.asdict(plan))``) for the array + finalize
    steps to read back (``DumpPlan(**json.loads(...))``).

    Attributes
    ----------
    store : str
        The store the per-task step region-writes into. For ``append``
        this is the ``<store>.tmp`` rebuild; for ``create`` it is the
        final store itself.
    final_store : str
        Where the store lands. Equals ``store`` for ``create``; for
        ``append`` the finalize step swaps ``store`` -> ``final_store``.
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
    """Phase A: lay out the empty union store; return a :class:`DumpPlan`.

    Single-process, cheap: it creates the deferred (NaN, ``compute=False``)
    zarr template over the full ``(taskID, schedule)`` union and -- for
    ``append`` -- carries every untouched old cell into a ``<store>.tmp``
    rebuild via hard links, leaving the incoming cells absent for the
    per-task step to fill. No per-task loss history is written here (the
    one task loaded to size the template uses ``write_raw=False``).

    Takes the same three source ExperimentData as :func:`dump_databank`.
    Raises the same ``FileExistsError`` / ``ValueError`` as
    :func:`dump_databank` for ``mode="error"`` on an existing store and
    for an empty run set.
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
    """Template for a fresh store (mirrors :func:`_dump_create` setup)."""
    taskids = [task.hash for task, _ in runs]
    schedules: list[str] = []
    for _, step_paths in runs:
        for step in step_paths:
            if step.name not in schedules:
                schedules.append(step.name)

    task0, steps0 = runs[0]
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


def _template_append(
    store: Path,
    rawroot: Path,
    runs: list,
    root: Path,
    dataset_name: str,
    easiest_quantile: float,
    taskset: str | None,
) -> DumpPlan:
    """Template + old-cell carry-over for append (like :func:`_dump_append`).

    Rebuilds into ``<store>.tmp`` over the old ∪ new coordinates and links
    every untouched old cell across; the incoming ``(task, schedule)``
    cells are left absent for the per-task step to write. The old store
    stays intact until the finalize step swaps.
    """
    with silence_zarr_warnings():
        old = xr.open_zarr(store, consolidated=True)
    budget_steps = int(old.sizes["budget"])
    old_taskids = [str(t) for t in old[TASK_ID].values]
    old_scheds = [str(s) for s in old[SCHEDULE].values]
    old_targets = old[TARGET_VALUE].load()
    mask = _populated_mask(old)
    loss_vars = [v for v in old.data_vars if "budget" in old[v].dims]
    perf_vars = [
        v
        for v in old.data_vars
        if "target" in old[v].dims and v != TARGET_VALUE
    ]

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

    task0, steps0 = runs[0]
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
    template, encoding = big_template(
        sample_loss, sample_perf, taskids, schedules
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
    arrays = _open_big_arrays(tmp, list(template.data_vars))

    written = {
        (task.hash, step.name)
        for task, step_paths in runs
        for step in step_paths
    }
    _carry_old_cells(
        old_store=store,
        tmp=tmp,
        old=old,
        mask=mask,
        arrays=arrays,
        old_taskids=old_taskids,
        old_scheds=old_scheds,
        taskids=taskids,
        schedules=schedules,
        written=written,
        loss_vars=loss_vars,
        perf_vars=perf_vars,
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

    return DumpPlan(
        store=str(tmp),
        final_store=str(store),
        rawroot=str(rawroot),
        taskids=taskids,
        schedules=schedules,
        run_taskids=run_taskids,
        budget_steps=budget_steps,
        mode="append",
        easiest_quantile=easiest_quantile,
        var_names=list(template.data_vars),
        root=str(root),
        dataset_name=dataset_name,
        existing_targets=existing_targets,
        taskset=taskset,
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


def dump_databank_finalize(
    task_smalls: list[xr.Dataset],
    plan: DumpPlan,
) -> Path:
    """Phase C: write small arrays, consolidate, archive, and (append) swap.

    Reassembles ``task_labels`` / ``target_value`` / ``optimizationstep``
    / ``provenance`` from the per-task payloads (dedup schedules,
    first-contributor-wins with the same alias-consistency guard as the
    serial path), fills old-store fallbacks for append, writes the small
    arrays + derived ``in_training`` mask via :func:`_finalize_small`,
    archives every *new* policy schedule's artifacts, optionally writes
    the taskset, and finally swaps ``<store>.tmp`` -> store for append.
    """
    store = Path(plan.store)
    labels: dict[str, xr.DataArray] = {}
    targets: dict[str, xr.DataArray] = {}
    recipes: dict[str, xr.DataArray] = {}
    provenances: dict[str, xr.DataArray] = {}

    for ds in task_smalls:
        th = str(ds.attrs["task_hash"])
        labels[th] = ds[TASK_LABELS]
        targets[th] = ds[TARGET_VALUE]
        for name, rp in json.loads(ds.attrs["schedules_json"]).items():
            _accumulate_schedule_payload(
                name, rp["recipe"], rp["provenance"], recipes, provenances
            )

    # Policy artifacts are archived only for schedules written by this
    # dump's runs -- capture the new names before the append fallback adds
    # old ones (mirrors _dump_append's new_policy_provs).
    new_schedule_names = set(recipes)

    if plan.mode == "append":
        with silence_zarr_warnings():
            old = xr.open_zarr(Path(plan.final_store), consolidated=True)
        old_scheds = [str(s) for s in old[SCHEDULE].values]
        old_labels = old[TASK_LABELS].load()
        old_targets = old[TARGET_VALUE].load()
        old_recipes = old[OPTIMIZATIONSTEP].load()
        if PROVENANCE in old:
            old_provenances = old[PROVENANCE].astype(object).load()
        else:
            old_provenances = xr.DataArray(
                np.array([""] * len(old_scheds), dtype=object),
                coords={SCHEDULE: old_scheds},
                dims=[SCHEDULE],
            )
        # An alias may only ever mean one artifact -- also across appends.
        for name in list(provenances):
            if name in old_scheds:
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
        shutil.rmtree(Path(plan.final_store))
        store.rename(Path(plan.final_store))

    logger.info(
        f"dump_databank: finalized {len(plan.taskids)} task(s) -> "
        f"{plan.final_store} (mode={plan.mode})"
    )
    return Path(plan.final_store).resolve()
