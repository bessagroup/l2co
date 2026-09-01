"""Open a databank into a single dataset / loader.

Opens the **single** consolidated store and lazily ``.sel``\\s the
requested ``taskID`` / ``schedule`` chunks. ``loss_history`` is already
interpolated to a fixed budget grid, so there is no per-pair resampling
at read time. Missing ``(task, optimizer)`` cells read back as ``NaN``
(unwritten chunks).

Numerical variables are returned as ``float`` (the on-disk dtype); cast
to ``int`` / ``bool`` downstream if needed.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
import logging
from pathlib import Path

# Third-party
import xarray as xr

from l2co._src.data.loader import DataLoader
from l2co._src.optimizer_schedule import OptimizationStep

# Local
from l2co.naming import (
    COST,
    IN_TRAINING,
    ITERATIONS,
    OPTIMIZATIONSTEP,
    PROVENANCE,
    SCHEDULE,
    SUCCESS,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
)

from . import layout
from .provenance import parse_provenance
from .records import loss_preprocess
from .tasksets import load_taskset
from .writer import silence_zarr_warnings

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)


def _resolve_schedule_names(
    ds: xr.Dataset, optimizers: list[OptimizationStep] | list[int] | list[str]
) -> list[str]:
    """Map the requested optimizers to ``schedule`` names in the store.

    The store keys ``schedule`` by the human-readable optimizer *name*.
    :class:`OptimizationStep` inputs use their ``.name``; ``str`` inputs
    are taken as schedule names directly; integer inputs are matched by
    ``OptimizationStep.hash`` reconstructed from each stored
    ``optimizationstep`` recipe.
    """
    if not optimizers:
        raise ValueError("'optimizers' must contain at least one entry.")
    name_by_hash: dict[int, str] = {}
    for name, recipe in zip(
        ds[SCHEDULE].values, ds[OPTIMIZATIONSTEP].values, strict=True
    ):
        step = OptimizationStep.from_dict(json.loads(str(recipe)))
        name_by_hash[step.hash] = str(name)
    names: list[str] = []
    for opt in optimizers:
        if isinstance(opt, OptimizationStep):
            names.append(opt.name)
        elif isinstance(opt, str):
            names.append(opt)
        else:
            names.append(name_by_hash[int(opt)])
    return names


def _resolve_task_ids(
    root: Path,
    dataset_name: str,
    available: list[str],
    task_ids: list[str] | None,
    taskset: str | None,
) -> list[str]:
    """Resolve selected task hashes from a list, a taskset, or the store.

    Tasksets live under the per-dataset folder
    (``<root>/<dataset_name>/tasksets``), written there by
    :func:`l2co._src.databank.writer.dump_databank`.
    """
    if task_ids is not None and taskset is not None:
        raise ValueError("Pass at most one of 'task_ids' / 'taskset'.")
    if taskset is not None:
        return load_taskset(layout.databank_root(root, dataset_name), taskset)
    if task_ids is not None:
        if not task_ids:
            raise ValueError("'task_ids' must contain at least one entry.")
        return list(task_ids)
    return available


def _open_store(
    root: Path,
    dataset_name: str,
    *,
    allow_incomplete: bool = False,
) -> xr.Dataset:
    """Open a dataset's consolidated store, guarding against a mid-append one.

    Every reader entry point that opens the zarr store routes through
    here, so the guard applies uniformly. An append grows the store *in
    place* (:func:`l2co._src.databank.writer._template_append`), so a
    store an aborted append left mid-flight would otherwise read back
    with the incoming ``(task, schedule)`` cells as ``NaN`` --
    indistinguishable from "not yet run". This raises instead unless
    ``allow_incomplete=True``; :func:`~l2co._src.databank.verify.
    verify_databank` passes that and reports the marker as a finding,
    since auditing a broken store is its job.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.
    allow_incomplete : bool, optional
        Read the store even if it is marked mid-append. Default
        ``False``.

    Returns
    -------
    xarray.Dataset
        The lazy (zarr-backed), consolidated store.

    Raises
    ------
    FileNotFoundError
        The store does not exist.
    RuntimeError
        The store is marked mid-append and ``allow_incomplete=False``.
    """
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        raise FileNotFoundError(f"No zarr store found at {store}.")
    if not allow_incomplete:
        marker = layout.append_marker_path(root, dataset_name)
        if marker.exists():
            try:
                info = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError):
                info = {}
            raise RuntimeError(
                f"databank '{dataset_name}' at {store} is mid-append "
                f"(started {info.get('started', 'unknown time')}, "
                f"+{info.get('incoming_tasks', '?')} task(s) / "
                f"+{info.get('incoming_schedules', '?')} schedule(s)); "
                "re-run the dump to complete it, or pass "
                "allow_incomplete=True to read the store as-is."
            )
    with silence_zarr_warnings():
        return xr.open_zarr(store, consolidated=True)


def populated_pairs(root: Path, dataset_name: str) -> set[tuple[str, str]]:
    """Return the ``(taskID, schedule)`` cells already written to a store.

    A cell counts as populated when its ``success`` is non-null for any
    interior index -- i.e. the ``(task, optimizer)`` pair was actually
    written. Unwritten cells read back as ``NaN`` and are excluded.

    Returns an **empty set** when the store does not exist yet, so a caller
    deciding which runs to launch can treat a missing store as "nothing
    present" without special-casing it.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory (the application supplies it).
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    set[tuple[str, str]]
        Every populated ``(task hash, schedule name)`` pair in the store.
    """
    if not layout.zarr_store_path(root, dataset_name).exists():
        return set()

    ds = _open_store(root, dataset_name)

    mask = ds[SUCCESS].notnull()
    for d in list(mask.dims):
        if d not in (TASK_ID, SCHEDULE):
            mask = mask.any(d)
    mask = mask.load()

    taskids = [str(t) for t in ds[TASK_ID].values]
    schedules = [str(s) for s in ds[SCHEDULE].values]
    pairs = {
        (t, s)
        for i, t in enumerate(taskids)
        for j, s in enumerate(schedules)
        if bool(mask.isel({TASK_ID: i, SCHEDULE: j}))
    }
    ds.close()
    return pairs


def populated_iterations(
    root: Path, dataset_name: str
) -> dict[tuple[str, str], int]:
    """Stored max-iterations per populated ``(taskID, schedule)`` cell.

    Like :func:`populated_pairs`, but instead of just *which* cells are
    written this reports, for each populated cell, the number of
    iterations its stored run used -- the ``iterations`` counter's max
    over the interior dims. ``iterations`` is a per-step cumulative sum
    (``+1`` per optimization step), so its max equals the run's
    ``n_iterations``; the write-time nearest-interpolation onto the fixed
    budget grid keeps that maximum exact. A caller deciding which runs to
    launch can therefore re-run only the cells whose proposed budget
    *exceeds* the stored one.

    Returns an **empty dict** when the store does not exist yet, so a
    missing store reads as "nothing present" (matching
    :func:`populated_pairs`).

    Reads the ``iterations`` variable (which carries the ``budget`` axis),
    so it is heavier than :func:`populated_pairs` -- which only touches
    ``success`` -- but it works against already-written stores without a
    schema change.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory (the application supplies it).
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    dict[tuple[str, str], int]
        ``{(task hash, schedule name): max_iterations}`` for every
        populated cell in the store.
    """
    if not layout.zarr_store_path(root, dataset_name).exists():
        return {}

    ds = _open_store(root, dataset_name)

    mask = ds[SUCCESS].notnull()
    for d in list(mask.dims):
        if d not in (TASK_ID, SCHEDULE):
            mask = mask.any(d)
    mask = mask.load()

    iters = ds[ITERATIONS]
    for d in list(iters.dims):
        if d not in (TASK_ID, SCHEDULE):
            iters = iters.max(d)
    iters = iters.load()

    taskids = [str(t) for t in ds[TASK_ID].values]
    schedules = [str(s) for s in ds[SCHEDULE].values]
    stored = {
        (t, s): int(round(float(iters.isel({TASK_ID: i, SCHEDULE: j}))))
        for i, t in enumerate(taskids)
        for j, s in enumerate(schedules)
        if bool(mask.isel({TASK_ID: i, SCHEDULE: j}))
    }
    ds.close()
    return stored


def _load_one(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None,
    taskset: str | None,
    strict: bool,
) -> xr.Dataset | None:
    """Open a ``(task, optimizer)`` selection from one store.

    The single-store worker behind :func:`load_databank_dataset`. With
    ``strict=True`` (the single-root default) a missing store or a
    selection with no present tasks / schedules raises. With
    ``strict=False`` (used per overlay root, where a store legitimately
    holds only *some* of the requested schedules — or none of them) the
    same conditions return ``None`` so the caller can skip that root's
    contribution instead of aborting the union.

    Parameters
    ----------
    dataset_name, optimizers, root, task_ids, taskset
        See :func:`load_databank_dataset`.
    strict : bool
        Raise (``True``) vs. return ``None`` (``False``) when the store
        is absent or contributes no cell.

    Returns
    -------
    xarray.Dataset | None
        The lazy selection, or ``None`` when ``strict=False`` and this
        store contributes nothing.
    """
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        if strict:
            raise FileNotFoundError(f"No zarr store found at {store}.")
        return None

    # A mid-append store refuses to load regardless of ``strict`` -- an
    # overlay root silently contributing NaN cells for the schedules an
    # aborted append was writing is exactly the failure mode the guard
    # exists to prevent, not something "no cell here" should paper over.
    ds = _open_store(root, dataset_name)

    names = _resolve_schedule_names(ds, optimizers)
    available = [str(t) for t in ds[TASK_ID].values]
    ids = _resolve_task_ids(root, dataset_name, available, task_ids, taskset)

    present_ids = [t for t in ids if t in available]
    missing_ids = [t for t in ids if t not in available]
    if missing_ids:
        logger.warning(
            f"load_databank_dataset: {len(missing_ids)} requested "
            f"task(s) absent from {store}: {missing_ids}"
        )
    if not present_ids:
        if strict:
            raise FileNotFoundError(
                f"None of the requested tasks are present in {store}."
            )
        return None

    store_names = set(str(s) for s in ds[SCHEDULE].values)
    present_names = [n for n in names if n in store_names]
    if not present_names:
        if strict:
            raise FileNotFoundError(
                f"None of the requested optimizers are present in {store}."
            )
        return None

    ds = ds.sel({TASK_ID: present_ids, SCHEDULE: present_names})

    # Completeness check: count populated cells.
    n_expected = ds.sizes[TASK_ID] * ds.sizes[SCHEDULE]
    succ = ds[SUCCESS].isel(
        {d: 0 for d in ds[SUCCESS].dims if d not in (TASK_ID, SCHEDULE)}
    )
    n_filled = int(succ.notnull().sum())
    logger.info(
        f"load_databank_dataset: {n_filled}/{n_expected} "
        f"(task, schedule) cells populated from {store}"
    )
    return ds


def load_databank_dataset(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
    overlay_roots: list[Path] | None = None,
) -> xr.Dataset:
    """Open a ``(task, optimizer)`` selection from the consolidated store.

    When ``overlay_roots`` is given, the same ``dataset_name`` is opened
    from ``root`` **and** each overlay root and the selections are unioned
    along ``schedule`` — the design for combining a read-only baseline
    databank with a separate learned-optimizer *runs* store (each store
    holds a disjoint slice of the requested schedules) without copying
    either. The union is virtual (lazy) and:

    - restricts to the ``taskID`` **intersection** across the contributing
      stores (a fair per-task comparison needs every schedule present on
      the same tasks);
    - keeps the first occurrence of a duplicated schedule name (``root``
      wins), so an overlay never shadows a primary record;
    - requires the contributing stores to share the same budget /
      realization grid (``xarray`` concat with ``join="exact"`` raises
      otherwise) — mixing stores dumped with different ``budget_steps`` is
      an error, not a silent truncation.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    optimizers : list[OptimizationStep] | list[int] | list[str]
        Selected optimizers, as :class:`OptimizationStep` recipes (their
        ``.name`` is matched), schedule-name strings (matched directly),
        or integer hashes (matched via the stored recipes). The selection
        may span schedules living in different roots.
    root : pathlib.Path
        The primary databank base directory (the application supplies it).
        Task-id / taskset resolution is anchored here.
    task_ids : list[str] | None, optional
        Explicit task hashes. Mutually exclusive with ``taskset``. If
        both are ``None``, every task in the primary store is selected.
    taskset : str | None, optional
        Name of a taskset file to resolve the task hashes from (read from
        the primary ``root``).
    overlay_roots : list[pathlib.Path] | None, optional
        Additional databank base directories overlaid on ``root``. Each is
        opened for the same ``dataset_name`` and its requested schedules
        are appended. ``None`` (default) is the single-store read.

    Returns
    -------
    xarray.Dataset
        Lazy (zarr-backed) selection. ``loss_history`` is already on the
        store's fixed budget grid; numerical variables are ``float``.

    Raises
    ------
    FileNotFoundError
        The primary store does not exist, or the selection is empty.
    ValueError
        ``optimizers`` is empty, or both ``task_ids`` and ``taskset``
        are given.
    """
    primary = _load_one(
        dataset_name,
        optimizers,
        root,
        task_ids=task_ids,
        taskset=taskset,
        strict=True,
    )
    if not overlay_roots:
        return primary

    # Anchor the task selection on the primary store: overlay roots are
    # restricted to the tasks the primary already resolved (explicit ids,
    # no re-resolution of a taskset that may not exist under an overlay).
    primary_ids = [str(t) for t in primary[TASK_ID].values]
    parts = [primary]
    seen_schedules = {str(s) for s in primary[SCHEDULE].values}
    for oroot in overlay_roots:
        part = _load_one(
            dataset_name,
            optimizers,
            oroot,
            task_ids=primary_ids,
            taskset=None,
            strict=False,
        )
        if part is None:
            continue
        fresh = [
            s
            for s in map(str, part[SCHEDULE].values)
            if s not in seen_schedules
        ]
        if not fresh:
            continue
        seen_schedules.update(fresh)
        parts.append(part.sel({SCHEDULE: fresh}))

    if len(parts) == 1:
        return primary

    # taskID intersection across every contributing store (primary order
    # preserved), then concat along schedule. join="exact" makes a budget /
    # realization grid mismatch fail loudly instead of silently dropping
    # points.
    present_per_part = [{str(t) for t in p[TASK_ID].values} for p in parts]
    common_ids = [
        t for t in primary_ids if all(t in s for s in present_per_part)
    ]
    if not common_ids:
        raise FileNotFoundError(
            "Overlay union is empty: the primary and overlay stores share "
            f"no task in dataset '{dataset_name}'."
        )
    parts = [p.sel({TASK_ID: common_ids}) for p in parts]
    combined = xr.concat(
        parts,
        dim=SCHEDULE,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )
    logger.info(
        f"load_databank_dataset: overlaid {len(parts)} store(s) into "
        f"{combined.sizes[SCHEDULE]} schedule(s) over "
        f"{len(common_ids)} shared task(s) for '{dataset_name}'"
    )
    return combined


def load_databank(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
    overlay_roots: list[Path] | None = None,
) -> DataLoader:
    """Open a selection and wrap it in a ``DataLoader``.

    Thin wrapper over :func:`load_databank_dataset`: drops the
    per-schedule metadata variables (``optimizationstep``,
    ``provenance``, ``in_training``) the DataLoader contract does not
    cover -- read those via :func:`load_databank_dataset`,
    :func:`load_provenance` or :func:`held_out_task_ids` -- and returns
    ``DataLoader.from_xarray(ds)``. The selection stays **lazy**
    (zarr-backed) -- dask reads only the chunks each downstream
    operation touches; call :meth:`l2co.DataLoader.load` to force the
    whole dataset into memory. See :func:`load_databank_dataset` for the
    parameter contract (including ``overlay_roots``).

    Returns
    -------
    DataLoader
        Loader wrapping the lazy selection.
    """
    ds = load_databank_dataset(
        dataset_name,
        optimizers,
        root,
        task_ids=task_ids,
        taskset=taskset,
        overlay_roots=overlay_roots,
    )
    drop = [v for v in (OPTIMIZATIONSTEP, PROVENANCE, IN_TRAINING) if v in ds]
    ds = ds.drop_vars(drop)
    return DataLoader.from_xarray(ds)


def load_databank_performance(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
    overlay_roots: list[Path] | None = None,
) -> DataLoader:
    """Open only the performance half of a selection as a loader.

    Like :func:`load_databank`, but keeps just the performance/task
    variables (``success`` / ``cost`` / ``target_value`` plus
    ``task_labels``), dropping the large ``loss_history`` arrays. The
    selection stays **lazy**, so the loss chunks are never read. Use this
    for ERTD reporting and any other performance-only consumer.

    Parameters
    ----------
    dataset_name, optimizers, root, task_ids, taskset, overlay_roots
        See :func:`load_databank_dataset`. ``overlay_roots`` unions a
        read-only baseline databank with a separate runs store for the
        combined ERTD report.

    Returns
    -------
    DataLoader
        Loader over the (lazy) performance variables.
    """
    ds = load_databank_dataset(
        dataset_name,
        optimizers,
        root,
        task_ids=task_ids,
        taskset=taskset,
        overlay_roots=overlay_roots,
    )
    # Drop the loss arrays (and the budget axis) by keeping only the
    # performance/task variables; the loss chunks stay unread.
    keep = [v for v in (SUCCESS, COST, TARGET_VALUE, TASK_LABELS) if v in ds]
    ds = ds[keep]
    return DataLoader.from_xarray(ds)


def load_loss_raw(
    dataset_name: str,
    root: Path,
    task_id: str,
    optimizer_hash: int,
    *,
    budget_steps: int | None = None,
) -> xr.Dataset:
    """Read one raw loss history from ``loss_history_raw/``.

    The raw histories keep their original variable-length ``budget``
    axis, so this is the source of truth for regenerating the
    interpolated store at a different ``budget_steps``.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    root : pathlib.Path
        The databank base directory.
    task_id : str
        Task hash.
    optimizer_hash : int
        Optimizer hash (the filename stem ``loss_<taskID>_<hash>``).
    budget_steps : int | None, optional
        If given, resample onto a common ``budget_steps`` grid via
        :func:`l2co._src.databank.records.loss_preprocess` (integer
        counters cast back to ``int``). If ``None`` (default) the raw
        dataset is returned unchanged.

    Returns
    -------
    xarray.Dataset
        The raw (or resampled) loss history.

    Raises
    ------
    FileNotFoundError
        The raw loss file does not exist.
    """
    rawroot = layout.raw_root(root, dataset_name)
    path = layout.loss_raw_path(rawroot, task_id, optimizer_hash)
    if not path.exists():
        raise FileNotFoundError(f"No raw loss history at {path}.")
    ds = xr.load_dataset(path)
    if budget_steps is not None:
        ds = loss_preprocess(ds, budget_steps)
    return ds


def load_provenance(root: Path, dataset_name: str) -> dict[str, dict]:
    """Parsed provenance record per policy schedule in a store.

    Static-optimizer schedules (empty provenance) are omitted, as is
    everything when the store predates the provenance schema, so the
    result maps exactly the policy schedules to their
    ``{"training": ..., "evaluation": ...}`` records (see
    :mod:`l2co._src.databank.provenance`).

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    dict[str, dict]
        ``{schedule name: provenance record}`` for every policy
        schedule.

    Raises
    ------
    FileNotFoundError
        The store does not exist.
    """
    ds = _open_store(root, dataset_name)
    out: dict[str, dict] = {}
    if PROVENANCE in ds:
        for name, value in zip(
            ds[SCHEDULE].values, ds[PROVENANCE].values, strict=True
        ):
            record = parse_provenance(value)
            if record is not None:
                out[str(name)] = record
    ds.close()
    return out


def held_out_task_ids(
    root: Path, dataset_name: str, schedule: str
) -> list[str]:
    """Task hashes a policy schedule was **not** trained on.

    Reads the store's ``in_training`` mask (materialized at dump time
    from the artifact's recorded training task hashes) and returns the
    tasks where it is ``0`` -- the selection to pass as ``task_ids``
    to :func:`load_databank` / :func:`load_databank_performance` for
    held-out-only comparisons (e.g. an ERTD of learned vs. static on
    unseen tasks).

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.
    schedule : str
        Name of the policy schedule.

    Returns
    -------
    list[str]
        Held-out task hashes, in store order.

    Raises
    ------
    FileNotFoundError
        The store does not exist.
    ValueError
        The schedule is absent, is not a policy schedule (static
        optimizers have no training set), or the store predates the
        provenance schema.
    """
    store = layout.zarr_store_path(root, dataset_name)
    ds = _open_store(root, dataset_name)
    try:
        if IN_TRAINING not in ds:
            raise ValueError(
                f"Store {store} has no '{IN_TRAINING}' variable (written "
                "before the provenance schema); re-dump the policy runs."
            )
        if schedule not in [str(s) for s in ds[SCHEDULE].values]:
            raise ValueError(
                f"Schedule '{schedule}' is not present in {store}."
            )
        mask = ds[IN_TRAINING].sel({SCHEDULE: schedule}).load()
        if mask.isnull().all():
            raise ValueError(
                f"Schedule '{schedule}' carries no in-training mask "
                "(static optimizers have no training set)."
            )
        return [
            str(t)
            for t, v in zip(ds[TASK_ID].values, mask.values, strict=True)
            if v == 0.0
        ]
    finally:
        ds.close()
