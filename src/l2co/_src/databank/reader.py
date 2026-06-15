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
    ITERATIONS,
    OPTIMIZATIONSTEP,
    SCHEDULE,
    SUCCESS,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
)

from . import layout
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


def populated_pairs(root: Path, dataset_name: str) -> set[tuple[str, str]]:
    """Return the ``(taskID, schedule)`` cells already written to a store.

    A cell counts as populated when its ``success`` is non-null for any
    interior index -- i.e. the ``(task, optimizer)`` pair was actually
    written (the same criterion :func:`l2co._src.databank.writer._dump_append`
    uses to copy untouched cells). Unwritten cells read back as ``NaN`` and
    are excluded.

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
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        return set()

    with silence_zarr_warnings():
        ds = xr.open_zarr(store, consolidated=True)

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
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        return {}

    with silence_zarr_warnings():
        ds = xr.open_zarr(store, consolidated=True)

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


def load_databank_dataset(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
) -> xr.Dataset:
    """Open a ``(task, optimizer)`` selection from the consolidated store.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    optimizers : list[OptimizationStep] | list[int] | list[str]
        Selected optimizers, as :class:`OptimizationStep` recipes (their
        ``.name`` is matched), schedule-name strings (matched directly),
        or integer hashes (matched via the stored recipes).
    root : pathlib.Path
        The databank base directory (the application supplies it).
    task_ids : list[str] | None, optional
        Explicit task hashes. Mutually exclusive with ``taskset``. If
        both are ``None``, every task in the store is selected.
    taskset : str | None, optional
        Name of a taskset file to resolve the task hashes from.

    Returns
    -------
    xarray.Dataset
        Lazy (zarr-backed) selection. ``loss_history`` is already on the
        store's fixed budget grid; numerical variables are ``float``.

    Raises
    ------
    FileNotFoundError
        The store does not exist.
    ValueError
        ``optimizers`` is empty, or both ``task_ids`` and ``taskset``
        are given.
    """
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        raise FileNotFoundError(f"No zarr store found at {store}.")

    with silence_zarr_warnings():
        ds = xr.open_zarr(store, consolidated=True)

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
        raise FileNotFoundError(
            f"None of the requested tasks are present in {store}."
        )

    store_names = set(str(s) for s in ds[SCHEDULE].values)
    present_names = [n for n in names if n in store_names]
    if not present_names:
        raise FileNotFoundError(
            f"None of the requested optimizers are present in {store}."
        )

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


def load_databank(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
) -> DataLoader:
    """Open a selection and wrap it in a ``DataLoader``.

    Thin wrapper over :func:`load_databank_dataset`: drops the
    ``optimizationstep`` provenance variable and returns
    ``DataLoader.from_xarray(ds)``. The selection stays **lazy**
    (zarr-backed) -- dask reads only the chunks each downstream
    operation touches; call :meth:`l2co.DataLoader.load` to force the
    whole dataset into memory. See :func:`load_databank_dataset` for the
    parameter contract.

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
    )
    if OPTIMIZATIONSTEP in ds:
        ds = ds.drop_vars(OPTIMIZATIONSTEP)
    return DataLoader.from_xarray(ds)


def load_databank_performance(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int] | list[str],
    root: Path,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
) -> DataLoader:
    """Open only the performance half of a selection as a loader.

    Like :func:`load_databank`, but keeps just the performance/task
    variables (``success`` / ``cost`` / ``target_value`` plus
    ``task_labels``), dropping the large ``loss_history`` arrays. The
    selection stays **lazy**, so the loss chunks are never read. Use this
    for ERTD reporting and any other performance-only consumer.

    Parameters
    ----------
    dataset_name, optimizers, root, task_ids, taskset
        See :func:`load_databank_dataset`.

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
