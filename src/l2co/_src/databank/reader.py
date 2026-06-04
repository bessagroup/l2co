"""Open and merge a per-record databank into a single dataset.

Each dataset type is opened in its own :func:`xarray.open_mfdataset`
call (they have different variables and concat axes) and the results are
``xr.merge``d into one dataset. ``loss_history`` files store the *raw*,
variable-length ``budget`` axis; it is resampled onto the canonical L2CO
budget grid at open time by :func:`loss_preprocess`. The merged dataset
matches what :meth:`l2co.DataLoader.from_xarray` consumes, so the
training pipeline is unaffected.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Literal

# Third-party
import xarray as xr

# Local
from l2co._src.data.loader import DataLoader
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.validation import normalize_budget_grid
from l2co.naming import (
    FEVALS,
    ITERATIONS,
    OPTIMIZATIONSTEP,
    SCHEDULE,
    SUCCESS,
    TASK_ID,
    UPDATE_STEP,
)

from . import layout
from .tasksets import load_taskset

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# Integer counters that nearest-neighbour interpolation promotes to
# float and that must be cast back so dtypes stay consistent with the
# rest of L2CO (see CLAUDE.md "Default float dtype").
_INT_LOSS_VARS = (FEVALS, ITERATIONS, UPDATE_STEP)

# Shared open keywords for every dataset type. ``compat="override"``
# keeps the open lazy (it skips element-wise comparison of shared
# coords); ``join="outer"`` unions ``taskID`` / ``schedule`` and
# NaN-pads any missing ``(task, optimizer)`` combination.
_OPEN_KW = dict(
    combine="by_coords",
    join="outer",
    data_vars="minimal",
    coords="minimal",
    compat="override",
    engine="netcdf4",
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


def _resolve_optimizer_hashes(
    optimizers: list[OptimizationStep] | list[int],
) -> list[int]:
    """Map a list of ``OptimizationStep`` or ints to optimizer hashes."""
    if not optimizers:
        raise ValueError("'optimizers' must contain at least one entry.")
    return [
        opt.hash if isinstance(opt, OptimizationStep) else int(opt)
        for opt in optimizers
    ]


def _resolve_task_ids(
    rootdir: Path,
    task_ids: list[str] | None,
    taskset: str | None,
) -> list[str]:
    """Resolve the selected task hashes from an explicit list, a taskset
    name, or (as a fallback) every task present on disk."""
    if task_ids is not None and taskset is not None:
        raise ValueError("Pass at most one of 'task_ids' / 'taskset'.")
    if taskset is not None:
        return load_taskset(rootdir, taskset)
    if task_ids is not None:
        if not task_ids:
            raise ValueError("'task_ids' must contain at least one entry.")
        return list(task_ids)
    # Discover every task on disk (skip macOS AppleDouble sidecars).
    task_dir = layout.subdir(rootdir, layout.TASK_DIR)
    discovered = sorted(
        p.stem for p in task_dir.glob("*.nc") if not p.name.startswith("._")
    )
    if not discovered:
        raise FileNotFoundError(
            f"No task files found under {task_dir}; pass 'task_ids' or "
            "'taskset' explicitly."
        )
    return discovered


def load_databank_dataset(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int],
    location: Literal["local", "brown"],
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
    budget_steps: int = 1000,
) -> xr.Dataset:
    """Open a ``(task, optimizer)`` selection and merge it into one dataset.

    Builds the exact candidate paths from the cross product of the
    selected tasks and optimizers (see
    :func:`l2co._src.databank.layout.select_paths`), opens each dataset
    type with its own :func:`xarray.open_mfdataset`, and ``xr.merge``s
    the four. The ``loss_history`` open resamples each file's raw
    ``budget`` axis onto a common ``budget_steps`` grid via
    :func:`loss_preprocess`.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder in the databank.
    optimizers : list[OptimizationStep] | list[int]
        Selected optimizers, given as :class:`OptimizationStep` recipes
        (``step.hash`` is used) or raw integer hashes.
    location : {"local", "brown"}
        Which databank to read from.
    task_ids : list[str] | None, optional
        Explicit task hashes to load. Mutually exclusive with
        ``taskset``. If both are ``None``, every task on disk is loaded.
    taskset : str | None, optional
        Name of a taskset file to resolve the task hashes from (see
        :func:`l2co._src.databank.tasksets.load_taskset`).
    budget_steps : int, optional
        Number of points on the common budget grid, by default 1000.

    Returns
    -------
    xarray.Dataset
        The merged dataset over dims ``taskID``, ``schedule``,
        ``realization``, ``output_dim``, ``budget``, ``target``,
        ``metric``, ``feature``. Lazy (dask-backed) until ``.load()``.

    Raises
    ------
    ValueError
        ``optimizers`` is empty, or both ``task_ids`` and ``taskset``
        are given, or ``location`` is unknown.
    FileNotFoundError
        No matching ``loss_history`` files exist for the selection.
    """
    rootdir = layout.databank_root(location, dataset_name)
    hashes = _resolve_optimizer_hashes(optimizers)
    ids = _resolve_task_ids(rootdir, task_ids, taskset)

    paths = layout.select_paths(rootdir, ids, hashes)
    if not paths["loss_history"]:
        raise FileNotFoundError(
            f"No loss_history files found for the selection under {rootdir} "
            f"({len(ids)} task(s) x {len(hashes)} optimizer(s))."
        )

    loss = xr.open_mfdataset(
        paths["loss_history"],
        preprocess=partial(loss_preprocess, budget_steps=budget_steps),
        **_OPEN_KW,
    )
    perf = xr.open_mfdataset(paths["performance"], **_OPEN_KW)
    task = xr.open_mfdataset(paths["task"], **_OPEN_KW)
    opt = xr.open_mfdataset(paths["optimizer"], **_OPEN_KW)

    ds = xr.merge([loss, perf, task, opt], join="outer", compat="override")

    # Completeness check: join="outer" silently NaN-pads gaps.
    n_expected = ds.sizes[TASK_ID] * ds.sizes[SCHEDULE]
    n_filled = int(
        perf[SUCCESS]
        .isel(
            {d: 0 for d in perf[SUCCESS].dims if d not in (TASK_ID, SCHEDULE)}
        )
        .notnull()
        .sum()
    )
    logger.info(
        f"load_databank_dataset: {n_filled}/{n_expected} (task, schedule) "
        f"cells populated from {rootdir}"
    )
    return ds


def load_databank(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[int],
    location: Literal["local", "brown"],
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
    budget_steps: int = 1000,
) -> DataLoader:
    """Open a databank selection and wrap it in a :class:`DataLoader`.

    Thin wrapper over :func:`load_databank_dataset`: materialises the
    merged dataset, drops the ``optimizationstep`` recipe variable
    (carried only for provenance, not training), and returns
    ``DataLoader.from_xarray(ds)``. The remaining variables match the
    legacy ``entire_dataset.odata`` schema, so downstream
    feature/label extraction is unchanged. See
    :func:`load_databank_dataset` for the full parameter contract.

    Returns
    -------
    DataLoader
        Loader wrapping the merged dataset.
    """
    ds = load_databank_dataset(
        dataset_name=dataset_name,
        optimizers=optimizers,
        location=location,
        task_ids=task_ids,
        taskset=taskset,
        budget_steps=budget_steps,
    ).load()
    if OPTIMIZATIONSTEP in ds:
        ds = ds.drop_vars(OPTIMIZATIONSTEP)
    return DataLoader.from_xarray(ds)
