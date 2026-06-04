"""Filesystem layout for the per-record databank.

A databank lives under ``<root>/<dataset_name>/`` and splits a sweep into
four typed subfolders plus a ``tasksets`` folder::

    <root>/<dataset_name>/
    ├── task/<taskID>.nc                       # one file per task
    ├── optimizer/<optimizer_hash>.nc          # one file per optimizer
    ├── loss_history/loss_<taskID>_<optimizer_hash>.nc
    ├── performance/perf_<taskID>_<optimizer_hash>.nc
    └── tasksets/<taskset>.txt

``taskID`` is :attr:`l2co_tasks.Task.hash` (a 16-hex string) and
``optimizer_hash`` is :attr:`l2co.OptimizationStep.hash` (an int). The
filenames mirror the concat axes so a selection is addressable directly
from the chosen ``(task, optimizer)`` combinations without globbing. See
``README.md`` in this package for the full data-structure reference.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Literal

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# Mapping from ``location`` keyword to the databank root directory on disk.
# Add a new entry here when introducing another machine. Kept self-contained
# (a copy of the legacy ``l2co._src.data.databank._LOCATION_ROOTS``) so this
# submodule has no dependency on the per-schedule databank module.
_LOCATION_ROOTS: dict[str, Path] = {
    "local": Path("/Volumes/T7/data/databank"),
    "brown": Path("/oscar/data/mbessa/mvander7/databank"),
}

# Subfolder names (one per dataset type) and the taskset folder.
TASK_DIR = "task"
OPTIMIZER_DIR = "optimizer"
LOSS_DIR = "loss_history"
PERF_DIR = "performance"
TASKSET_DIR = "tasksets"

_SUBDIRS = (TASK_DIR, OPTIMIZER_DIR, LOSS_DIR, PERF_DIR, TASKSET_DIR)


def databank_root(
    location: Literal["local", "brown"], dataset_name: str
) -> Path:
    """Return the root folder of a databank dataset.

    Parameters
    ----------
    location : {"local", "brown"}
        Which databank to use; maps to a fixed root directory via
        :data:`_LOCATION_ROOTS`.
    dataset_name : str
        Name of the dataset folder (e.g. ``"bbob"``).

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>`` (not required to exist).

    Raises
    ------
    ValueError
        ``location`` is not a known key of ``_LOCATION_ROOTS``.
    """
    if location not in _LOCATION_ROOTS:
        raise ValueError(
            f"Unknown location {location!r}; expected one of "
            f"{sorted(_LOCATION_ROOTS)}."
        )
    return _LOCATION_ROOTS[location] / dataset_name


def subdir(rootdir: Path, which: str) -> Path:
    """Return one of the typed subfolders under ``rootdir``."""
    if which not in _SUBDIRS:
        raise ValueError(
            f"Unknown subfolder {which!r}; expected one of {_SUBDIRS}."
        )
    return rootdir / which


def task_path(rootdir: Path, task_id: str) -> Path:
    """Path to ``task/<taskID>.nc``."""
    return rootdir / TASK_DIR / f"{task_id}.nc"


def optimizer_path(rootdir: Path, optimizer_hash: int) -> Path:
    """Path to ``optimizer/<optimizer_hash>.nc``."""
    return rootdir / OPTIMIZER_DIR / f"{optimizer_hash}.nc"


def loss_path(rootdir: Path, task_id: str, optimizer_hash: int) -> Path:
    """Path to ``loss_history/loss_<taskID>_<optimizer_hash>.nc``."""
    return rootdir / LOSS_DIR / f"loss_{task_id}_{optimizer_hash}.nc"


def perf_path(rootdir: Path, task_id: str, optimizer_hash: int) -> Path:
    """Path to ``performance/perf_<taskID>_<optimizer_hash>.nc``."""
    return rootdir / PERF_DIR / f"perf_{task_id}_{optimizer_hash}.nc"


def select_paths(
    rootdir: Path,
    task_ids: list[str],
    optimizer_hashes: list[int],
) -> dict[str, list[Path]]:
    """Build per-type path lists for a ``task × optimizer`` selection.

    The filenames are fully determined by ``(taskID, optimizer_hash)``,
    so the candidate paths are constructed directly from the cross
    product and split into the ones that exist on disk and the ones that
    are missing. Direct construction avoids a directory listing (one
    ``stat`` per candidate) and makes a ragged selection explicit
    instead of letting the dataset silently shrink. The ``task`` and
    ``optimizer`` paths come from the *distinct* ids, not the cross
    product.

    Parameters
    ----------
    rootdir : pathlib.Path
        Databank dataset root (see :func:`databank_root`).
    task_ids : list[str]
        Selected task hashes.
    optimizer_hashes : list[int]
        Selected optimizer hashes.

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Keys ``"task"``, ``"optimizer"``, ``"loss_history"``,
        ``"performance"`` map to the **existing** files of each type;
        ``"loss_history_missing"`` and ``"performance_missing"`` map to
        the ``(task, optimizer)`` files absent on disk. Missing
        combinations are also logged.
    """
    pairs = list(product(task_ids, optimizer_hashes))

    loss_candidates = [(loss_path(rootdir, t, h), (t, h)) for t, h in pairs]
    perf_candidates = [(perf_path(rootdir, t, h), (t, h)) for t, h in pairs]

    loss_existing = [p for p, _ in loss_candidates if p.is_file()]
    loss_missing = [p for p, _ in loss_candidates if not p.is_file()]
    perf_existing = [p for p, _ in perf_candidates if p.is_file()]
    perf_missing = [p for p, _ in perf_candidates if not p.is_file()]

    task_existing = [
        p for t in task_ids if (p := task_path(rootdir, t)).is_file()
    ]
    opt_existing = [
        p
        for h in optimizer_hashes
        if (p := optimizer_path(rootdir, h)).is_file()
    ]

    if loss_missing:
        logger.warning(
            f"select_paths: {len(loss_missing)}/{len(loss_candidates)} "
            f"loss_history (task, optimizer) files missing under {rootdir}"
        )

    return {
        "task": task_existing,
        "optimizer": opt_existing,
        "loss_history": loss_existing,
        "performance": perf_existing,
        "loss_history_missing": loss_missing,
        "performance_missing": perf_missing,
    }
