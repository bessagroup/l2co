"""Tasksets: named, human-editable selections of task hashes.

A *taskset* is a plain-text file under ``<rootdir>/tasksets/<name>.txt``
with one task hash per line. Blank lines and ``#`` comments are ignored,
and an inline ``#`` comment may follow a hash to record what it is (the
hash alone is opaque)::

    # tasksets/bbob.txt
    a1b2c3   # sphere     10D i3
    9f8e7d   # rastrigin  10D i1
    # f12 omitted intentionally — known instability

The file is one row per task and diffs cleanly in git, while the inline
``#`` descriptors keep the opaque hashes readable.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import logging
from pathlib import Path

from l2co._src.databank.layout import TASKSET_DIR

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)


def load_taskset(rootdir: Path, taskset: str) -> list[str]:
    """Load the task hashes of a named taskset.

    Strips ``#`` comments and blank lines and returns the task hashes as
    strings (kept as ``str`` so a numeric-looking or leading-zero hash is
    never coerced to ``int``). Raises if the named taskset is missing or
    empty rather than returning an empty list that would silently produce
    an empty dataset downstream.

    Parameters
    ----------
    rootdir : pathlib.Path
        Databank dataset root (see
        :func:`l2co._src.databank.layout.databank_root`).
    taskset : str
        Name of the taskset (without the ``.txt`` suffix).

    Returns
    -------
    list[str]
        Task hashes, in file order.

    Raises
    ------
    FileNotFoundError
        The taskset file does not exist.
    ValueError
        The taskset file contains no hashes (all blank/comment lines).
    """
    path = rootdir / TASKSET_DIR / f"{taskset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"taskset '{taskset}' not found at {path}")
    hashes = [
        line.split("#", 1)[0].strip() for line in path.read_text().splitlines()
    ]
    hashes = [h for h in hashes if h]  # drop blanks / comment-only lines
    if not hashes:
        raise ValueError(f"taskset '{taskset}' is empty: {path}")
    return hashes


def save_taskset(
    rootdir: Path,
    taskset: str,
    task_ids: list[str],
    descriptions: dict[str, str] | None = None,
) -> Path:
    """Write a taskset file from a list of task hashes.

    Parameters
    ----------
    rootdir : pathlib.Path
        Databank dataset root.
    taskset : str
        Name of the taskset (without the ``.txt`` suffix).
    task_ids : list[str]
        Task hashes to write, one per line, in order.
    descriptions : dict[str, str] | None, optional
        Optional ``taskID -> description`` mapping; a present
        description is written as an inline ``# ...`` comment. By
        default no inline comments are written.

    Returns
    -------
    pathlib.Path
        The written taskset file.
    """
    descriptions = descriptions or {}
    taskset_dir = rootdir / TASKSET_DIR
    taskset_dir.mkdir(parents=True, exist_ok=True)
    path = taskset_dir / f"{taskset}.txt"

    lines = []
    for t in task_ids:
        desc = descriptions.get(t)
        lines.append(f"{t}   # {desc}" if desc else t)
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"save_taskset: wrote {len(task_ids)} task(s) -> {path}")
    return path
