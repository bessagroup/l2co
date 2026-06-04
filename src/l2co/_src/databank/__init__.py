"""Per-record databank: one file per task, optimizer, and (task, optimizer).

A lazy-loadable databank. A sweep is split into four typed subfolders
(``task``, ``optimizer``, ``loss_history``, ``performance``) plus
``tasksets``; files are hash-named so a ``(task, optimizer)`` selection
is addressable by path. Self-contained: it does not depend on the legacy
per-schedule databank. See ``README.md`` in this package for the full
data-structure reference and opening recipe.
"""

from .layout import (
    databank_root,
    loss_path,
    optimizer_path,
    perf_path,
    select_paths,
    task_path,
)
from .reader import load_databank, load_databank_dataset, loss_preprocess
from .tasksets import load_taskset, save_taskset
from .writer import dump_databank

__all__ = [
    "databank_root",
    "dump_databank",
    "load_databank",
    "load_databank_dataset",
    "load_taskset",
    "loss_path",
    "loss_preprocess",
    "optimizer_path",
    "perf_path",
    "save_taskset",
    "select_paths",
    "task_path",
]
