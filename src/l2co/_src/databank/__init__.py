"""Consolidated-zarr databank: one store per dataset (+ raw loss sidecar).

A whole dataset lives in a single zarr store chunked one chunk per
``(taskID, schedule)``, so a subset read is a lazy ``.sel`` over the
selected chunks instead of opening thousands of files. ``loss_history``
is pre-interpolated to a fixed :data:`BUDGET_STEPS` budget grid; the
original variable-length histories are kept as per-pair netCDF under
``loss_history_raw/`` for regeneration, and named task selections live
under ``tasksets/``. Audit a store's integrity with
:func:`verify_databank`. See ``README.md`` in this package for the full
layout and rationale.
"""

from .layout import BUDGET_STEPS, databank_root, zarr_store_path
from .reader import (
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
    populated_iterations,
    populated_pairs,
)
from .records import loss_preprocess
from .tasksets import load_taskset, save_taskset
from .verify import (
    DatabankVerificationReport,
    VerificationFinding,
    verify_databank,
)
from .writer import dump_databank

__all__ = [
    "BUDGET_STEPS",
    "DatabankVerificationReport",
    "VerificationFinding",
    "databank_root",
    "dump_databank",
    "load_databank",
    "load_databank_dataset",
    "load_databank_performance",
    "load_loss_raw",
    "load_taskset",
    "loss_preprocess",
    "populated_iterations",
    "populated_pairs",
    "save_taskset",
    "verify_databank",
    "zarr_store_path",
]
