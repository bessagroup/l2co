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

from .layout import (
    BUDGET_STEPS,
    databank_root,
    policy_artifacts_root,
    zarr_store_path,
)
from .provenance import (
    MissingManifestError,
    WeightsHashMismatchError,
    bump_manifest,
    dumps_provenance,
    load_manifest,
    manifest_path,
    mint_run_id,
    parse_provenance,
    policy_identity_hash,
    provenance_record,
    record_task_hashes,
    record_weights_hash,
    weights_hash,
    write_manifest,
)
from .reader import (
    held_out_task_ids,
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
    load_provenance,
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
from .writer import (
    DumpPlan,
    dump_databank,
    dump_databank_finalize,
    dump_databank_task,
    dump_databank_template,
)

__all__ = [
    "BUDGET_STEPS",
    "DatabankVerificationReport",
    "DumpPlan",
    "MissingManifestError",
    "VerificationFinding",
    "WeightsHashMismatchError",
    "bump_manifest",
    "databank_root",
    "dump_databank",
    "dump_databank_finalize",
    "dump_databank_task",
    "dump_databank_template",
    "dumps_provenance",
    "held_out_task_ids",
    "load_databank",
    "load_databank_dataset",
    "load_databank_performance",
    "load_loss_raw",
    "load_manifest",
    "load_provenance",
    "load_taskset",
    "loss_preprocess",
    "manifest_path",
    "mint_run_id",
    "parse_provenance",
    "policy_artifacts_root",
    "policy_identity_hash",
    "populated_iterations",
    "populated_pairs",
    "provenance_record",
    "record_task_hashes",
    "record_weights_hash",
    "save_taskset",
    "verify_databank",
    "weights_hash",
    "write_manifest",
    "zarr_store_path",
]
