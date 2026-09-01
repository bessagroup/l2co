"""Consolidated-zarr databank (one store per dataset).

Public facade for :mod:`l2co._src.databank`. A whole dataset is stored in
a single zarr store (fast subset reads, no thousands-of-files open storm),
chunked one chunk per ``(task, optimizer)``; ``loss_history`` is
pre-interpolated to a fixed :data:`BUDGET_STEPS` grid and the raw
variable-length histories live in a ``loss_history_raw/`` netCDF sidecar.

>>> import l2co
>>> from pathlib import Path
>>> root = Path("/data/databank")   # supplied by the app's config
>>> store = l2co.databank.dump_databank(
...     experiment_data, tasks_ed, schedules_ed,
...     root=root, dataset_name="bbob",
... )
>>> d = l2co.databank.load_databank("bbob", optimizers, root=root)

Use :func:`load_databank_performance` for performance-only consumers
(ERTD reporting) so the large ``loss_history`` chunks are never read, and
:func:`load_loss_raw` to reach the raw variable-length histories. Use
:func:`verify_databank` to audit a store for anomalies (NaN/inf, partial
writes); it returns a structured report and the caller decides whether
to raise.

Learned optimizers (rl2co / l2co policies) enter the store as regular
schedules — same arrays, same dims — carrying a *provenance record*
that identifies the evaluated **policy artifact** (one checkpoint of
one training run). Training code writes a manifest next to every
checkpoint (:func:`mint_run_id` + :func:`write_manifest` /
:func:`bump_manifest`); evaluation loads it (:func:`load_manifest` — a
checkpoint without one is not admissible), assembles the record
(:func:`provenance_record`) and attaches it to the loss history as the
``l2co.naming.PROVENANCE`` attribute. The writer then stores the record
per schedule, materializes the ``in_training`` mask and archives the
artifact files under ``policy_artifacts/<schedule>/``. Read back with
:func:`load_provenance` / :func:`held_out_task_ids`. See
``docs/adr/0001-policy-artifact-provenance-manifest.md``.
"""

from l2co._src.databank import (
    BUDGET_STEPS,
    DatabankVerificationReport,
    DumpPlan,
    MissingManifestError,
    VerificationFinding,
    WeightsHashMismatchError,
    bump_manifest,
    databank_root,
    dump_databank,
    dump_databank_finalize,
    dump_databank_task,
    dump_databank_template,
    dump_runs,
    dumps_provenance,
    held_out_task_ids,
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
    load_manifest,
    load_provenance,
    load_taskset,
    loss_preprocess,
    manifest_path,
    mint_run_id,
    parse_provenance,
    policy_artifacts_root,
    policy_identity_hash,
    populated_iterations,
    populated_pairs,
    provenance_record,
    record_task_hashes,
    record_weights_hash,
    save_taskset,
    verify_databank,
    weights_hash,
    write_manifest,
    zarr_store_path,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

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
    "dump_runs",
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
