"""Policy-artifact provenance for the databank.

A *policy artifact* is one checkpoint of one training run — the unit of
learned-optimizer identity in the databank (two checkpoints of one run
are distinct artifacts, as are two retrainings of the same config). A
checkpoint becomes an artifact by carrying a *provenance manifest*: a
JSON sidecar written next to the checkpoint **at training time** that
records where the weights came from (run, step, resolved config, code
versions, training task hashes). A checkpoint without a manifest is not
admissible to the databank — there is no backfill for pre-manifest
checkpoints and no bypass flag (see
``docs/adr/0001-policy-artifact-provenance-manifest.md``).

At evaluation time the application assembles a *provenance record*
(:func:`provenance_record`) from the loaded manifest(s) plus an
``evaluation`` section, serializes it onto the loss-history dataset as
the :data:`l2co.naming.PROVENANCE` attribute, and the writer stores it
per ``schedule`` in the consolidated store — alongside an
``in_training`` mask derived from the manifest's training task hashes
and an archived copy of the artifact files under
``policy_artifacts/<schedule>/``.

Composite artifacts (e.g. an L2CO stage-0 + stage-1 pair, each trained
in its own run) carry one manifest per component under a
``components`` map; every helper here accepts both shapes.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Local
from l2co.naming import PROVENANCE

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Schema version stamped into every manifest.
MANIFEST_VERSION = 1

# Optimizer kinds whose schedules are policy artifacts: the writer
# refuses to store a pair for these without a provenance record (the
# dump-time backstop; the evaluation pipeline fails fast earlier).
POLICY_OPTIMIZERS = ("l2co", "rl2co")

# Hyperparameter keys holding filesystem paths to artifact files. They
# are excluded from :func:`policy_identity_hash` — identity must derive
# from the weights, never from where a file happens to sit on disk.
_PATH_HYPERPARAMETERS = (
    "model",
    "feature_processor_task",
    "stage0_model",
    "stage1_model",
)

# Hyperparameter keys that are labels/provenance, not evaluation settings,
# and so are excluded from :func:`policy_identity_hash` alongside the path
# keys. ``alias`` (see
# :data:`l2co._src.optimizer_schedule.ALIAS_HYPERPARAMETER`) only sets the
# human-facing SCHEDULE label; two evaluations of the same weights under
# different aliases must still share one identity.
_NON_IDENTITY_HYPERPARAMETERS = (*_PATH_HYPERPARAMETERS, "alias")

# Sidecar files a checkpoint may consist of, next to the ``.eqx`` main
# file (e.g. a scikit Stage-0 model keeps its estimator in ``.joblib``).
_WEIGHTS_SUFFIXES = (".eqx", ".joblib", ".pkl")


class MissingManifestError(FileNotFoundError):
    """A checkpoint has no provenance manifest — not admissible."""


class WeightsHashMismatchError(ValueError):
    """Checkpoint bytes do not match the manifest's ``weights_hash``."""


def manifest_path(checkpoint: Path | str) -> Path:
    """Path of the provenance manifest sitting next to a checkpoint.

    ``checkpoint`` may be given with or without its file extension
    (loaders such as ``RL2COLoader`` take extension-less stems); the
    manifest is always ``<stem>.manifest.json`` in the same directory.
    """
    p = Path(checkpoint)
    return p.with_name(p.stem + ".manifest.json")


def _weights_files(checkpoint: Path | str) -> list[Path]:
    """Resolve the file(s) a checkpoint physically consists of.

    Returns the checkpoint file itself when ``checkpoint`` points at an
    existing file, otherwise every ``<stem><suffix>`` sibling for the
    known weight suffixes (an ``.eqx`` main file plus e.g. a
    ``.joblib`` estimator sidecar). Raises when nothing exists.
    """
    p = Path(checkpoint)
    if p.is_file():
        files = [p]
        files += [
            p.with_suffix(s)
            for s in _WEIGHTS_SUFFIXES
            if s != p.suffix and p.with_suffix(s).is_file()
        ]
        return files
    files = [p.with_suffix(s) for s in _WEIGHTS_SUFFIXES]
    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No checkpoint file found for '{checkpoint}' (tried the "
            f"path itself and the {_WEIGHTS_SUFFIXES} siblings)."
        )
    return files


def file_sha256(path: Path | str) -> str:
    """Hex SHA-256 of one file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_hash(checkpoint: Path | str) -> str:
    """Content hash of a checkpoint (all files it consists of).

    Single-file checkpoints hash to that file's SHA-256; multi-file
    checkpoints (``.eqx`` + sidecars) hash to the SHA-256 of the
    name-sorted per-file digests, so the hash is independent of where
    the files live but sensitive to each byte.
    """
    files = sorted(_weights_files(checkpoint), key=lambda p: p.name)
    if len(files) == 1:
        return file_sha256(files[0])
    joined = "\n".join(f"{f.name}:{file_sha256(f)}" for f in files)
    return hashlib.sha256(joined.encode()).hexdigest()


def mint_run_id(experiment: str) -> str:
    """Mint a training-run identifier at training start.

    ``<experiment>-<UTC timestamp>-<short hex>`` — human-sortable and
    unconditionally available (no SLURM job id, no external tracker).
    Mint **once** when training begins and reuse it for every
    checkpoint the run saves; a resumed run keeps its original id.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}-{stamp}-{secrets.token_hex(2)}"


def write_manifest(
    checkpoint: Path | str,
    *,
    run_id: str,
    experiment: str,
    training_step: int | None,
    config: dict | None,
    git: dict | None,
    training_task_hashes: list[str],
    training_data: dict | None = None,
    feature_processor: str | Path | None = None,
    total_steps: int | None = None,
    resumed_from: str | None = None,
) -> Path:
    """Write the provenance manifest next to a checkpoint.

    Called by the *training* code every time it saves a checkpoint —
    the only place all provenance facts are cheaply and reliably known.
    The manifest makes the checkpoint a policy artifact; without it the
    checkpoint is not admissible to the databank.

    Parameters
    ----------
    checkpoint : Path | str
        The checkpoint just saved (with or without extension). The
        ``weights_hash`` is computed from its file(s) now, so write the
        manifest *after* the checkpoint.
    run_id : str
        Training-run identifier from :func:`mint_run_id`, minted once
        at training start (kept across resumes).
    experiment : str
        Name of the experiment that produced the run.
    training_step : int | None
        Step at which this checkpoint was saved; ``None`` for one-shot
        fits (e.g. scikit stage-0 models).
    config : dict | None
        The full **resolved** training config, inlined — the manifest
        must answer content questions without the run directory.
    git : dict | None
        Per-package ``{name: {"sha": ..., "dirty": ...}}`` snapshot.
    training_task_hashes : list[str]
        Explicit hashes of every task instance trained on. Recorded
        eagerly because task hashes are cloudpickle-derived and not
        stable across library versions — a symbolic reference could
        not be re-resolved reliably later.
    training_data : dict | None, optional
        Human-readable description of the training data source
        (dataset name, taskset, tasks-config choice, ...). Merged into
        the ``training_data`` section next to the hash list.
    feature_processor : str | Path | None, optional
        Path of the paired feature processor, hashed and recorded so
        the artifact names everything needed to act.
    total_steps : int | None, optional
        Total planned steps of the run, when known.
    resumed_from : str | None, optional
        Note recorded when the run re-attached after an interruption
        (the ``run_id`` itself stays the same).

    Returns
    -------
    pathlib.Path
        Path of the written manifest.
    """
    config = dict(config) if config is not None else None
    config_hash = (
        hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()
        if config is not None
        else None
    )
    fp: dict[str, Any] | None = None
    if feature_processor is not None:
        fp = {
            "path": str(feature_processor),
            "hash": weights_hash(feature_processor),
        }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "experiment": experiment,
        "training_step": training_step,
        "total_steps": total_steps,
        "created_at": datetime.now(UTC).isoformat(),
        "weights_hash": weights_hash(checkpoint),
        "config": config,
        "config_hash": config_hash,
        "git": git,
        "training_data": {
            **(training_data or {}),
            "task_hashes": [str(h) for h in training_task_hashes],
        },
        "feature_processor": fp,
        "resumed_from": resumed_from,
    }
    path = manifest_path(checkpoint)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def bump_manifest(checkpoint: Path | str, training_step: int) -> Path:
    """Refresh an existing manifest after an in-place checkpoint save.

    For trainers that overwrite a single source-of-truth checkpoint
    every update: reload the manifest (skipping the weights check — the
    weights just changed), stamp the new ``training_step``, recompute
    ``weights_hash`` from the current file(s) and record the update
    time. Everything else (``run_id``, config, git, training data) is
    kept from training start.
    """
    manifest = load_manifest(checkpoint, verify_weights=False)
    manifest["training_step"] = int(training_step)
    manifest["weights_hash"] = weights_hash(checkpoint)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    path = manifest_path(checkpoint)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def load_manifest(
    checkpoint: Path | str, *, verify_weights: bool = True
) -> dict:
    """Load and validate the manifest of a checkpoint.

    This is the admissibility gate: a missing manifest raises
    :class:`MissingManifestError` (the checkpoint is not a policy
    artifact — retrain with manifest writing enabled; there is no
    backfill), and mismatching checkpoint bytes raise
    :class:`WeightsHashMismatchError` (the file was overwritten after
    the manifest was written).

    Parameters
    ----------
    checkpoint : Path | str
        The checkpoint whose sibling manifest to load.
    verify_weights : bool, optional
        Verify the checkpoint file(s) against the manifest's
        ``weights_hash``, by default True. Pass ``False`` only when
        the weights are known to have just changed (see
        :func:`bump_manifest`).

    Returns
    -------
    dict
        The parsed manifest.
    """
    path = manifest_path(checkpoint)
    if not path.is_file():
        raise MissingManifestError(
            f"No provenance manifest at {path}: '{checkpoint}' is not a "
            "policy artifact and is not admissible to the databank. "
            "Manifests are written at training time; pre-manifest "
            "checkpoints have no backfill — retrain to obtain one."
        )
    manifest = json.loads(path.read_text())
    for key in ("run_id", "weights_hash"):
        if not manifest.get(key):
            raise ValueError(
                f"Provenance manifest {path} is missing required "
                f"field '{key}'."
            )
    if verify_weights:
        actual = weights_hash(checkpoint)
        if actual != manifest["weights_hash"]:
            raise WeightsHashMismatchError(
                f"Checkpoint '{checkpoint}' hashes to {actual[:12]}… but "
                f"its manifest records {manifest['weights_hash'][:12]}…: "
                "the weights were overwritten after the manifest was "
                "written, so the artifact's identity is broken."
            )
    return manifest


def provenance_record(training: dict, evaluation: dict) -> dict:
    """Assemble the per-schedule provenance record the store keeps.

    Parameters
    ----------
    training : dict
        Either a single manifest (as returned by :func:`load_manifest`)
        or ``{"components": {name: manifest, ...}}`` for composite
        artifacts (e.g. L2CO's stage-0 + stage-1 pair).
    evaluation : dict
        Evaluation-time facts: at least ``git`` and ``evaluated_at``;
        an ``artifacts`` map ``{label: {"path": ..., "weights_hash":
        ... | None}}`` names the physical files the writer archives
        into ``policy_artifacts/<schedule>/``.

    Returns
    -------
    dict
        ``{"training": training, "evaluation": evaluation}``.
    """
    manifests = _component_manifests(training)
    for name, m in manifests.items():
        if not m.get("run_id") or not m.get("weights_hash"):
            raise ValueError(
                f"training manifest '{name}' lacks run_id/weights_hash — "
                "load it via load_manifest() rather than hand-building it."
            )
    return {"training": training, "evaluation": evaluation}


def _component_manifests(training: dict) -> dict[str, dict]:
    """Normalize a training section to a ``{name: manifest}`` map."""
    if "components" in training:
        return dict(training["components"])
    return {"model": training}


def record_weights_hash(record: dict) -> str:
    """Combined weights hash identifying a record's policy artifact.

    Single-component records return the manifest's ``weights_hash``;
    composite records hash the name-sorted per-component hashes, so
    any component change changes the identity.
    """
    manifests = _component_manifests(record["training"])
    if len(manifests) == 1:
        return next(iter(manifests.values()))["weights_hash"]
    joined = "\n".join(
        f"{name}:{m['weights_hash']}" for name, m in sorted(manifests.items())
    )
    return hashlib.sha256(joined.encode()).hexdigest()


def record_task_hashes(record: dict) -> set[str]:
    """Union of training task hashes across a record's components.

    A task counts as *in-training* for the artifact when **any**
    component saw it — the conservative reading for held-out claims.
    """
    hashes: set[str] = set()
    for m in _component_manifests(record["training"]).values():
        hashes.update(
            str(h) for h in m.get("training_data", {}).get("task_hashes", [])
        )
    return hashes


def policy_identity_hash(record: dict, step) -> int:
    """Identity hash for a policy schedule's raw-sidecar filenames.

    Plays the role ``OptimizationStep.hash`` plays for static
    optimizers, but derives from the artifact's combined weights hash
    plus the evaluation-relevant hyperparameters — never from file
    paths (which are neither stable nor unique: the same artifact can
    move, and a path can be overwritten with new weights).

    Parameters
    ----------
    record : dict
        The provenance record (see :func:`provenance_record`).
    step : OptimizationStep
        The evaluated schedule; its hyperparameters minus the
        non-identity keys (:data:`_NON_IDENTITY_HYPERPARAMETERS` — the
        path-valued keys plus the label-only ``alias``) enter the hash.

    Returns
    -------
    int
        Same format as ``OptimizationStep.hash`` (first 6 hex chars of
        a SHA-256, as int), so raw filenames stay uniform.
    """
    settings = {
        k: v
        for k, v in sorted(step.hyperparameters.items())
        if k not in _NON_IDENTITY_HYPERPARAMETERS
    }
    payload = json.dumps(
        {"weights": record_weights_hash(record), "settings": settings},
        sort_keys=True,
        default=str,
    )
    return int(hashlib.sha256(payload.encode()).hexdigest()[:6], 16)


def parse_provenance(value) -> dict | None:
    """Parse one stored ``provenance`` cell back into a record.

    The store keeps an empty string for static-optimizer schedules
    (and legacy stores may miss the variable entirely); both read back
    as ``None``.
    """
    if value is None:
        return None
    s = str(value)
    if not s or s == "nan":
        return None
    return json.loads(s)


def dumps_provenance(record: dict) -> str:
    """Serialize a provenance record for the dataset attribute / store.

    The inverse of :func:`parse_provenance`; use this to attach the
    record to a loss-history dataset as
    ``ds.attrs[l2co.naming.PROVENANCE]``.
    """
    return json.dumps(record, default=str)


__all__ = [
    "MANIFEST_VERSION",
    "POLICY_OPTIMIZERS",
    "PROVENANCE",
    "MissingManifestError",
    "WeightsHashMismatchError",
    "bump_manifest",
    "dumps_provenance",
    "file_sha256",
    "load_manifest",
    "manifest_path",
    "mint_run_id",
    "parse_provenance",
    "policy_identity_hash",
    "provenance_record",
    "record_task_hashes",
    "record_weights_hash",
    "weights_hash",
    "write_manifest",
]
