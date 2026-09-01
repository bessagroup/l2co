"""Filesystem layout and storage constants for the databank.

A dataset is stored as a **single zarr store** plus a sidecar folder of
raw loss histories and a folder of human-editable tasksets, all under a
caller-supplied root directory ``root``::

    <root>/<dataset_name>.zarr/                # consolidated store
        loss_history (interp), performance, task, optimizer
    <root>/<dataset_name>/loss_history_raw/    # per-pair raw netCDF
        loss_<taskID>_<optimizer_hash>.nc
    <root>/<dataset_name>/tasksets/<name>.txt  # named task selections

The databank holds **no machine-specific paths** of its own: the caller
passes ``root`` (the application resolves it from its own config, e.g. a
per-cluster ``databank_root`` key). The zarr ``loss_history`` is
pre-interpolated to a fixed :data:`BUDGET_STEPS` grid so every array is
dense and uniform; a subset read is a lazy ``.sel`` over the ``taskID`` /
``schedule`` chunks rather than opening thousands of files.
``loss_history_raw`` keeps the original variable-length ``budget`` axis so
the interpolated store can be regenerated at a different ``BUDGET_STEPS``
without re-running the sweep. See ``README.md`` in this package for the
full data-structure reference.

``mode="append"`` grows the store's arrays in place (see
:mod:`l2co._src.databank.writer`) rather than rebuilding it, so it marks
the store mid-flight with an :func:`append_marker_path` sidecar (removed
on success; a leftover marker means a crashed append) and stages an
:func:`rollback_path` snapshot of every path the grow is about to mutate
(removed on success; a leftover snapshot is auto-restored by the next
append).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import logging
from pathlib import Path

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# Folder of named, human-editable task selections (see ``tasksets.py``).
TASKSET_DIR = "tasksets"

# Subfolder holding the per-pair raw (variable-length budget) loss files.
LOSS_RAW_DIR = "loss_history_raw"

# Subfolder holding archived policy artifacts (checkpoint + manifest +
# feature processor per policy schedule). Training run dirs live on
# purged scratch; archiving here keeps the dataset self-sufficient for
# re-evaluation (see ``provenance.py``).
POLICY_ARTIFACTS_DIR = "policy_artifacts"

# Number of points on the fixed budget grid the zarr ``loss_history`` is
# interpolated onto at write time. The raw histories under
# ``loss_history_raw`` keep their original length, so regenerating the
# store with a different value is always possible.
BUDGET_STEPS = 1000

# Name of the in-store marker an append writes before growing the store
# and removes on successful finalize (see ``writer._template_append``).
# Living *inside* the store keeps the "do not trust this" flag attached
# to the store bytes across a copy (e.g. ``sync_databank.sh``).
APPEND_MARKER_NAME = "append_in_progress.json"

# Suffix of the sibling directory an append snapshots the store's
# about-to-be-mutated paths into before growing (see
# ``writer._snapshot_for_append`` / ``writer._restore_rollback``).
ROLLBACK_SUFFIX = ".rollback"


def databank_root(root: Path, dataset_name: str) -> Path:
    """Return the per-dataset folder (sidecar + tasksets live here).

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory (supplied by the application).
    dataset_name : str
        Name of the dataset folder (e.g. ``"bbob"``).

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>`` (not required to exist).
    """
    return Path(root) / dataset_name


def zarr_store_path(root: Path, dataset_name: str) -> Path:
    """Return the consolidated zarr store path for a dataset.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset (e.g. ``"bbob"``).

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>.zarr`` (sibling of the per-dataset
        folder; not required to exist).
    """
    return Path(root) / f"{dataset_name}.zarr"


def raw_root(root: Path, dataset_name: str) -> Path:
    """Return the sidecar folder holding the raw loss histories.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>/loss_history_raw`` (not required to
        exist).
    """
    return databank_root(root, dataset_name) / LOSS_RAW_DIR


def loss_raw_path(rawroot: Path, task_id: str, optimizer_hash: int) -> Path:
    """Path to one raw loss file ``loss_<taskID>_<optimizer_hash>.nc``.

    For static optimizers ``optimizer_hash`` is ``OptimizationStep.hash``;
    for policy schedules it is
    :func:`l2co._src.databank.provenance.policy_identity_hash` (derived
    from the artifact's weights hash + evaluation settings, never from
    file paths).
    """
    return rawroot / f"loss_{task_id}_{optimizer_hash}.nc"


def append_marker_path(root: Path, dataset_name: str) -> Path:
    """Path of the in-store "append in progress" marker.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>.zarr/append_in_progress.json`` (not
        required to exist -- its presence is the signal).
    """
    return zarr_store_path(root, dataset_name) / APPEND_MARKER_NAME


def rollback_path(root: Path, dataset_name: str) -> Path:
    """Path of an in-flight append's rollback snapshot directory.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>.zarr.rollback`` (sibling of the store,
        not required to exist).
    """
    store = zarr_store_path(root, dataset_name)
    return store.with_name(store.name + ROLLBACK_SUFFIX)


def policy_artifacts_root(root: Path, dataset_name: str) -> Path:
    """Return the folder holding archived policy artifacts.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset_name>/policy_artifacts`` (not required to
        exist). One subfolder per policy ``schedule``, holding the
        checkpoint file(s), the provenance manifest(s), the paired
        feature processor and a ``provenance.json`` with the full
        record.
    """
    return databank_root(root, dataset_name) / POLICY_ARTIFACTS_DIR
