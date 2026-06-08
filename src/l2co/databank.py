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
:func:`load_loss_raw` to reach the raw variable-length histories.
"""

from l2co._src.databank import (
    BUDGET_STEPS,
    databank_root,
    dump_databank,
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
    load_taskset,
    loss_preprocess,
    save_taskset,
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
    "databank_root",
    "dump_databank",
    "load_databank",
    "load_databank_dataset",
    "load_databank_performance",
    "load_loss_raw",
    "load_taskset",
    "loss_preprocess",
    "save_taskset",
    "zarr_store_path",
]
