"""Per-record databank (one file per task / optimizer / pair).

Public facade for :mod:`l2co._src.databank`. A lazy-loadable alternative
to the legacy per-schedule databank exposed at the top level
(``l2co.dump_databank`` / ``l2co.load_databank`` etc.), kept under the
``l2co.databank`` namespace so the two coexist without name collisions.

>>> import l2co
>>> root = l2co.databank.dump_databank(
...     experiment_data, tasks_ed, schedules_ed,
...     location="local", dataset_name="bbob",
... )
>>> d = l2co.databank.load_databank("bbob", optimizers, location="local")
"""

from l2co._src.databank import (
    databank_root,
    dump_databank,
    load_databank,
    load_databank_dataset,
    load_taskset,
    loss_path,
    loss_preprocess,
    optimizer_path,
    perf_path,
    save_taskset,
    select_paths,
    task_path,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

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
