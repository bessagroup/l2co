"""
l2co - A package for learning to choose optimizers
==================================================

This package investigates the use of machine learning to automatically select
the most suitable optimizer for a given optimization problem.

Usage
-----
>>> import l2co

Author
------
Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

# Local imports
from l2co import naming, optimizers, plotting, sampling
from l2co._src import tree_utils
from l2co._src.batching import BatchState
from l2co._src.data_loader import DataLoader
from l2co._src.history_state import HistoryState
from l2co._src.models import AbstractModel, Stage0Loader, Stage0Model
from l2co._src.opt_history import OptHistory
from l2co._src.optimizers import optimizer_mapping
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.preprocessing import preprocessing_factory
from l2co._src.run_optimization import RolloutWrapper, RunOptimization
from l2co._src.run_state import RunState, reset, run
from l2co._src.task import Task
from l2co._src.task_features import FeatureProcessor, MixedEmbedding
from l2co._src.update_class import UpdateClass, evaluate
from l2co._src.validation import validation_fn

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "DataLoader",
    "HistoryState",
    "AbstractModel",
    "OptimizationStep",
    "RunState",
    "RunOptimization",
    "Stage0Loader",
    "Task",
    "FeatureProcessor",
    "MixedEmbedding",
    "BatchState",
    "OptHistory",
    "optimizer_mapping",
    "validation_fn",
    "preprocessing_factory",
    "plotting",
    "sampling",
    "naming",
    "UpdateClass",
    "RolloutWrapper",
    "reset",
    "run",
    "evaluate",
    "Stage0Model",
    "tree_utils",
]
