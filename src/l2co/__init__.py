"""
l2co - A package for learning to choose optimizers

In this package, we investigate the use of machine learning to select the best
optimizer for a given optimization problem problem.

Usage
-----

>>> import l2co

Author: Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

# Local

import l2co._src.tree_utils as tree_utils
from l2co import plotting, sampling
from l2co._src._logging import logger
from l2co._src.data_loader import DataLoader, OptimizationDynamics
from l2co._src.history_state import HistoryState
from l2co._src.optimizers import OPTIMIZER_MAPPING, OptHistory, UpdateClass
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.preprocessing import preprocessing_factory
from l2co._src.run_optimization import RunOptimization
from l2co._src.stopping_criteria import (
    combined_stopping_criteria,
    covariance_matrix_stopping_criteria,
)
from l2co._src.task import Task
from l2co._src.update_state import UpdateState, new

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================
#
# =============================================================================

__all__ = [
    "DataLoader",
    "HistoryState",
    "OPTIMIZER_MAPPING",
    "UpdateClass",
    "OptHistory",
    "OptimizationDynamics",
    "OptimizationStep",
    "preprocessing_factory",
    "RunOptimization",
    "Task",
    "UpdateState",
    "new",
    "plotting",
    "sampling",
    "tree_utils",
    "logger",
    "combined_stopping_criteria",
    "covariance_matrix_stopping_criteria",
]
