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

from . import plotting
from ._src._logging import logger
from ._src.data_loader import DataLoader, OptimizationDynamics
from ._src.l2co_model import L2CO_model
from ._src.optimizerschedule import OptimizationSchedule, OptimizationStep
from ._src.preprocessing import preprocessing_factory
from ._src.tasks.benchmarkfunctions.task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================

__all__ = [
    "plotting",
    "logger",
    "DataLoader",
    "preprocessing_factory",
    "OptimizationSchedule",
    "OptimizationStep",
    "OptimizationDynamics",
    "Task",
    "L2CO_model",
]
