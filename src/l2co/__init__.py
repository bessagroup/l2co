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
from ._src.data_loader import DataLoader, create_entire_dataset
from ._src.preprocessing import preprocessing_factory
from ._src.stage0_model import Stage0Model
from ._src.stage1_model import Stage1Model
from ._src.tasks.benchmarkfunctions.run_optimization import \
    run_multiple_realizations
from ._src.tasks.benchmarkfunctions.run_sampling import run_sampling
from ._src.trainer import Trainer

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
    "Stage0Model",
    "Stage1Model",
    "run_multiple_realizations",
    "run_sampling",
    "Trainer",
    "create_entire_dataset"
]
