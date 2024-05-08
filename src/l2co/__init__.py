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

from . import classifier, data, performance, preprocessing
from ._src._logging import logger
from ._src.run_optimization import run_multiple_realizations
from ._src.run_sampling import run_sampling
from ._src.stage1_model import LSTMMultiOutputClassifier

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================

__all__ = ["logger",
           "run_multiple_realizations",
           "run_sampling",
           "classifier",
           "data",
           "performance",
           "preprocessing",
           "LSTMAutoencoder",
           "LSTMMultiOutputClassifier",
           "set_size",
           ]
