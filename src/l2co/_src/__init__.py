"""
Source code for the l2co package.

Modules
-------
_logger
    Module for logging.
collaborative_filtering
    Module for collaborative filtering.
content_based
    Module for content-based filtering.
data_loader
    Module for loading data.
ela
    Module for benchmarking model Exploratory Landscape Analysis
performance
    Module for performance evaluation.
plotting
    Module for plotting results.
preprocessing
    Module for preprocessing data.
stage0_model
    Module for creating a stage0 model.
stage1_model
    Module for creating a stage1 model.
trainer
    Module for training the optimization dynamics classifier.

Subpackage
----------
tasks
    Subpackage for optimization tasks.
"""

#                                                                       Modules
# =============================================================================

# Third-party
from f3dasm._src.datageneration.functions.function_factory import \
    FUNCTION_MAPPING
from matplotlib import pyplot as plt

from l2co._src.tasks.benchmarkfunctions.opfunu_functions import (
    CEC_FUNCTIONS, NAMED_FUNCTIONS)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


# make sure no plots are showing
plt.ioff()

# =============================================================================

FUNCTION_MAPPING.update(NAMED_FUNCTIONS)
FUNCTION_MAPPING.update(CEC_FUNCTIONS)
