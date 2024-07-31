"""
Source code for the l2co package.
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
