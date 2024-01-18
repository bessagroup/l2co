"""
Submodule for calculating the performance of optimizers
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.perf_profile import (create_performance_profile,
                                plot_performance_profile)
from ._src.performance import query_t

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================

__all__ = ["create_performance_profile",
           "plot_performance_profile",
           "query_t"]
