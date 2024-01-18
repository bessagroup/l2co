
"""
Submodule for data loading
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.data_loader import DataLoader
from ._src.perf_profile import load_all_data

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================

__all__ = ["DataLoader",
           "load_all_data"]
