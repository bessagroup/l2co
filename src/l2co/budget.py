"""
Submodule with sampling strategies
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.budget import based_on_dimensionality, fixed_budget

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================
#
# =============================================================================

__all__ = [
    "fixed_budget",
    "based_on_dimensionality",
]
