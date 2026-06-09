"""
Submodule with sampling strategies
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.sampler import (
    constant_sampling,
    get_sampler,
    grid_sampling,
    normal_sampling,
    random_sampling,
    xavier_sampling,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================
#
# =============================================================================

__all__ = [
    "random_sampling",
    "grid_sampling",
    "normal_sampling",
    "xavier_sampling",
    "constant_sampling",
    "get_sampler",
]
