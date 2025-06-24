"""
Submodule with sampling strategies
"""

#                                                                       Modules
# =============================================================================

# Local

from ._src.sampler import (GridWeightSampler, NormalWeightSampler,
                           RandomWeightSampler, XavierNormalWeightSampler)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================

__all__ = [
    "RandomWeightSampler",
    "NormalWeightSampler",
    "GridWeightSampler",
    "XavierNormalWeightSampler",
]
