"""
Module for samplers.
"""

#                                                                       Modules
# =============================================================================

# Third party
import jax
import jax.random as jr

# Local
from ._typing import InputParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def random_sampling(key, params, n_samples: int, lower_bound: float = 0.0,
                    upper_bound: float = 1.0) -> InputParameters:
    return jax.tree.map(lambda x: jr.uniform(
        key, shape=(n_samples,) + params.shape,
        minval=lower_bound,
        maxval=upper_bound),
        params)
