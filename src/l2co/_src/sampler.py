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
    """
    Perform random sampling within the specified bounds.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    params : PyTree
        Parameters to sample.
    n_samples : int
        Number of samples to generate.
    lower_bound : float, optional
        Lower bound for sampling, by default 0.0.
    upper_bound : float, optional
        Upper bound for sampling, by default 1.0.

    Returns
    -------
    InputParameters
        Randomly sampled parameters.
    """
    return jax.tree.map(lambda x: jr.uniform(
        key, shape=(n_samples,) + params.shape,
        minval=lower_bound,
        maxval=upper_bound),
        params)
