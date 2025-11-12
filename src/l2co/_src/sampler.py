"""
Module for samplers.
"""

#                                                                       Modules
# =============================================================================

# Third party
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree

# Local
from .typing import InputParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def random_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> InputParameters:
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
    return jax.tree.map(
        lambda x: jr.uniform(
            key,
            shape=(n_samples,) + params.shape,
            minval=lower_bound,
            maxval=upper_bound,
        ),
        params,
    )


def normal_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
    mean: float = 0.0,
    std: float = 1.0,
) -> PyTree:
    """
    Perform normal (Gaussian) sampling for a given PyTree of parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    params : PyTree
        Parameters to sample (used for shape reference).
    n_samples : int
        Number of samples to generate.
    mean : float, optional
        Mean of the normal distribution, by default 0.0.
    std : float, optional
        Standard deviation of the normal distribution, by default 1.0.

    Returns
    -------
    PyTree
        Randomly sampled parameters from a normal distribution.
    """

    def sample_fn(x):
        shape = (n_samples,) + x.shape
        return mean + std * jr.normal(key, shape)

    return jax.tree.map(sample_fn, params)


def grid_sampling(
    params: PyTree,
    n_points_per_dim: int,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> InputParameters:
    grid_params = []
    for param in jax.tree_util.tree_leaves(params):
        if param.ndim == 0:  # Scalar parameter
            grid = jnp.linspace(lower_bound, upper_bound, num=n_points_per_dim)
        else:  # Tensor parameter
            grid = jnp.meshgrid(
                *[jnp.linspace(lower_bound, upper_bound, num=n_points_per_dim)]
                * param.size
            )
            # Reshape into parameter shape
            grid = jnp.stack(grid, axis=-1).reshape((-1,) + param.shape)
        grid_params.append(grid)
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params), grid_params
    )
