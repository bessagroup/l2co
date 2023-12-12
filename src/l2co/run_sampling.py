"""
Module containing functions that can be used to sample from the search-space
of a benchmark function using random sampling

Functions
---------
run_sampling
    Sample from the search-space of a benchmark function using a given sampling
    technique.
"""

#                                                                       Modules
# =============================================================================

# Standard
from time import perf_counter

# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from f3dasm.design import make_nd_continuous_domain

# Local
from ._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def run_sampling(data_generator: str, sampler: str, dimensionality: int,
                 noise: float, offset: bool,
                 seed: int, n_samples: int,
                 parallelization: bool) -> xr.Dataset:
    """
    Sample from the search-space of a benchmark function using a given
    sampling technique.


    Parameters
    ----------
    data_generator : str
        Name of the benchmark function to use
    sampler : str
        Sampling technique to use
    dimensionality : int
        dimensionality of the benchmark function
    noise : float
        noise added to the benchmark function
    offset : bool, optional
        if the benchmark function is offset
    seed : int
        Seed for the pseudo-random number generator
    n_samples : int
        number of samples
    parallelization : bool
        if the evaluation of the samples should be parallelized

    Returns
    -------
    xarray.Dataset
        xarray Dataset object containg the results of the sampling
    """
    start_timer = perf_counter()
    parallelization = 'parallel' if parallelization is True else 'sequential'

    domain = make_nd_continuous_domain(bounds=np.tile(
        [0., 1.], (dimensionality, 1)), dimensionality=dimensionality)

    kwargs = {'seed': seed,
              'scale_bounds': domain.get_bounds(),
              'noise': noise,
              'offset': offset}

    data = ExperimentData.from_sampling(
        sampler=sampler, domain=domain, n_samples=n_samples, seed=seed)

    # Evaluate the initial samplers
    data.evaluate(data_generator=data_generator,
                  kwargs=kwargs, mode=parallelization)

    logger.info(f"Sampled {dimensionality}D {data_generator} function "
                f"({perf_counter() - start_timer:.2f} s)")

    return data.to_xarray()
