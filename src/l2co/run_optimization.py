"""
Module containing functions that can be used to compare the performance of
different optimizers.

Functions
---------

run_multiple_realizations
    Run multiple realizations of the same optimization experiment.
run_second_multiple_realizations
    Run multiple realizations of the second optimizer
run_first_multiple_realizations
    Run a single optimizer on a single function for multiple realizations.
experiment_two_optimizers
    Experiment where we find all combinations of two optimizers and run them
    each for half of the total budget over many realizations.
"""

#                                                                       Modules
# =============================================================================

# Standard
from copy import deepcopy
from functools import lru_cache
from itertools import product
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from f3dasm.design import make_nd_continuous_domain
from pathos.helpers import mp

# Local
from ._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

#                                                                     Constants
# =============================================================================

MAX_CACHE_SIZE = 20

#                                                                  Type aliases
# =============================================================================

OptimizerScheduler = List[Tuple[str, int, str]]

#                                                       Run single optimization
# =============================================================================


def run_multiple_realizations(
    optimizer_scheduler: OptimizerScheduler,
    data_generator: str,
    sampler: str,
    realizations: int,
    dimensionality: int,
    noise: float,
    offset: bool,
    n_samples: int,
    parallelization: bool,
    seed: int,
) -> xr.Dataset:
    """
    Run multiple realizations of the same optimization experiment.

    Parameters
    ----------
    optimizer_scheduler : List[Tuple[str, int, str]]
        List of tuples. For each tuple, determine the optimization\
        strategy used, the number of iterations\
        and the handshake method
    data_generator : str
        Name of the benchmark function to use
    sampler : str
        Sampling technique to use
    realizations : int
        number of realizations to run
    dimensionality : int
        dimensionality of the benchmark function
    noise : float
        noise added to the benchmark function
    offset : bool, optional
        if the benchmark function is offset
    n_samples : int
        number of initial samples
    parallelization : bool, optional
        if the optimizations are run in parallel on multiple cores\
        , by default True
    seed : int
        seed for the pseudo-random number generator

    Returns
    -------
    xarray.Dataset
        xarray Dataset containing the results of the optimizations

    Notes
    -----
    The parallelization is done using the pathos package.
    A message is logged to the L2CO logger once the optimization is completed.
    """

    start_timer = perf_counter()

    name = _get_name_from_scheduler(optimizer_scheduler)

    # Some optimizers are buggy when used in parallel
    buggy_optimizers = ['EvoSaxCMAES',
                        'EvoSaxSimAnneal', 'EvoSaxPSO', 'EvoSaxDE']
    if any(o[0] in buggy_optimizers for o in optimizer_scheduler):
        parallelization = False

    _optimizer_scheduler = deepcopy(optimizer_scheduler)

    optimizer, n_iterations, x0_selection = _optimizer_scheduler.pop(0)

    # FIRST OPTIMIZERS
    results = run_first_multiple_realizations(
        optimizer=optimizer,
        data_generator=data_generator,
        sampler=sampler,
        dimensionality=dimensionality,
        noise=noise,
        offset=offset,
        realizations=realizations,
        n_samples=n_samples,
        seed=seed,
        n_iterations=n_iterations,
        x0_selection=x0_selection,
        parallelization=parallelization,
    )

    logger.debug(f"LRU-cache: {run_first_multiple_realizations.cache_info()}")

    # SECOND OPTIMIZER
    domain = make_nd_continuous_domain(bounds=np.tile(
        [0., 1.], (dimensionality, 1)), dimensionality=dimensionality)

    kwargs = {'seed': seed,
              'scale_bounds': domain.get_bounds(),
              'noise': noise,
              'offset': offset}

    for optimizer, n_iterations, x0_selection in _optimizer_scheduler:
        results = run_second_multiple_realizations(
            results=results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection,
            parallelization=parallelization,
            sampler=sampler
        )

    logger.info(f"Optimized {dimensionality}D {data_generator} function with "
                f"{name}: "
                f"({perf_counter() - start_timer:.2f} s)")

    xarr = _listexperimentdata_to_dataset(results)

    # Add attributes
    xarr.attrs['number_of_samples']: int = n_samples
    xarr.attrs['realization_seeds']: List[int] = [
        seed + index for index in range(realizations)]

    # Benchmark functions
    xarr.attrs['function_seed']: int = seed
    xarr.attrs['function_name']: str = data_generator
    xarr.attrs['function_noise']: str = noise
    xarr.attrs['function_dimensionality']: int = dimensionality

    # Global minimum function
    func = _datagenerator_factory(data_generator, domain=domain, kwargs=kwargs)
    _, g = func.get_global_minimum(d=dimensionality)
    xarr.attrs['function_global_minimum']: float = float(
        np.array(g if not isinstance(g, list) else g[0])[0, 0])

    # Add another dimension with coordinate to xarr
    xarr = xarr.assign_coords({'optimizer': [name]})

    return xarr


def run_second_multiple_realizations(
    results: List[ExperimentData],
    optimizer: str,
    data_generator: str,
    kwargs: Dict[str, Any],
    n_iterations: int,
    x0_selection: str,
    parallelization: bool,
    sampler: str = 'random',
) -> List[ExperimentData]:
    """
    Run multiple realizations of the second optimizer

    Parameters
    ----------
    results : List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer.
    optimizer : str
        Name of the second optimizer
    data_generator : str
        Name of the benchmark function
    kwargs : Dict[str, Any]
        Key-word arguments of the data generator
    n_iterations : int
        number of iterations
    x0_selection : str
        x0 selection strategy
    parallelization : bool
        if the optimizations are run in parallel on multiple cores

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of ExperimentData objects with concatenated results from the
        second optimizer run.
    """

    if parallelization:
        results = _run_second_multiple_realizations_parallel(
            results=results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection, sampler=sampler
        )
    else:
        results = _run_second_multiple_realizations_sequential(
            results=results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection, sampler=sampler
        )

    return results


def _run_second_multiple_realizations_sequential(
    results: List[ExperimentData],
    optimizer: str,
    data_generator: str,
    kwargs: Dict[str, Any],
    n_iterations: int,
    x0_selection: str,
    sampler: str,
) -> ExperimentData:
    """
    Run multiple realizations of the second optimizer
    This operation is done sequentially


    Parameters
    ----------
    results : List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer.
    optimizer : str
        Name of the second optimizer
    data_generator : str
        Name of the benchmark function
    kwargs : Dict[str, Any]
        Key-word arguments of the data generator
    n_iterations : int
        number of iterations
    x0_selection : str
        x0 selection strategy
    sampler: str
        name of the sampler if x0_selection = 'new'

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of ExperimentData objects with concatenated results from the
        second optimizer run.
    """

    for index, experiment_data in enumerate(results):
        hyperparameters = {'seed': kwargs['seed'] + index + 1}
        experiment_data.optimize(optimizer=optimizer,
                                 data_generator=data_generator,
                                 iterations=n_iterations, kwargs=kwargs,
                                 hyperparameters=hyperparameters,
                                 x0_selection=x0_selection,
                                 sampler=sampler)
    return results


def _run_second_multiple_realizations_parallel(
    results: List[ExperimentData],
    optimizer: str,
    data_generator: str,
    kwargs: Dict[str, Any],
    n_iterations: int,
    x0_selection: str,
    sampler: str
) -> List[ExperimentData]:
    """
    Run multiple realizations of the second optimizer
    This operation is done in parallel.


    Parameters
    ----------
    results : List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer.
    optimizer : str
        Name of the second optimizer
    data_generator : str
        Name of the benchmark function
    kwargs : Dict[str, Any]
        Key-word arguments of the data generator
    n_iterations : int
        number of iterations
    x0_selection : str
        x0 selection strategy
    sampler: str
        name of the sampler if x0_selection = 'new'

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of ExperimentData objects with concatenated results from the
        second optimizer run.
    """

    def f(index: int, experiment_data: ExperimentData) -> ExperimentData:
        hyperparameters = {'seed': kwargs['seed'] + index + 1}
        experiment_data.optimize(optimizer=optimizer,
                                 data_generator=data_generator,
                                 iterations=n_iterations, kwargs=kwargs,
                                 hyperparameters=hyperparameters,
                                 x0_selection=x0_selection,
                                 sampler=sampler)
        return experiment_data

    args = [(index, experiment_data)
            for index, experiment_data in enumerate(results)]

    with mp.Pool() as pool:
        results = pool.starmap(f, args)

    return results

# =============================================================================


@lru_cache(maxsize=MAX_CACHE_SIZE)
def run_first_multiple_realizations(
    optimizer: str,
    data_generator: str,
    sampler: str,
    dimensionality: int,
    noise: float,
    offset: bool,
    realizations: int,
    n_samples: int,
    seed: int,
    n_iterations: int,
    x0_selection: str,
    parallelization: bool,
) -> List[ExperimentData]:
    """
    Run a single optimizer on a single function for multiple realizations.
    This operation is done in parallel

    Parameters
    ----------
    optimizer : str
        Name of the optimization algorithm
    data_generator : str
        Name of the benchmark function
    sampler : str
        Name of the sampler for the initial points
    dimensionality : int
        Number of input dimensions
    noise : float
        Standard deviation of the Gaussian noise
    offset : bool
        If True, the function is offset by a random value
    realizations : int
        Number of realizations
    n_samples : int
        number of initial samples
    seed : int
        Seed of the pseudo-random number generator
    n_iterations : int
        number of iterations
    x0_selection : str
        Name of the x0 selection strategy
    parallelization : bool
        If True, the realizations are run in parallel

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of Experiment data object containing the results
        of the optimization for multiple realizations.

    Note
    ----
    This function is cached with the LRU cache decorator.
    """

    # FIRST OPTIMIZERS
    if parallelization:
        results = _run_first_multiple_realizations_parallel(
            optimizer=optimizer, data_generator=data_generator,
            sampler=sampler, dimensionality=dimensionality, noise=noise,
            offset=offset,
            realizations=realizations, n_samples=n_samples, seed=seed,
            n_iterations=n_iterations, x0_selection=x0_selection)
    else:
        results = _run_first_multiple_realizations_sequential(
            optimizer=optimizer, data_generator=data_generator,
            sampler=sampler,
            dimensionality=dimensionality, noise=noise, offset=offset,
            realizations=realizations, n_samples=n_samples, seed=seed,
            n_iterations=n_iterations, x0_selection=x0_selection)

    logger.debug(f"Cached {optimizer} optimization for {n_iterations} "
                 f"iterations on {dimensionality}D {data_generator} function "
                 f"(noise={noise}, offset={offset}, seed={seed}) "
                 f"with {x0_selection} x0 selection. "
                 f"for {realizations} realizations.")

    return results


def _run_first_multiple_realizations_sequential(
    optimizer: str,
    data_generator: str,
    sampler: str,
    dimensionality: int,
    noise: float,
    offset: bool,
    realizations: int,
    n_samples: int,
    seed: int,
    n_iterations: int,
    x0_selection: str,
) -> List[ExperimentData]:
    """
    Run a single optimizer on a single function for multiple realizations.
    This operation is done sequentially

    Parameters
    ----------
    optimizer : str
        Name of the optimization algorithm
    data_generator : str
        Name of the benchmark function
    sampler : str
        Name of the sampler for the initial points
    dimensionality : int
        Number of input dimensions
    noise : float
        Standard deviation of the Gaussian noise
    offset : bool
        If True, the function is offset by a random value
    realizations : int
        Number of realizations
    n_samples : int
        number of initial samples
    seed : int
        Seed of the pseudo-random number generator
    n_iterations : int
        number of iterations
    x0_selection : str
        Name of the x0 selection strategy

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of Experiment data object containing the results
        of the optimization for multiple realizations.
    """

    return [_run_single_optimizer(optimizer, data_generator, sampler,
                                  dimensionality, noise, offset, seed + index,
                                  n_samples, n_iterations, x0_selection)
            for index in range(realizations)]


def _run_first_multiple_realizations_parallel(
    optimizer: str,
    data_generator: str,
    sampler: str,
    dimensionality: int,
    noise: float,
    offset: bool,
    realizations: int,
    n_samples: int,
    seed: int,
    n_iterations: int,
    x0_selection: str,
) -> List[ExperimentData]:
    """
    Run a single optimizer on a single function for multiple realizations.
    This operation is done in parallel

    Parameters
    ----------
    optimizer : str
        Name of the optimization algorithm
    data_generator : str
        Name of the benchmark function
    sampler : str
        Name of the sampler for the initial points
    dimensionality : int
        Number of input dimensions
    noise : float
        Standard deviation of the Gaussian noise
    offset : bool
        If True, the function is offset by a random value
    realizations : int
        Number of realizations
    n_samples : int
        number of initial samples
    seed : int
        Seed of the pseudo-random number generator
    n_iterations : int
        number of iterations
    x0_selection : str
        Name of the x0 selection strategy

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of Experiment data object containing the results
        of the optimization for multiple realizations.
    """

    args = [(optimizer, data_generator, sampler,
             dimensionality, noise, offset, seed + index,
             n_samples, n_iterations, x0_selection)
            for index in range(realizations)]

    with mp.Pool() as pool:
        results = pool.starmap(_run_single_optimizer, args)

    return results


def _run_single_optimizer(
        optimizer: str, data_generator: str, sampler: str, dimensionality: int,
        noise: float, offset: bool, seed: int, n_samples: int,
        n_iterations: int, x0_selection: str) -> ExperimentData:
    """
    Run a single optimizer on a single function.

    Parameters
    ----------
    optimizer : str
        Name of the optimization algorithm
    data_generator : str
        Name of the benchmark function
    sampler : str
        Name of the sampler for the initial points
    dimensionality : int
        Number of input dimensions
    noise : float
        Standard deviation of the Gaussian noise
    offset : bool
        If True, the function is offset by a random value
    seed : int
        Seed of the pseudo-random number generator
    n_samples : int
        number of initial samples
    n_iterations : int
        number of iterations
    x0_selection : str
        Name of the x0 selection strategy

    Returns
    -------
    ExperimentData
        Experiment data object containing the results of the optimization
    """

    domain = make_nd_continuous_domain(bounds=np.tile(
        [0., 1.], (dimensionality, 1)), dimensionality=dimensionality)

    kwargs = {'seed': seed,
              'scale_bounds': domain.get_bounds(),
              'noise': noise,
              'offset': offset}

    hyperparameters = {'seed': seed}

    if n_samples == 0:
        data = ExperimentData(domain=domain)
    else:
        data = ExperimentData.from_sampling(
            sampler=sampler, domain=domain, n_samples=n_samples, seed=seed)

        # Evaluate the initial samplers
        data.evaluate(data_generator=data_generator,
                      kwargs=kwargs, mode='sequential')

    data.optimize(
        optimizer=optimizer, data_generator=data_generator,
        kwargs=kwargs, hyperparameters=hyperparameters,
        iterations=n_iterations,
        x0_selection=x0_selection,
        sampler=sampler)

    logger.debug(f"Finished {optimizer} optimization for {n_iterations} "
                 f"iterations on {dimensionality}D {data_generator} function "
                 f"(noise={noise}, offset={offset}, seed={seed}) "
                 f"with {x0_selection} x0 selection.")

    return data


def experiment_two_optimizers(
    optimizers: List[str],
    data_generator: str,
    sampler: str,
    realizations: int,
    iterations: int,
    dimensionality: int,
    n_samples: int = 30,
    parallelization: bool = True,
    seed: Optional[int] = None,
    noise: float = 0.0,
    offset: bool = True,
) -> xr.Dataset:
    """
    Experiment where we find all combinations of two optimizers and run them\
    each for half of the total budget over many realizations.

    Parameters
    ----------
    optimizers : List[str]
        list of optimizer names to use
    data_generator : str
        name of the benchmark function
    sampler : str
        name of the sampling technique
    realizations : int
        number of realizations
    iterations : int
        total number of iterations
    dimensionality : int
        number of input dimensions of the problem
    n_samples : int, optional
        number of initial samples, by default 30
    parallelization : bool, optional
        if True, run the realizations in parallel, by default True
    seed : int, optional
        seed for the pseudo-random number generator, by default None
    noise : float, optional
        standard deviation of Gaussian noise on the objective, by default 0.0
    offset : bool, optional
        if True, the function is offset by a random vector, by default True

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the results of the experiment

    Notes
    -----
    Additional arguments of the benchmark function are stored in the xarray\
    Dataset as attributes.

    If no seed is given, a seed is randomly generator and stored between\
    0 and 1e5.
    """

    seed = seed if seed is not None else np.random.randint(0, 1e5)

    optimizer_schedulers = _get_all_combinations(optimizers, iterations)

    return xr.concat([run_multiple_realizations(
        optimizer_scheduler=optimizer_schedulers,
        data_generator=data_generator,
        sampler=sampler,
        realizations=realizations,
        dimensionality=dimensionality,
        noise=noise,
        offset=offset,
        n_samples=n_samples,
        parallelization=parallelization,
        seed=seed)
        for optimizer_schedulers in optimizer_schedulers], dim='optimizer')


# =============================================================================


def _get_handshake(optimizer_1: str, optimizer_2: str) -> str:
    """
    Return the handshake method to use between two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : str
        name of the optimizer to switch from

    Returns
    -------
    str
        method to determine the starting population of the second optimizer

    Notes
    -----
    The handshake method is either 'best', 'last' or 'random'
    """
    if optimizer_1.lower() == optimizer_2.lower():
        return 'last'

    if optimizer_2.lower() in ['lbfgsb', 'adam', 'randomsearch']:
        return 'best'

    if optimizer_1.lower() == 'randomsearch':
        return 'best'

    if optimizer_1.lower() == 'cmaes' and optimizer_2.lower() == 'pso':
        return 'last'

    if optimizer_1.lower() == 'pso' and optimizer_2.lower() == 'cmaes':
        return 'last'

    return 'random'


def _get_scheduler(
        optimizer_1: str, optimizer_2,
        iterations: int) -> OptimizerScheduler:
    """
    Get the scheduler for two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : _type_
        name of the second optimizer
    iterations : int
        total number of iterations across both optimizers

    Returns
    -------
    List[Tuple[str, int, str]]
        List of schedulers for the two optimizers
    """
    i_0 = (iterations // 2)
    i_1 = iterations // 2 + (iterations % 2)

    if optimizer_1 == optimizer_2:
        return [(optimizer_1, iterations, 'best')]

    return [(optimizer_1, i_0, 'new'),
            (optimizer_2, i_1, _get_handshake(optimizer_1, optimizer_2))]


def _get_all_combinations(
        optimizers: List[str], iterations: int) -> List[OptimizerScheduler]:
    """
    Get all combination of two optimizer names

    Parameters
    ----------
    optimizers : List[str]
        list of optimizer names
    iterations : int
        total number of iterations across both optimizers

    Returns
    -------
    List[OptimizerScheduler]
        List of optimizer schedulers
    """
    return [_get_scheduler(opt_1, opt_2, iterations)
            for opt_1, opt_2 in product(optimizers, repeat=2)]


def _get_name_from_scheduler(scheduler: OptimizerScheduler) -> str:
    """
    Create a represented name of the scheduler

    Parameters
    ----------
    scheduler : List[Tuple[str, int, str]]
        optimizer scheduler

    Returns
    -------
    str
        name joining the optimizer names with a plus ("+") sign
    """
    return "+".join(name[0] for name in scheduler)


def _listexperimentdata_to_dataset(
        results: List[ExperimentData]) -> xr.Dataset:
    """
    Convert a list of ExperimentData to an xarray Dataset

    Parameters
    ----------
    results : List[f3dasm.ExperimentData]
        List of optimization results in ExperimentData format

    Returns
    -------
    xr.Dataset
        xarray Dataset
    """
    return xr.concat([r.to_xarray() for r in results],
                     dim=xr.DataArray(np.arange(len(results)),
                                      dims='realization'))
