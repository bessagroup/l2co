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
from functools import lru_cache, wraps
from itertools import product
from multiprocessing import get_start_method, set_start_method
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from f3dasm._src.optimization.optimizer_factory import OPTIMIZER_MAPPING
from f3dasm.design import Domain, make_nd_continuous_domain
from pathos.helpers import mp

# Local
from l2co._src._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

set_start_method('forkserver')

#                                                                     Constants
# =============================================================================

MAX_CACHE_SIZE = 20
GRADIENT_BASED = [
    key for key, value in OPTIMIZER_MAPPING.items() if value.require_gradients
]

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
    results, histories, best_solutions = run_first_multiple_realizations(
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

    # SECOND OPTIMIZER
    domain = make_nd_continuous_domain(bounds=np.tile(
        [0., 1.], (dimensionality, 1)), dimensionality=dimensionality)

    kwargs = {'seed': seed,
              'scale_bounds': domain.get_bounds(),
              'noise': noise,
              'offset': offset}

    for optimizer, n_iterations, x0_selection in _optimizer_scheduler:
        results, histories, best_solutions = run_second_multiple_realizations(
            results=results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection,
            parallelization=parallelization,
            sampler=sampler,
            best_solutions=best_solutions,
            histories=histories,
        )

    logger.info(f"Optimized {dimensionality}D {data_generator} function with "
                f"{name}: "
                f"({perf_counter() - start_timer:.2f} s)")

    xarr_histories = _listexperimentdata_to_dataset(histories)

    # Add attributes
    xarr_histories.attrs['number_of_samples'] = n_samples
    xarr_histories.attrs['realization_seeds'] = [
        seed + index for index in range(realizations)]

    # Benchmark functions
    xarr_histories.attrs['function_seed'] = seed
    xarr_histories.attrs['function_name'] = data_generator
    xarr_histories.attrs['function_noise'] = noise
    xarr_histories.attrs['function_dimensionality'] = dimensionality

    # Global minimum function
    func = _datagenerator_factory(data_generator, domain=domain, kwargs=kwargs)
    _, g = func.get_global_minimum(d=dimensionality)
    xarr_histories.attrs['function_global_minimum'] = float(
        np.array(g if not isinstance(g, list) else g[0])[0, 0])

    # Add another dimension with coordinate to xarr
    xarr_histories = xarr_histories.assign_coords({'optimizer': [name]})

    return xarr_histories


def run_multiple_realizations_continuing_from_file(
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
    best_solutions: xr.Dataset,
    histories: xr.Dataset,
    results: xr.Dataset,
) -> xr.Dataset:

    start_timer = perf_counter()

    name = _get_name_from_scheduler(optimizer_scheduler)

    _optimizer_scheduler = deepcopy(optimizer_scheduler)

    first_optimizer, first_iterations, _ = _optimizer_scheduler.pop(0)

    domain = make_nd_continuous_domain(bounds=np.tile(
        [0., 1.], (dimensionality, 1)), dimensionality=dimensionality)

    domain.add_output(name='y', to_disk=False, exist_ok=True)

    # open files and convert to List[ExperimentData]
    list_results = _dataset_to_listexperimentdata(
        dataset=results, domain=domain)

    list_best_solutions = _dataset_to_listexperimentdata(
        dataset=best_solutions, domain=domain)

    list_histories = _dataset_to_listexperimentdata(
        dataset=histories, domain=domain, has_input=False)

    # SECOND OPTIMIZER

    kwargs = {'seed': seed,
              'scale_bounds': domain.get_bounds(),
              'noise': noise,
              'offset': offset}

    for optimizer, n_iterations, x0_selection in _optimizer_scheduler:
        (list_results, list_histories,
         list_best_solutions) = run_second_multiple_realizations(
            results=list_results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection,
            parallelization=parallelization,
            sampler=sampler,
            best_solutions=list_best_solutions,
            histories=list_histories,
        )

    logger.info(f"Optimized {dimensionality}D {data_generator} function with "
                f"{name}: "
                f"({perf_counter() - start_timer:.2f} s)")

    xarr_histories = _listexperimentdata_to_dataset(list_histories)
    xarr_results = _listexperimentdata_to_dataset(list_results)
    xarr_best_solutions = _listexperimentdata_to_dataset(list_best_solutions)

    # Add attributes
    xarr_histories.attrs['number_of_samples'] = n_samples
    xarr_histories.attrs['realization_seeds'] = [
        seed + index for index in range(realizations)]

    # Benchmark functions
    xarr_histories.attrs['function_seed'] = seed
    xarr_histories.attrs['function_name'] = data_generator
    xarr_histories.attrs['function_noise'] = noise
    xarr_histories.attrs['function_dimensionality'] = dimensionality

    # Global minimum function
    func = _datagenerator_factory(data_generator, domain=domain, kwargs=kwargs)
    _, g = func.get_global_minimum(d=dimensionality)
    xarr_histories.attrs['function_global_minimum'] = float(
        np.array(g if not isinstance(g, list) else g[0])[0, 0])

    # Add another dimension with coordinate to xarr
    xarr_histories = xarr_histories.assign_coords({'optimizer': [name]})

    return xarr_results, xarr_histories, xarr_best_solutions


def run_second_multiple_realizations(
    results: List[ExperimentData],
    optimizer: str,
    data_generator: str,
    kwargs: Dict[str, Any],
    n_iterations: int,
    x0_selection: str,
    parallelization: bool,
    sampler: str,
    histories: List[ExperimentData],
    best_solutions: List[ExperimentData],
) -> Tuple[List[ExperimentData], List[ExperimentData], List[ExperimentData]]:
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
    sampler: str
        name of the sampler if x0_selection = 'new'
    histories : List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer.
    best_solutions : List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer containing
        the best solution per realization.

    Returns
    -------
    Tuple[List[ExperimentData], List[ExperimentData], List[ExperimentData]]
        Tuple of lists of the ExperimentData objects containing the results
        of the optimizations, the histories and the best solutions.
    """

    if parallelization:
        _results = _run_second_multiple_realizations_parallel(
            results=results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection, sampler=sampler,
            histories=histories, best_solutions=best_solutions
        )
    else:
        _results = _run_second_multiple_realizations_sequential(
            results=results, optimizer=optimizer,
            data_generator=data_generator,
            kwargs=kwargs, n_iterations=n_iterations,
            x0_selection=x0_selection, sampler=sampler,
            histories=histories, best_solutions=best_solutions
        )

    data, history, best_solutions = list(zip(*_results))

    logger.debug(f"(2/2) Created {optimizer} optimization for {n_iterations} "
                 f"iterations on {data_generator} function ")

    return data, history, best_solutions


def _run_second_optimizer(data: ExperimentData, index: int,
                          optimizer: str, data_generator: str, sampler: str,
                          n_iterations: int, kwargs: dict, x0_selection: str,
                          best_solution: ExperimentData,
                          history: ExperimentData) -> List[ExperimentData]:
    """
    Run the second optimizer

    Parameters
    ----------
    data : f3dasm.ExperimentData
        ExperimentData object from the first optimizer.
    index : int
        index of the realization
    optimizer : str
        Name of the second optimizer
    data_generator : str
        Name of the benchmark function
    sampler: str
        name of the sampler if x0_selection = 'new'
    n_iterations : int
        number of iterations
    kwargs : Dict[str, Any]
        Key-word arguments of the data generator
    x0_selection : str
        x0 selection strategy
    best_solution : f3dasm.ExperimentData
        ExperimentData object containing the best solution from the first
        optimizer
    history : f3dasm.ExperimentData
        ExperimentData object containing the history of the first optimizer

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of ExperimentData objects with concatenated results from the
        second optimizer run.
    """

    hyperparameters = {'seed': kwargs['seed'] + index + 1}
    n_before = len(history)

    if x0_selection == 'best':
        _x0_selection = deepcopy(best_solution)

    else:
        _x0_selection = x0_selection

    def call(data: ExperimentData):
        nonlocal history
        nonlocal best_solution
        history += data.get_output_data()
        # history += data

        trial_solution = data.get_n_best_output(1)

        _, y_best = best_solution.to_numpy()
        _, y_trial = trial_solution.to_numpy()

        if y_best.size == 0:
            best_solution = trial_solution
            return

        if y_trial < y_best:
            best_solution = trial_solution

    data.optimize(optimizer=optimizer,
                  data_generator=data_generator,
                  iterations=n_iterations, kwargs=kwargs,
                  hyperparameters=hyperparameters,
                  x0_selection=_x0_selection,
                  sampler=sampler,
                  overwrite=True,
                  callback=call)

    best_solution._reset_index()
    history._reset_index()

    while len(history) < n_iterations + n_before:
        history += best_solution.get_output_data()

    history = history.select(range(n_before + n_iterations))

    return [data, history, best_solution]


def _run_second_multiple_realizations_sequential(
    results: List[ExperimentData],
    optimizer: str,
    data_generator: str,
    kwargs: Dict[str, Any],
    n_iterations: int,
    x0_selection: str,
    sampler: str,
    histories: List[ExperimentData],
    best_solutions: List[ExperimentData],
) -> List[List[ExperimentData]]:
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
    histories: List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer.
    best_solutions: List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer containing
        the best solution per realization.

    Returns
    -------
    List[List[ExperimentData]]
        List of three ExperimentData lists: the results, the histories and the
        best solutions.
    """

    return [
        _run_second_optimizer(
            data=experiment_data,
            index=index,
            optimizer=optimizer,
            data_generator=data_generator,
            sampler=sampler,
            n_iterations=n_iterations,
            kwargs=kwargs,
            x0_selection=x0_selection,
            best_solution=best_solution,
            history=history_first
        ) for index, (experiment_data, history_first, best_solution
                      ) in enumerate(
            zip(results, histories, best_solutions))
    ]


def _run_second_multiple_realizations_parallel(
    results: List[ExperimentData],
    optimizer: str,
    data_generator: str,
    kwargs: Dict[str, Any],
    n_iterations: int,
    x0_selection: str,
    sampler: str,
    histories: List[ExperimentData],
    best_solutions: List[ExperimentData],
) -> List[List[ExperimentData]]:
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
    histories: List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer.
    best_solutions: List[f3dasm.ExperimentData]
        List of ExperimentData objects from the first optimizer containing
        the best solution per realization.

    Returns
    -------
    List[List[ExperimentData]]
        List of three ExperimentData lists: the results, the histories and the
        best solutions.
    """

    def f(index: int, experiment_data: ExperimentData,
          history_first: ExperimentData, best_solution: ExperimentData
          ) -> ExperimentData:

        return _run_second_optimizer(
            data=experiment_data,
            index=index,
            optimizer=optimizer,
            data_generator=data_generator,
            sampler=sampler,
            n_iterations=n_iterations,
            kwargs=kwargs,
            x0_selection=x0_selection,
            best_solution=best_solution,
            history=history_first
        )

    args = [
        (index, experiment_data, history_first, best_solution)
        for index, (experiment_data, history_first, best_solution
                    ) in enumerate(zip(results, histories, best_solutions))
    ]

    with mp.Pool() as pool:
        results = pool.starmap(f, args)

    return results

# =============================================================================


@lru_cache(maxsize=MAX_CACHE_SIZE)
def _run_first_multiple_realizations(
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
) -> Tuple[List[ExperimentData], List[ExperimentData], List[ExperimentData]]:
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
    Tuple[List[ExperimentData], List[ExperimentData], List[ExperimentData]]
        Tuple of lists of the ExperimentData objects containing the results
        of the optimizations, the histories and the best solutions.

    Note
    ----
    This function is cached with the LRU cache decorator.
    """

    # FIRST OPTIMIZERS
    if parallelization:
        _results = _run_first_multiple_realizations_parallel(
            optimizer=optimizer, data_generator=data_generator,
            sampler=sampler, dimensionality=dimensionality, noise=noise,
            offset=offset,
            realizations=realizations, n_samples=n_samples, seed=seed,
            n_iterations=n_iterations, x0_selection=x0_selection)
    else:
        _results = _run_first_multiple_realizations_sequential(
            optimizer=optimizer, data_generator=data_generator,
            sampler=sampler,
            dimensionality=dimensionality, noise=noise, offset=offset,
            realizations=realizations, n_samples=n_samples, seed=seed,
            n_iterations=n_iterations, x0_selection=x0_selection)

    data, history, best_solutions = list(zip(*_results))

    logger.debug(f"(1/2) Created {optimizer} optimization for {n_iterations} "
                 f"iterations on {dimensionality}D {data_generator} function "
                 f"(noise={noise}, offset={offset}, seed={seed}) "
                 f"with {x0_selection} x0 selection. "
                 f"for {realizations} realizations.")

    return data, history, best_solutions


def cached_function_with_logging(*args, **kwargs):
    hits_before = _run_first_multiple_realizations.cache_info().hits
    result = _run_first_multiple_realizations(*args, **kwargs)
    hits_after = _run_first_multiple_realizations.cache_info().hits
    if hits_after > hits_before:
        logger.debug(f"(1/2) Used cached result of {kwargs['optimizer']} "
                     f"optimization for {kwargs['n_iterations']} "
                     f"iterations on {kwargs['dimensionality']}D "
                     f"{kwargs['data_generator']} function "
                     f"(noise={kwargs['noise']}, offset={kwargs['offset']}, "
                     f"seed={kwargs['seed']}) "
                     f"with {kwargs['x0_selection']} x0 selection. "
                     f"for {kwargs['realizations']} realizations.")

    return result


run_first_multiple_realizations = wraps(
    _run_first_multiple_realizations)(cached_function_with_logging)


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
) -> List[List[ExperimentData]]:
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
    List[List[ExperimentData]]
        List of three ExperimentData lists: the results, the histories and the
        best solutions.
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
) -> List[List[ExperimentData]]:
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
    List[List[ExperimentData]]
        List of three ExperimentData lists: the results, the histories and the
        best solutions.
    """

    args = [(optimizer, data_generator, sampler,
             dimensionality, noise, offset, seed + index,
             n_samples, n_iterations, x0_selection)
            for index in range(realizations)]

    logger.debug(f"Current start method: {get_start_method()}")

    with mp.Pool() as pool:
        results = pool.starmap(_run_single_optimizer, args)

    return results


def _run_single_optimizer(
        optimizer: str, data_generator: str, sampler: str, dimensionality: int,
        noise: float, offset: bool, seed: int, n_samples: int,
        n_iterations: int, x0_selection: str
) -> List[ExperimentData]:
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
    List[ExperimentData]
        List of three ExperimentData objects: the results, the histories
        and the best solutions.
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

    experimentdata_callback = data.get_output_data()
    best_solution = data.get_n_best_output(1)

    def call(data: ExperimentData):
        nonlocal best_solution
        nonlocal experimentdata_callback
        experimentdata_callback += data.get_output_data()
        # experimentdata_callback += data

        trial_solution = data.get_n_best_output(1)

        _, y_best = best_solution.to_numpy()
        _, y_trial = trial_solution.to_numpy()

        if y_best.size == 0:
            best_solution = trial_solution
            return

        if y_trial < y_best:
            best_solution = trial_solution

    data.optimize(
        optimizer=optimizer, data_generator=data_generator,
        kwargs=kwargs, hyperparameters=hyperparameters,
        iterations=n_iterations,
        x0_selection=x0_selection,
        sampler=sampler,
        overwrite=True,
        callback=call)

    best_solution._reset_index()

    while len(experimentdata_callback) < n_iterations + n_samples:
        experimentdata_callback += best_solution.get_output_data()

    experimentdata_callback._reset_index()

    # Cutoff overiterations
    experimentdata_callback = experimentdata_callback.select(
        range(n_iterations + n_samples))

    return [data, experimentdata_callback, best_solution]


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
    The handshake method is either 'best', 'last', 'new' or 'random'
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

    return 'new'


def _get_scheduler(
        optimizer_1: str, optimizer_2: str, switch: float,
        iterations: int, first_x0: str = 'new') -> OptimizerScheduler:
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
    switch : float, optional
        fraction of the iterations to use for the first optimizer
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'

    Returns
    -------
    List[Tuple[str, int, str]]
        List of schedulers for the two optimizers
    """

    i_0 = int(iterations // (1/switch))
    i_1 = iterations - i_0

    if optimizer_1 == optimizer_2:
        return [(optimizer_1, iterations, first_x0)]

    return [(optimizer_1, i_0, first_x0),
            (optimizer_2, i_1, _get_handshake(optimizer_1, optimizer_2))]


def _get_all_combinations(
        optimizers: List[str], iterations: int,
        first_x0: str = 'new',
        switch: float = 0.5) -> List[OptimizerScheduler]:
    """
    Get all combination of two optimizer names

    Parameters
    ----------
    optimizers : List[str]
        list of optimizer names
    iterations : int
        total number of iterations across both optimizers
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'
    switch : float, optional
        fraction of the iterations to use for the first
        optimizer, by default 0.5

    Returns
    -------
    List[OptimizerScheduler]
        List of optimizer schedulers
    """
    return [_get_scheduler(
        optimizer_1=opt_1, optimizer_2=opt_2,
        iterations=iterations, first_x0=first_x0,
        switch=switch)
        for opt_1, opt_2 in product(optimizers, repeat=2)]


def _get_all_combinations_no_switch(
        optimizers: List[str], iterations: int,
        first_x0: str = 'new') -> List[OptimizerScheduler]:
    """
    Get all combination of two optimizer names

    Parameters
    ----------
    optimizers : List[str]
        list of optimizer names
    iterations : int
        total number of iterations
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'

    Returns
    -------
    List[OptimizerScheduler]
        List of optimizer schedulers
    """
    return [[(opt, iterations, first_x0)] for opt in optimizers]


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


def _dataset_to_listexperimentdata(
        dataset: xr.Dataset,
        domain: Optional[Domain] = None,
        has_input: bool = True) -> List[ExperimentData]:
    """
    Convert an xarray Dataset to a list of ExperimentData

    Parameters
    ----------
    dataset : xr.Dataset
        xarray Dataset

    Returns
    -------
    List[f3dasm.ExperimentData]
        List of optimization results in ExperimentData format
    """
    if has_input:
        return [ExperimentData(
            input_data=dataset.isel(realization=r).input.to_pandas(),
            output_data=dataset.isel(
                realization=r).output.to_pandas(),
            domain=domain) for r in dataset.realization
        ]
    else:
        return [ExperimentData(
            output_data=dataset.isel(
                realization=r).output.to_pandas(),
        ) for r in dataset.realization
        ]


# =============================================================================

def needs_gradient(optimizer_scheduler: OptimizerScheduler) -> bool:
    """
    Check if the optimizer scheduler needs gradient information

    Parameters
    ----------
    optimizer_scheduler : List[Tuple[str, int, str]]
        optimizer scheduler

    Returns
    -------
    bool
        True if the optimizer scheduler contains a gradient-based optimizer
    """
    return any(
        name.lower() in GRADIENT_BASED for name, _, _ in optimizer_scheduler)
