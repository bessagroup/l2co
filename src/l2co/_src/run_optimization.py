"""
Optimization execution module for L2CO optimization tasks.

This module provides functions and classes for executing optimization tasks
with configurable parameters, supporting both single and multiple realization
runs. It includes utilities for initializing optimizers, running optimization
loops, and managing optimization history.

The main components include:
- Functions for optimizer initialization and optimization execution
- Support for vectorized multiple realizations using JAX
- Data generator class for integration with f3dasm framework
- History tracking and result storage

Classes
-------
RunOptimizationSimple : DataGenerator
    Main class for executing optimization tasks with configurable parameters.

Functions
---------
initialize_optimizer : Initialize an optimizer with given parameters
run_optimization : Execute a single optimization process
run_optimization_many_realizations : Run multiple optimization realizations

Notes
-----
This module heavily relies on JAX for efficient computation and vectorization,
particularly for running multiple optimization realizations in parallel. The
implementation is designed to work with the f3dasm framework for experiment
management and data generation.
"""

#                                                                       Modules
# =============================================================================

# Standard
import gc
from collections.abc import Callable
from time import time

# Third-party
import equinox as eqx
import jax
import jax.random as jr
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from jaxtyping import PRNGKeyArray

# Local
from ._logging import logger
from .handshake import new
from .history_state import HistoryState
from .naming import INPUT_MEAN
from .optimizers import OPTIMIZER_MAPPING, OptHistory, UpdateClass
from .optimizerschedule import OptimizationStep
from .task import Task
from .update_state import UpdateState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def initialize_optimizer(
    optimizer: OptimizationStep,
    task: Task,
    bounded: tuple[float, float],
    *,
    key: PRNGKeyArray,
) -> tuple[UpdateState, UpdateClass]:
    """
    Initialize an optimizer with the given parameters.

    Parameters
    ----------
    optimizer : OptimizationStep
        The optimization step configuration containing optimizer type and
        hyperparameters.
    task : Task
        The optimization task containing the model to be optimized.
    bounded : Tuple[float, float]
        The bounds for the optimization variables as (lower, upper).
    key : PRNGKeyArray
        JAX random key for stochastic operations.

    Returns
    -------
    Tuple[UpdateState, UpdateClass]
        A tuple containing the initialized update state and update class.
    """

    update_class: UpdateClass = OPTIMIZER_MAPPING[optimizer.optimizer](
        **optimizer.hyperparameters,
        task=task,
        opt_hash=optimizer.hash,
        bounded=bounded,
        stop_fn=optimizer.stopping_fn,
    )

    update_state = UpdateState.init(
        params=eqx.filter(task.model, eqx.is_inexact_array),
        memory_size=update_class.popsize,
    )

    return update_state, update_class


def run_optimization(
    update_state: UpdateState,
    optimizer: UpdateClass,
    sampler: Callable,
    task: Task,
    n_iterations: int,
    key: PRNGKeyArray,
) -> HistoryState:
    """
    Run a single optimization process for a given task.

    This function executes the main optimization loop, starting with an
    initial parameter sample and iteratively improving the model parameters
    using the specified optimizer.

    Parameters
    ----------
    update_state : UpdateState
        The current state of the optimization containing parameters and
        memory.
    optimizer : UpdateClass
        The optimizer instance to use for parameter updates.
    sampler : Callable
        Function to generate initial parameter samples.
    task : Task
        The optimization task containing the model and objective function.
    n_iterations : int
        Number of optimization iterations to perform.
    key : PRNGKeyArray
        JAX random key for stochastic operations.

    Returns
    -------
    HistoryState
        Complete history of the optimization process including parameters,
        losses, and update steps.
    """

    # Initialize the optimizer state
    opt_state = optimizer.init(update_state=update_state, key=key)

    # Create the initial sample of model parameters
    update_state = new(
        update_state=update_state, task=task, sampler=sampler, key=key
    )

    # Initialize the history state
    history_state = HistoryState.init(
        n_iterations=n_iterations + 1,
        params=eqx.filter(task.model, eqx.is_inexact_array),
    )

    # Evaluate the initial parameters
    history_state = history_state.add(
        OptHistory.from_init_sample(
            update_state=update_state, task=task, key=key
        )
    )

    # Main optimization loop
    update_state, opt_state, history_state = optimizer.step(
        opt_state=opt_state,
        update_state=update_state,
        history_state=history_state,
        n_iterations=n_iterations,
        key=key,
    )
    return history_state


def run_optimization_many_realizations(
    update_state: UpdateState,
    optimizer: UpdateClass,
    sampler: Callable,
    task: Task,
    n_iterations: int,
    n_realizations: int,
    *,
    key: PRNGKeyArray,
) -> HistoryState:
    """
    Run optimization for multiple realizations using vectorized computation.

    This function runs the optimization process multiple times with different
    random seeds, enabling statistical analysis of the optimization
    performance across multiple independent runs.

    Parameters
    ----------
    update_state : UpdateState
        The current state of the optimization containing parameters and
        memory.
    optimizer : UpdateClass
        The optimizer instance to use for parameter updates.
    sampler : Callable
        Function to generate initial parameter samples.
    task : Task
        The optimization task containing the model and objective function.
    n_iterations : int
        Number of optimization iterations to perform.
    n_realizations : int
        Number of independent optimization runs to execute.
    key : PRNGKeyArray
        JAX random key for stochastic operations.

    Returns
    -------
    HistoryState
        Combined history state containing results from all realizations.

    Notes
    -----
    This function uses JAX's vectorization (`vmap`) to efficiently run
    multiple optimization realizations in parallel, splitting the random
    key to ensure each realization uses a different random seed.
    """

    return jax.vmap(
        run_optimization, in_axes=(None, None, None, None, None, 0)
    )(
        update_state,
        optimizer,
        sampler,
        task,
        n_iterations,
        jax.random.split(key, n_realizations),
    )


# =============================================================================


class RunOptimization(DataGenerator):
    """
    Data generator class for running optimization tasks.

    This class provides a framework for executing optimization tasks with
    configurable parameters including sampling strategies, budget functions,
    and multiple realizations for statistical analysis.

    Parameters
    ----------
    sampler : Callable
        Sampler function for generating initial parameters.
    budget_fn : Callable[[], int]
        Function to determine the optimization budget (number of iterations)
        for the task.
    seed : int
        Random seed for reproducibility across optimization runs.
    n_realizations : int, optional
        Number of independent optimization realizations to run, by default 1.
    bounded : Optional[Tuple[float, float]], optional
        Bounds for the optimization variables as (lower, upper), by default
        (None, None).
    clear_jax_cache : bool, optional
        Whether to clear the JAX cache after execution to free memory, by
        default True.
    verbose : bool, optional
        Whether to enable verbose logging during optimization, by default
        False.
    save_input_parameters : bool, optional
        Whether to save the history of input parameters in the output
        xarray, by default False.

    Methods
    -------
    execute(experiment_sample)
        Execute the optimization task for a given experiment sample.
    run_optimization(experiment_sample, task, optimizer)
        Run the optimization process with specified configuration.

    Examples
    --------
    >>> import jax.random as jr
    >>> from l2co._src.sampling import uniform_sampler
    >>>
    >>> # Create a simple optimization runner
    >>> runner = RunOptimizationSimple(
    ...     sampler=uniform_sampler,
    ...     budget_fn=lambda task: 100,
    ...     seed=42,
    ...     n_realizations=5
    ... )
    >>>
    >>> # Execute optimization on an experiment sample
    >>> result = runner.execute(experiment_sample)
    """

    def __init__(
        self,
        sampler: Callable,
        budget_fn: Callable[[], int],
        seed: int,
        n_realizations: int = 1,
        bounded: tuple[float, float] | None = (None, None),
        clear_jax_cache: bool = True,
        verbose: bool = False,
        save_input_parameters: bool = False,
    ):
        self.sampler = sampler
        self.budget_fn = budget_fn
        self.seed = seed
        self.n_realizations = n_realizations
        self.bounded = bounded
        self.clear_jax_cache = clear_jax_cache
        self.verbose = verbose
        self.save_input_parameters = save_input_parameters

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        """
        Execute the optimization task for a given experiment sample.

        This method extracts the task and optimizer schedule from the
        experiment sample and runs the optimization process, storing the
        results back into the experiment sample.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The experiment sample containing task and optimizer configuration
            in the input_data.

        Returns
        -------
        ExperimentSample
            The updated experiment sample with optimization results stored.

        Notes
        -----
        If clear_jax_cache is True, this method will clear JAX caches and
        call garbage collection after optimization to free memory.
        """
        es = experiment_sample.input_data
        task: Task = es["task"]
        optimizer: OptimizationStep = es["optimizer schedule"]

        experiment_sample = self.run_optimization(
            experiment_sample=experiment_sample,
            task=task,
            optimizer=optimizer,
        )

        if self.clear_jax_cache:
            eqx.clear_caches()
            jax.clear_caches()
            gc.collect()

        return experiment_sample

    def run_optimization(
        self,
        experiment_sample: ExperimentSample,
        task: Task,
        optimizer: OptimizationStep,
    ) -> ExperimentSample:
        """
        Run the optimization process with specified configuration.

        This method performs the core optimization logic including
        initializing the optimizer, running optimization iterations,
        and storing the results.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The experiment sample to store results in.
        task : Task
            The optimization task containing the model and objective function.
        optimizer : OptimizationStep
            The optimization step configuration containing optimizer type and
            hyperparameters.

        Returns
        -------
        ExperimentSample
            The updated experiment sample with optimization results stored
            as an xarray dataset.

        Notes
        -----
        The optimization history is stored as 'loss history' in the
        experiment sample's output data. If save_input_parameters is False,
        the input parameters are excluded from the stored data to save space.
        """
        t1 = time()

        key = jr.key(self.seed)

        n_iterations = self.budget_fn(task=task)

        update_state, update_class = initialize_optimizer(
            optimizer=optimizer, task=task, bounded=self.bounded, key=key
        )

        history_state: HistoryState = run_optimization_many_realizations(
            update_state=update_state,
            optimizer=update_class,
            sampler=self.sampler,
            task=task,
            n_iterations=n_iterations,
            n_realizations=self.n_realizations,
            key=key,
        )

        data_xarr = history_state.to_xarray_realizations()

        if not self.save_input_parameters:
            data_xarr = data_xarr.drop_vars(INPUT_MEAN)

        experiment_sample.store(
            name="loss history", object=data_xarr, to_disk=True
        )

        logger.info(
            f"Finished optimization task {task.name} ({task.dimensionality}D) "
            f"with {optimizer.name} for"
            f" {n_iterations} "
            f"iterations for {self.n_realizations} realizations"
            f" (took {time() - t1:.2f} seconds)"
        )

        return experiment_sample
