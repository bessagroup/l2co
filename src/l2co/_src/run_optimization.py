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
from __future__ import annotations

import gc
import logging
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
from l2co._src.history_state import HistoryState
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.run_state import RunState, batch_reset, batch_run
from l2co._src.task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


def run_optimization(
    optimizer: OptimizationStep,
    task: Task,
    sampler: Callable,
    bounded: tuple[float, float],
    n_iterations: int,
    n_realizations: int,
    *,
    key: PRNGKeyArray,
    verbose: bool,
) -> HistoryState:
    """
    Run a single optimization process for a given task.

    This function executes the main optimization loop, starting with an
    initial parameter sample and iteratively improving the model parameters
    using the specified optimizer.

    Parameters
    ----------
    optimizer : OptimizationStep
        The optimization step configuration containing optimizer type and
        hyperparameters.
    task : Task
        The optimization task containing the model and objective function.
    sampler : Callable
        Sampler function for generating initial parameters.
    bounded : tuple[float, float]
        Bounds for the optimization variables as (lower, upper).
    n_iterations : int
        Number of optimization iterations to perform.
    n_realizations : int
        Number of independent optimization realizations to run.
    key : PRNGKeyArray
        JAX random key for reproducibility.
    verbose : bool
        Whether to enable verbose logging during optimization.

    Returns
    -------
    HistoryState
        Complete history of the optimization process including parameters,
        losses, and update steps.
    """
    run_state = RunState.init(
        optimizer=optimizer,
        task=task,
        bounded=bounded,
        max_iterations=n_iterations,
        key=key,
    )

    run_state = batch_reset(
        run_state, task, sampler, jr.split(key, n_realizations)
    )

    return batch_run(
        run_state, task.loaded_dataset, n_iterations, key, verbose
    )


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

        history_state = run_optimization(
            optimizer=optimizer,
            task=task,
            sampler=self.sampler,
            bounded=self.bounded,
            n_iterations=n_iterations,
            n_realizations=self.n_realizations,
            key=key,
            verbose=self.verbose,
        )

        data_xarr = history_state.to_xarray_realizations(
            include_input_mean=self.save_input_parameters
        )

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
