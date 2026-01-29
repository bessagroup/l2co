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
import xarray as xr
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from jaxtyping import PRNGKeyArray, PyTree

from l2co._src.batching import BatchState

# Local
from l2co._src.history_state import HistoryState
from l2co._src.models import L2COModel
from l2co._src.naming import BUDGET, OUTPUT_MEAN, OUTPUT_MIN, REALIZATION
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.run_state import (
    RunState,
    batch_evaluate,
    evaluate,
    reset_from_stage0,
    run,
)
from l2co._src.task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


class RolloutWrapper(eqx.Module):
    run_state: RunState
    batch_state: BatchState
    static: PyTree
    dataset: dict[str, jax.Array]
    loss_fn: Callable
    sampler: Callable
    pass_rng: bool
    verbose: bool

    @classmethod
    def init(
        cls,
        optimizer: OptimizationStep,
        task: Task,
        sampler: Callable,
        key: PRNGKeyArray,
        bounded: tuple[float, float] = (None, None),
        verbose: float = False,
    ) -> RolloutWrapper:
        run_state = RunState.init(
            optimizer=optimizer,
            task=task,
            bounded=bounded,
            key=key,
        )

        pass_rng = task.pass_rng

        dataset = task.loaded_dataset

        batch_state = BatchState.init(
            dataset=dataset,
            batch_size=task.batch_size,
            key=key,
        )

        _, static = eqx.partition(task.model, eqx.is_inexact_array)

        return cls(
            run_state=run_state,
            batch_state=batch_state,
            static=static,
            dataset=dataset,
            loss_fn=task.loss_fn,
            sampler=sampler,
            pass_rng=pass_rng,
            verbose=verbose,
        )

    @classmethod
    def init_from_stage0(
        cls,
        previous_run_state: RunState,
        batch_state: BatchState,
        optimizer: OptimizationStep,
        task: Task,
        sampler: Callable,
        key: PRNGKeyArray,
        switch: bool,
        bounded: tuple[float, float] = (None, None),
        verbose: float = False,
    ) -> RolloutWrapper:
        dataset = task.loaded_dataset
        _, static = eqx.partition(task.model, eqx.is_inexact_array)
        pass_rng = task.pass_rng

        if switch:
            run_state = previous_run_state
        else:
            # Initialize new run state
            run_state = RunState.init(
                optimizer=optimizer,
                task=task,
                bounded=bounded,
                key=key,
            )

            run_state = reset_from_stage0(
                run_state=run_state,
                best_params=previous_run_state.best_params,
                batch_state=batch_state,
                static=static,
                dataset=dataset,
                loss_fn=task.loss_fn,
                pass_rng=pass_rng,
                sampler=sampler,
                key=key,
            )

        # Replace one of run_state params with best_params

        return cls(
            run_state=run_state,
            batch_state=batch_state,
            static=static,
            dataset=dataset,
            loss_fn=task.loss_fn,
            sampler=sampler,
            pass_rng=pass_rng,
            verbose=verbose,
        )

    @eqx.filter_jit
    def batch_evaluate(
        self,
        key: PRNGKeyArray,
        n_iterations: int,
        n_realizations: int = 1,
    ) -> tuple[RunState, BatchState, HistoryState]:
        return batch_evaluate(
            self.run_state,
            self.batch_state,
            self.static,
            self.dataset,
            self.loss_fn,
            self.sampler,
            n_iterations,
            self.pass_rng,
            jr.split(key, n_realizations),
            self.verbose,
        )

    def evaluate(
        self,
        key,
        n_iterations: int,
    ) -> tuple[RunState, BatchState, HistoryState]:
        return evaluate(
            self.run_state,
            self.batch_state,
            self.static,
            self.dataset,
            self.loss_fn,
            self.sampler,
            n_iterations,
            self.pass_rng,
            key,
            self.verbose,
        )

    def run(
        self,
        key: PRNGKeyArray,
        n_iterations: int,
    ) -> tuple[RunState, BatchState, HistoryState]:
        return run(
            run_state=self.run_state,
            batch_state=self.batch_state,
            dataset=self.dataset,
            n_iterations=n_iterations,
            key=key,
            verbose=self.verbose,
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

    def execute(
        self,
        experiment_sample: ExperimentSample,
    ) -> ExperimentSample:
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

        if optimizer.optimizer == "l2co":
            model = L2COModel.load(
                stage0_model=optimizer.hyperparameters["stage0_model"],
                stage1_model=optimizer.hyperparameters["stage1_model"],
            )

            experiment_sample = self.run_optimization_l2co(
                experiment_sample=experiment_sample,
                task=task,
                model=model,
                t=optimizer.hyperparameters["t"],
            )
        else:
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

    def run_optimization_l2co(
        self,
        experiment_sample: ExperimentSample,
        task: Task,
        model: L2COModel,
        t: float,
    ) -> ExperimentSample:
        t1 = time()

        key = jr.key(self.seed)
        n_iterations = self.budget_fn(task=task)

        # Calculate split for stage 0
        n_iterations_stage0 = int(round(n_iterations * t))

        # Ensure stage 0 doesn't gobble everything or nothing
        # if n_iterations > 0
        n_iterations_stage0 = max(min(n_iterations_stage0, n_iterations), 0)

        # The remainder goes to stage 1
        n_iterations_stage1 = n_iterations - n_iterations_stage0

        stage0_optimizer: OptimizationStep = model.predict_stage0(
            task_features=task.tag,
        )

        # First run stage0 optimizer
        rollout_wrapper: RolloutWrapper = RolloutWrapper.init(
            optimizer=stage0_optimizer,
            task=task,
            sampler=self.sampler,
            key=key,
            bounded=self.bounded,
            verbose=self.verbose,
        )
        # Stage0
        stage0 = rollout_wrapper.batch_evaluate(
            key=key,
            n_iterations=n_iterations_stage0,
            n_realizations=self.n_realizations,
        )

        stage0_individual = [
            jax.tree_util.tree_map(lambda x, i=i: x[i], stage0)
            for i in range(self.n_realizations)
        ]

        keys = jr.split(key, num=self.n_realizations)
        histories = []
        # for k in keys:
        for (run_state_stage0, batch_state, history_state_stage0), k in zip(
            stage0_individual,
            keys,
            strict=True,
        ):
            data_xarr_stage0 = history_state_stage0.to_xarray()

            stage1_optimizer: OptimizationStep = model.predict_stage1(
                task_features=task.tag,
                optimizer_features={"optimizer": stage0_optimizer.name},
                x_output_mean=data_xarr_stage0[OUTPUT_MEAN].values.flatten(),
                x_output_min=data_xarr_stage0[OUTPUT_MIN].values.flatten(),
                global_min=task.global_min,
            )

            # Stage 1
            _, k1 = jr.split(k)
            run_state_stage1: RolloutWrapper = RolloutWrapper.init_from_stage0(
                previous_run_state=run_state_stage0,
                batch_state=batch_state,
                optimizer=stage1_optimizer,
                task=task,
                sampler=self.sampler,
                key=k1,
                switch=(stage0_optimizer == stage1_optimizer),
                bounded=self.bounded,
                verbose=self.verbose,
            )

            _, _, history_state_stage1 = run_state_stage1.run(
                key=k1, n_iterations=n_iterations_stage1
            )

            data_xarr_stage1 = history_state_stage1.to_xarray()

            combined_xarr = xr.concat(
                [
                    data_xarr_stage0,
                    data_xarr_stage1,
                ],
                dim=BUDGET,
            )

            histories.append(combined_xarr)

        data_xarr = xr.concat(
            histories,
            dim=xr.DataArray(
                range(self.n_realizations),
                dims=[REALIZATION],
            ),
        )

        experiment_sample.store(
            name="loss history", object=data_xarr, to_disk=True
        )

        logger.info(
            f"Finished optimization task {task.name} ({task.dimensionality}D) "
            f"with L2CO model for"
            f" {n_iterations} "
            f"iterations for {self.n_realizations} realizations. "
            f"Optimizers used: ({stage0_optimizer.name} -> "
            f"{stage1_optimizer.name})"
            f" (took {time() - t1:.2f} seconds)"
        )

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

        rollout_wrapper: RolloutWrapper = RolloutWrapper.init(
            optimizer=optimizer,
            task=task,
            sampler=self.sampler,
            key=key,
            bounded=self.bounded,
            verbose=self.verbose,
        )

        _, _, history_state = rollout_wrapper.batch_evaluate(
            key=key,
            n_iterations=n_iterations,
            n_realizations=self.n_realizations,
        )

        data_xarr = history_state.to_xarray_realizations()

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
