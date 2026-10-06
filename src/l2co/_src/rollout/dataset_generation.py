"""``RunOptimization`` - the ``DataGenerator`` that executes the algorithm
portfolio on a benchmark task and collects RTD trajectories.

Used exclusively from the ``experiments/create_optimization_dataset/main.py``
Hydra pipeline; not part of the inference-time API.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from time import time

import equinox as eqx
import jax
import jax.random as jr
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from l2co_optimizers import OptimizationStep
from l2co_tasks import Task

from l2co._src.databank.provenance import dumps_provenance
from l2co._src.history_state import history_to_xarray_realizations
from l2co._src.naming import PROVENANCE
from l2co._src.rollout.rollout_wrapper import RolloutWrapper

logger = logging.getLogger(__name__)

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

        # No dispatch on the optimizer name: ``"l2co"`` resolves through
        # the ordinary optimizer registry to the strategy-wrapper bridge
        # (``L2COUpdateClass.l2co_population_update`` — see
        # ``docs/adr/0002`` and ``0007``), so the L2CO strategy runs
        # through the same path as any plain optimizer.
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
        provenance: dict | None = None,
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
        provenance : dict | None, optional
            Policy-artifact provenance record (see
            :func:`l2co.databank.provenance_record`), attached to the
            stored loss history as the ``l2co.naming.PROVENANCE``
            attribute so it travels into the databank. Only meaningful
            for learned optimizers (e.g. ``"l2co"``, which runs through
            this generic path via the optimizer registry — see
            ``docs/adr/0002``). ``None`` (the default) attaches
            nothing — such a run is not admissible to the databank.

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

        data_xarr = history_to_xarray_realizations(history_state)

        if provenance is not None:
            data_xarr.attrs[PROVENANCE] = dumps_provenance(provenance)

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
