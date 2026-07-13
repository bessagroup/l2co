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
import xarray as xr
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from l2co_tasks import Task

from l2co._src.databank.provenance import dumps_provenance
from l2co._src.models import L2COModel
from l2co._src.naming import (
    BUDGET,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    PROVENANCE,
    REALIZATION,
)
from l2co._src.optimizer_schedule import OptimizationStep

from .rollout_wrapper import RolloutWrapper

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

        if optimizer.optimizer == "l2co":
            model = L2COModel.load(
                stage0_model=optimizer.hyperparameters["stage0_model"],
                stage1_model=optimizer.hyperparameters["stage1_model"],
            )

            if optimizer.hyperparameters["swap_optimizers"] is not None:
                model = model.swap_optimizers(
                    optimizers=[
                        OptimizationStep(optimizer="sepcmaes"),
                        OptimizationStep(
                            optimizer="adam",
                            hyperparameters={"learning_rate": 0.03},
                        ),
                        OptimizationStep(optimizer="lbfgs"),
                        OptimizationStep(optimizer="pso"),
                    ]
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
        provenance: dict | None = None,
    ) -> ExperimentSample:
        """Run two-stage L2CO optimization.

        Executes stage 0 with the model-predicted initial
        optimizer, then stage 1 with the model-predicted
        follow-up optimizer, combining the resulting
        histories.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            Experiment sample to store results in.
        task : Task
            Optimization task.
        model : L2COModel
            Trained L2CO model for optimizer selection.
        t : float
            Fraction of total budget allocated to stage 0.
        provenance : dict | None, optional
            Policy-artifact provenance record (see
            :func:`l2co.databank.provenance_record`), attached to the
            stored loss history as the ``l2co.naming.PROVENANCE``
            attribute so it travels into the databank. ``None`` (the
            default) attaches nothing — such a run is not admissible
            to the databank.

        Returns
        -------
        ExperimentSample
            Updated experiment sample with combined history.
        """
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

            # ## HACK for now
            # if stage1_optimizer.optimizer == "adam":
            #     stage1_optimizer = OptimizationStep(
            #         optimizer="adam", hyperparameters={"learning_rate": 0.03}
            #     )

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

        if provenance is not None:
            data_xarr.attrs[PROVENANCE] = dumps_provenance(provenance)

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
