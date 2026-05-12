"""``RolloutWrapper`` - JIT-friendly rollout entry point for a fixed run.

This wraps a ``RunState`` + ``BatchState`` so user code can call
``evaluate``, ``batch_evaluate``, or ``run`` without manually re-assembling
the optimizer/state tuple every time.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree
from l2co_tasks import Task

from l2co._src.batching import BatchState
from l2co._src.history_state import HistoryState
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.run_state import (
    RunState,
    batch_evaluate,
    evaluate,
    reset_from_stage0,
    run,
)

# =============================================================================


class RolloutWrapper(eqx.Module):
    """Wrapper for running optimization rollouts.

    Bundles a ``RunState``, ``BatchState``, and task
    configuration so that optimization can be launched
    via ``evaluate``, ``batch_evaluate``, or ``run``.
    """

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
        """Create a RolloutWrapper from an optimizer and task.

        Parameters
        ----------
        optimizer : OptimizationStep
            Optimizer configuration.
        task : Task
            Optimization task with model and loss function.
        sampler : Callable
            Parameter sampling function.
        key : PRNGKeyArray
            Random key for initialization.
        bounded : tuple[float, float], optional
            Parameter bounds as (lower, upper),
            by default (None, None).
        verbose : float, optional
            Whether to print progress, by default False.

        Returns
        -------
        RolloutWrapper
            Initialized rollout wrapper.
        """
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
        """Create a RolloutWrapper continuing from a stage-0 run.

        Re-uses the best parameters from ``previous_run_state``
        when switching optimizers between stages.

        Parameters
        ----------
        previous_run_state : RunState
            Run state from the completed stage-0 run.
        batch_state : BatchState
            Batch state from the completed stage-0 run.
        optimizer : OptimizationStep
            Stage-1 optimizer configuration.
        task : Task
            Optimization task.
        sampler : Callable
            Parameter sampling function.
        key : PRNGKeyArray
            Random key for initialization.
        switch : bool
            If True, reuse the previous run state as-is.
        bounded : tuple[float, float], optional
            Parameter bounds, by default (None, None).
        verbose : float, optional
            Whether to print progress, by default False.

        Returns
        -------
        RolloutWrapper
            Initialized rollout wrapper for stage 1.
        """
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
        """Run optimization across multiple realizations.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key, split for each realization.
        n_iterations : int
            Number of optimization iterations.
        n_realizations : int, optional
            Number of independent runs, by default 1.

        Returns
        -------
        tuple[RunState, BatchState, HistoryState]
            Final states for all realizations.
        """
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
        """Run a single optimization realization.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key.
        n_iterations : int
            Number of optimization iterations.

        Returns
        -------
        tuple[RunState, BatchState, HistoryState]
            Final run state, batch state, and history.
        """
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
        """Run optimization without JIT compilation.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key.
        n_iterations : int
            Number of optimization iterations.

        Returns
        -------
        tuple[RunState, BatchState, HistoryState]
            Final run state, batch state, and history.
        """
        return run(
            run_state=self.run_state,
            batch_state=self.batch_state,
            dataset=self.dataset,
            n_iterations=n_iterations,
            key=key,
            verbose=self.verbose,
        )
