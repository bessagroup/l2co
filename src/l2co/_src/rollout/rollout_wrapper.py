"""``RolloutWrapper`` - JIT-friendly rollout entry point for a fixed run.

This wraps a ``RunState`` + ``BatchState`` so user code can call
``batch_evaluate`` without manually re-assembling the optimizer/state
tuple every time.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import (
    BatchState,
    HistoryState,
    OptimizationStep,
    RunState,
    batch_evaluate,
)
from l2co_tasks import Task

# Local
from l2co._src.task_bridge import optimizer_mapping, task_bounds

# =============================================================================


class RolloutWrapper(eqx.Module):
    """Wrapper for running optimization rollouts.

    Bundles a ``RunState``, ``BatchState``, and task
    configuration so that optimization can be launched
    via ``batch_evaluate``.
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
        verbose : float, optional
            Whether to print progress, by default False.

        Returns
        -------
        RolloutWrapper
            Initialized rollout wrapper.
        """
        # Create the UpdateClass
        update_class = optimizer_mapping(optimizer.optimizer)(
            **optimizer.hyperparameters,
            task=task,
            opt_hash=optimizer.hash,
            bounded=task_bounds(task),
            stop_fn=optimizer.stopping_fn,
        )

        # Initialize the RunState
        run_state = RunState.init(
            update_class,
            model=task.model,
            dataset=task.loaded_dataset,
            batch_size=task.batch_size,
            key=key,
        )

        dataset = task.loaded_dataset

        # Initialize the BatchState
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
            pass_rng=task.pass_rng,
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
