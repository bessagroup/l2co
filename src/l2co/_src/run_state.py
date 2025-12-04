#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

# Local
from .batching import BatchState
from .history_state import HistoryState
from .optimizers import optimizer_mapping
from .optimizerschedule import OptimizationStep
from .task import Task
from .typing import InputParameters, SamplerFunction
from .update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class RunState(eqx.Module):
    """
    State container for running optimization processes.
    This class encapsulates the state required to run optimization processes,
    including the update state, history state, optimizer state, batch state,
    and update class. It provides methods for initializing and running the
    optimization process in a batched manner for multiple realizations.

    Parameters
    ----------
    params : InputParameters
        The current input parameters being optimized, shaped as (popsize, dim).
    opt_state : PyTree
        The state of the optimizer.
    batch_state : BatchState
        The state of the data batching process.
    update_class : UpdateClass
        The class defining the optimizer update logic.
    """

    params: InputParameters
    opt_state: PyTree
    batch_state: BatchState
    update_class: UpdateClass = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        optimizer: OptimizationStep,
        task: Task,
        bounded: tuple[float, float],
        key: PRNGKeyArray,
    ) -> RunState:
        """Initialize a new RunState.

        Parameters
        ----------
        optimizer : OptimizationStep
            Optimizer configuration.
        task : Task
            Task to optimize.
        bounded : tuple[float, float]
            Bounds for parameter values (lower, upper).
        key : PRNGKeyArray
            PRNG key for random operations.

        Returns
        -------
        RunState
            Initialized run state.
        """
        dataset = task.loaded_dataset

        update_class: UpdateClass = optimizer_mapping(optimizer.optimizer)(
            **optimizer.hyperparameters,
            task=task,
            opt_hash=optimizer.hash,
            bounded=bounded,
            stop_fn=optimizer.stopping_fn,
        )

        # Initialize batch state
        batch_state = BatchState.init(
            dataset=dataset,
            batch_size=task.batch_size,
            key=key,
        )

        params = jax.tree.map(
            lambda x: jnp.repeat(x[None, ...], update_class.popsize, axis=0),
            eqx.filter(task.model, eqx.is_inexact_array),
        )
        # Initialize the optimizer state
        opt_state = update_class.init(
            params=params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
        )

        return cls(
            params=params,
            opt_state=opt_state,
            batch_state=batch_state,
            update_class=update_class,
        )


def reset(
    run_state: RunState,
    task: Task,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> RunState:
    """Reset the run state with new samples.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    task : Task
        Task to optimize.
    sampler : SamplerFunction
        Sampling function for generating new parameters.
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    RunState
        Reset run state with new samples.
    """
    batch_state = run_state.batch_state.reset(key=key)

    dataset = task.loaded_dataset

    params = sampler(
        key,
        eqx.filter(task.model, eqx.is_inexact_array),
        run_state.update_class.popsize,
    )

    opt_state = run_state.update_class.init(
        params=params,
        batch_state=batch_state,
        dataset=dataset,
        key=key,
    )

    return RunState(
        params=params,
        opt_state=opt_state,
        batch_state=batch_state,
        update_class=run_state.update_class,
    )


def run(
    run_state: RunState,
    dataset: dict[str, jax.Array],
    n_iterations: int,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, HistoryState]:
    """Run optimization for a specified number of iterations.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    n_iterations : int
        Number of iterations to run.
    key : PRNGKeyArray
        PRNG key for random operations.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[RunState, HistoryState]
        Updated RunState and history state after optimization.
    """
    # Main optimization loop
    params, opt_state, batch_state, history_state = (
        run_state.update_class.step(
            opt_state=run_state.opt_state,
            params=run_state.params,
            batch_state=run_state.batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=verbose,
        )
    )

    run_state = RunState(
        params=params,
        opt_state=opt_state,
        batch_state=batch_state,
        update_class=run_state.update_class,
    )

    return run_state, history_state


def evaluate(
    run_state: RunState,
    task: Task,
    sampler: Callable,
    n_iterations: int,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, HistoryState]:
    """Evaluate optimization by resetting and running to completion.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    task : Task
        Task to optimize.
    sampler : Callable
        Sampling function for generating new parameters.
    n_iterations : int
        Number of iterations to run.
    key : PRNGKeyArray
        PRNG key for random operations.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[RunState, HistoryState]
        Tuple of updated RunState and History state after complete optimization
        run.
    """
    run_state = reset(run_state, task=task, sampler=sampler, key=key)
    return run(
        run_state=run_state,
        dataset=task.loaded_dataset,
        n_iterations=n_iterations,
        key=key,
        verbose=verbose,
    )


batch_reset = eqx.filter_vmap(
    reset,
    in_axes=(None, None, None, 0),
)


batch_run = eqx.filter_vmap(
    run,
    in_axes=(0, None, None, None, None),
)


batch_evaluate = eqx.filter_vmap(
    evaluate,
    in_axes=(None, None, None, None, 0, None),
)
