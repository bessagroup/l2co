#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree

# Local
from .batching import BatchState
from .history_state import HistoryState
from .optimizers import optimizer_mapping
from .optimizerschedule import OptimizationStep
from .task import Task
from .update_class import UpdateClass
from .update_state import UpdateState

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
    update_state : UpdateState
        The current state of the optimizer updates.
    history_state : HistoryState
        The history of the optimization process.
    opt_state : PyTree
        The state of the optimizer.
    batch_state : BatchState
        The state of the data batching process.
    update_class : UpdateClass
        The class defining the optimizer update logic.
    max_iterations : int
        The maximum number of optimization iterations to perform.
    """

    update_state: UpdateState
    history_state: HistoryState
    opt_state: PyTree
    batch_state: BatchState
    update_class: UpdateClass = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        optimizer: OptimizationStep,
        task: Task,
        bounded: tuple[float, float],
        max_iterations: int,
        key: PRNGKeyArray,
    ) -> RunState:
        params = eqx.filter(task.model, eqx.is_inexact_array)

        update_class: UpdateClass = optimizer_mapping(optimizer.optimizer)(
            **optimizer.hyperparameters,
            task=task,
            opt_hash=optimizer.hash,
            bounded=bounded,
            stop_fn=optimizer.stopping_fn,
        )
        update_state = UpdateState.init(
            params=params,
            memory_size=update_class.popsize,
        )

        # Initialize the optimizer state
        opt_state = update_class.init(update_state=update_state, key=key)

        # Initialize the history state
        history_state = HistoryState.init(
            n_iterations=max_iterations + 1,
            params=params,
        )

        # Initialize batch state
        batch_state = BatchState.init(
            dataset=task.loaded_dataset,
            batch_size=task.batch_size,
            key=key,
        )

        return cls(
            update_state=update_state,
            history_state=history_state,
            opt_state=opt_state,
            batch_state=batch_state,
            update_class=update_class,
            max_iterations=max_iterations,
        )


@eqx.filter_jit
def reset(
    run_state: RunState, task: Task, sampler: Callable, key: PRNGKeyArray
) -> RunState:
    batch_state = run_state.batch_state.reset(key=key)

    _, static = eqx.partition(task.model, eqx.is_inexact_array)

    update_state = run_state.update_state.reset().sample(
        batch_state=batch_state,
        static=static,
        dataset=task.loaded_dataset,
        loss_fn=task.loss_fn,
        sampler=sampler,
        key=key,
    )

    history_state = run_state.history_state.reset(update_state)
    opt_state = run_state.update_class.init(update_state=update_state, key=key)

    return RunState(
        update_state=update_state,
        history_state=history_state,
        opt_state=opt_state,
        batch_state=batch_state,
        update_class=run_state.update_class,
        max_iterations=run_state.max_iterations,
    )


@eqx.filter_jit
def run(
    run_state: RunState,
    dataset: dict[str, jax.Array],
    n_iterations: int,
    key: PRNGKeyArray,
    verbose: bool,
) -> HistoryState:
    # Main optimization loop
    _, _, _, history_state = run_state.update_class.step(
        opt_state=run_state.opt_state,
        update_state=run_state.update_state,
        history_state=run_state.history_state,
        batch_state=run_state.batch_state,
        dataset=dataset,
        key=key,
        n_iterations=n_iterations,
        verbose=verbose,
    )

    return history_state


@eqx.filter_jit
def evaluate(
    run_state: RunState,
    task: Task,
    sampler: Callable,
    key: PRNGKeyArray,
    verbose: bool,
) -> HistoryState:
    run_state = reset(run_state, task=task, sampler=sampler, key=key)
    return run(
        run_state=run_state,
        dataset=task.loaded_dataset,
        n_iterations=run_state.max_iterations,
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
    in_axes=(None, None, None, 0, None),
)
