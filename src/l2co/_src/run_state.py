"""Per-optimizer mutable run state used during optimization rollouts."""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from l2co_tasks import Task

# Local
from .batching import BatchState
from .history_state import HistoryState
from .optimizer_schedule import OptimizationStep
from .optimizers import optimizer_mapping
from .typing import InputParameters, SamplerFunction
from .update_class import UpdateClass
from .update_class import evaluate as evaluate_update_class

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
    best_params : InputParameters
        The best input parameters found so far, shaped as (dim,).
    best_loss : Array[Float, ""]
        The best loss value found so far.
    opt_state : PyTree
        The state of the optimizer.
    update_class : UpdateClass
        The class defining the optimizer update logic.
    """

    params: InputParameters
    # TODO: this has shape (dim,) not (1, dim)
    best_params: InputParameters
    best_loss: Array[Float, ""]
    opt_state: PyTree
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
        model_params = eqx.filter(task.model, eqx.is_inexact_array)

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
            model_params,
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
            best_params=model_params,
            best_loss=jnp.array(jnp.inf, dtype=jnp.float32),
            opt_state=opt_state,
            update_class=update_class,
        )


def reset(
    run_state: RunState,
    batch_state: BatchState,
    static: PyTree,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    pass_rng: bool,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> RunState:
    """Reset the run state with new samples.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    batch_state : BatchState
        Current batch state.
    static : PyTree
        Model of the task to optimize.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    loss_fn : Callable
        Loss function for evaluation.
    pass_rng : bool
        Whether to pass RNG keys to the loss function.
    sampler : SamplerFunction
        Sampling function for generating new parameters.
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    RunState
        Reset run state with new samples.
    """
    samples = sampler(
        key,
        run_state.best_params,
        run_state.update_class.popsize,
    )

    loss = evaluate_update_class(
        eqx.combine(static, samples),
        batch_state,
        dataset,
        loss_fn,
        pass_rng,
        key,
        jr.split(key, run_state.update_class.popsize),
    )

    # calculate the best params and best loss
    best_idx = jnp.argmin(loss)
    best_params = jax.tree.map(lambda x: x[best_idx], samples)
    best_loss = loss[best_idx]

    opt_state = run_state.update_class.init(
        params=samples,
        batch_state=batch_state,
        dataset=dataset,
        key=key,
    )

    return RunState(
        params=samples,
        best_params=best_params,
        best_loss=best_loss,
        opt_state=opt_state,
        update_class=run_state.update_class,
    )


def reset_from_stage0(
    run_state: RunState,
    best_params: PyTree,
    batch_state: BatchState,
    static: PyTree,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    pass_rng: bool,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> RunState:
    """Reset the run state with new samples.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    best_params : PyTree
        Best parameters from stage 0.
    batch_state : BatchState
        Current batch state.
    static : PyTree
        Model of the task to optimize.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    loss_fn : Callable
        Loss function for evaluation.
    pass_rng : bool
        Whether to pass RNG keys to the loss function.
    sampler : SamplerFunction
        Sampling function for generating new parameters.
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    RunState
        Reset run state with new samples.
    """
    samples = sampler(
        key,
        run_state.best_params,
        run_state.update_class.popsize,
    )

    # Replace the first sample with the best params from stage 0
    samples = jax.tree.map(
        lambda x, y: x.at[0].set(y),
        samples,
        best_params,
    )

    loss = evaluate_update_class(
        eqx.combine(static, samples),
        batch_state,
        dataset,
        loss_fn,
        pass_rng,
        key,
        jr.split(key, run_state.update_class.popsize),
    )

    # calculate the best params and best loss
    best_idx = jnp.argmin(loss)
    best_params = jax.tree.map(lambda x: x[best_idx], samples)
    best_loss = loss[best_idx]

    opt_state = run_state.update_class.init(
        params=samples,
        batch_state=batch_state,
        dataset=dataset,
        key=key,
    )

    return RunState(
        params=samples,
        best_params=best_params,
        best_loss=best_loss,
        opt_state=opt_state,
        update_class=run_state.update_class,
    )


def run(
    run_state: RunState,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    n_iterations: int,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, BatchState, HistoryState]:
    """Run optimization for a specified number of iterations.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    batch_state : BatchState
        Current batch state.
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
    tuple[RunState, BatchState, HistoryState]
        Updated RunState, BatchState and history state after optimization.
    """
    # Main optimization loop
    params, best_params, best_loss, opt_state, batch_state, history_state = (
        run_state.update_class.step(
            opt_state=run_state.opt_state,
            params=run_state.params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=verbose,
        )
    )

    run_state = RunState(
        params=params,
        best_params=best_params,
        best_loss=best_loss,
        opt_state=opt_state,
        update_class=run_state.update_class,
    )

    return run_state, batch_state, history_state


def evaluate(
    run_state: RunState,
    batch_state: BatchState,
    static: PyTree,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    sampler: Callable,
    n_iterations: int,
    pass_rng: bool,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, BatchState, HistoryState]:
    """Evaluate optimization by resetting and running to completion.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    batch_state : BatchState
        Current batch state.
    static : PyTree
        Model of the task to optimize.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    loss_fn : Callable
        Loss function for evaluation.
    sampler : Callable
        Sampling function for generating new parameters.
    n_iterations : int
        Number of iterations to run.
    pass_rng : bool
        Whether to pass RNG keys to the loss function.
    key : PRNGKeyArray
        PRNG key for random operations.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[RunState, BatchState, HistoryState]
        Tuple of updated RunState, BatchState and History state after
        complete optimization run.
    """
    run_state = reset(
        run_state=run_state,
        batch_state=batch_state,
        static=static,
        dataset=dataset,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        sampler=sampler,
        key=key,
    )
    return run(
        run_state=run_state,
        batch_state=batch_state,
        dataset=dataset,
        n_iterations=n_iterations,
        key=key,
        verbose=verbose,
    )


batch_reset = eqx.filter_vmap(
    reset,
    in_axes=(None, None, None, None, None, None, None, 0),
)


batch_run = eqx.filter_vmap(
    run,
    in_axes=(0, None, None, None, None, None),
)


batch_evaluate = eqx.filter_vmap(
    evaluate,
    in_axes=(None, None, None, None, None, None, None, None, 0, None),
)
