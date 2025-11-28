"""
Module for the update class.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax_tqdm import scan_tqdm
from jaxtyping import PRNGKeyArray, PyTree
from tqdm import tqdm

# Local
from .batching import BatchState
from .history_state import HistoryState
from .opt_history import OptHistory
from .typing import InitFunction, StepFunction, StopFunction
from .update_state import UpdateState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# =============================================================================


class UpdateClass(eqx.Module):
    """Class for managing optimizer update operations.

    This class encapsulates the initialization, stepping, and stopping logic
    for optimization algorithms.

    Attributes
    ----------
    init_fn : InitFunction
        Function to initialize optimizer state.
    step_fn : StepFunction
        Function to perform optimization step.
    stop_fn : StopFunction
        Function to check stopping criteria.
    popsize : int
        Population size for the optimizer.
    hash : int
        Hash identifier for the optimizer configuration.
    """

    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    stop_fn: StopFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)

    def __init__(
        self,
        init_fn: Callable,
        step_fn: Callable,
        popsize: int,
        hash: int,
        stop_fn: Callable | None = None,
    ):
        """Initialize UpdateClass.

        Parameters
        ----------
        init_fn : Callable
            Function to initialize optimizer state.
        step_fn : Callable
            Function to perform optimization step.
        popsize : int
            Population size for the optimizer.
        hash : int
            Hash identifier for the optimizer configuration.
        stop_fn : Callable | None, optional
            Function to check stopping criteria, by default None.
        """
        self.init_fn = init_fn
        self.step_fn = step_fn
        self.popsize = popsize
        self.hash = hash
        self.stop_fn = stop_fn or (lambda *_, **__: False)

    @eqx.filter_jit
    def init(self, update_state: UpdateState, key: PRNGKeyArray) -> PyTree:
        """Initialize the optimizer state.

        Parameters
        ----------
        update_state : UpdateState
            Current update state.
        key : PRNGKeyArray
            PRNG key for random operations.

        Returns
        -------
        PyTree
            Initialized optimizer state.
        """
        return self.init_fn(update_state, key)

    def step(
        self,
        opt_state: PyTree,
        update_state: UpdateState,
        history_state: HistoryState,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        n_iterations: int,
        verbose: bool,
    ) -> tuple[UpdateState, PyTree, BatchState, HistoryState]:
        """Perform optimization steps for specified iterations.

        Parameters
        ----------
        opt_state : PyTree
            Current optimizer state.
        update_state : UpdateState
            Current update state.
        history_state : HistoryState
            Current history state.
        batch_state : BatchState
            Current batch state.
        dataset : dict[str, jax.Array]
            Dataset for evaluation.
        key : PRNGKeyArray
            PRNG key for random operations.
        n_iterations : int
            Number of iterations to perform.
        verbose : bool
            Whether to print progress information.

        Returns
        -------
        tuple[UpdateState, PyTree, BatchState, HistoryState]
            Updated states after optimization steps.
        """
        # Retrieve the last parameters from the update state
        params = jax.tree.map(
            lambda x: x[-self.popsize :], update_state.last_params
        )

        window_size = 20

        # Initialize moving history buffer with NaNs
        recent_history_init = OptHistory.init(
            params=params,
            popsize=self.popsize,
            last_step=update_state.last_step,
        )

        # Initialize the recent history
        recent_history_init = jax.tree.map(
            lambda x: jnp.tile(x[None], (window_size, *([1] * x.ndim))),
            recent_history_init,
        )

        # Initialize done flag
        init_done = jnp.array(False)

        # Run the optimization loop
        (_, opt_state, _, batch_state, _, _), history = self.scan_loop(
            carry=(
                params,
                opt_state,
                key,
                batch_state,
                init_done,
                recent_history_init,
            ),
            n_iterations=n_iterations + 1,
            dataset=dataset,
            verbose=verbose,
        )

        # Remove first element for dim=0 history
        history: OptHistory = jax.tree.map(lambda x: x[1:, :], history)

        # Flatten the first and second dimension of each key in history
        history_flattened = jax.tree.map(
            lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
            history,
        )

        # Create final update state from history
        final_update_state = UpdateState.from_history(
            history=history_flattened,
            memory_size=update_state.memory_size,
            last_step=self.hash,
        )

        # Merge with previous update state
        update_state = update_state.merge(final_update_state)

        # Update history state
        new_history_state = history_state.add(history)

        return update_state, opt_state, batch_state, new_history_state

    def do_step(
        self,
        carry: tuple[
            PyTree,
            tuple[PyTree],
            PRNGKeyArray,
            BatchState,
            jax.Array,
            OptHistory,
        ],
        # step_fn: StepFunction,
        dataset: dict[str, jax.Array],
    ) -> tuple[
        tuple[
            PyTree,
            tuple[PyTree],
            PRNGKeyArray,
            BatchState,
            jax.Array,
            OptHistory,
        ],
        OptHistory,
    ]:
        """Execute a single optimization step.

        Parameters
        ----------
        carry : tuple
            Tuple containing (params, opt_state, key, batch_state, done,
            recent_history).
        dataset : dict[str, jax.Array]
            Dataset for evaluation.

        Returns
        -------
        tuple
            Updated carry and optimization history.
        """
        (params, opt_state, key, batch_state, done, recent_history) = carry
        batch_idxs, batch_state = batch_state.next(key)
        sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
        new_carry, history = self.step_fn(
            (params, opt_state, key), sample=sample
        )

        new_recent_history = recent_history.update(history)

        return (
            *new_carry,
            batch_state,
            done,
            new_recent_history,
        ), history

    def skip_step(
        self,
        carry: tuple[
            PyTree,
            tuple[PyTree],
            PRNGKeyArray,
            BatchState,
            jax.Array,
            OptHistory,
        ],
    ):
        """Skip an optimization step when stopping criteria is met.

        Parameters
        ----------
        carry : tuple
            Tuple containing (params, opt_state, key, batch_state, done,
            recent_history).

        Returns
        -------
        tuple
            Unchanged carry and dummy history.
        """
        (params, opt_state, key, batch_state, done, recent_history) = carry
        dummy_history = recent_history.init(
            params, self.popsize, recent_history.update_step[-1]
        )

        return (
            params,
            opt_state,
            key,
            batch_state,
            done,
            recent_history,
        ), dummy_history

    def scan_step(
        self,
        carry: tuple[
            PyTree,
            tuple[PyTree],
            PRNGKeyArray,
            BatchState,
            jax.Array,
            OptHistory,
        ],
        xs,
        dataset: dict[str, jax.Array],
    ) -> tuple[
        tuple[
            PyTree,
            tuple[PyTree],
            PRNGKeyArray,
            BatchState,
            jax.Array,
            OptHistory,
        ],
        OptHistory,
    ]:
        """Perform a single scan step with conditional execution.

        Parameters
        ----------
        carry : tuple
            Tuple containing (params, opt_state, key, batch_state, done,
            recent_history).
        xs : Any
            Scan iteration variable (unused).
        dataset : dict[str, jax.Array]
            Dataset for evaluation.

        Returns
        -------
        tuple
            Updated carry and optimization history.
        """
        _, opt_state, _, _, done, recent_history = carry

        (new_carry, history_entry) = jax.lax.cond(
            done | self.stop_fn(recent_history, opt_state),
            self.skip_step,
            Partial(self.do_step, dataset=dataset),
            carry,
        )

        return new_carry, jax.lax.stop_gradient(history_entry)

    def scan_loop(
        self,
        carry,
        n_iterations: int,
        dataset: dict[str, jax.Array],
        verbose: bool,
    ) -> tuple[
        tuple[
            PyTree,
            tuple[PyTree],
            PRNGKeyArray,
            BatchState,
            jax.Array,
            OptHistory,
        ],
        OptHistory,
    ]:
        """Execute the optimization loop using JAX scan.

        Parameters
        ----------
        carry : tuple
            Initial carry state.
        n_iterations : int
            Number of iterations to perform.
        dataset : dict[str, jax.Array]
            Dataset for evaluation.
        verbose : bool
            Whether to show progress bar.

        Returns
        -------
        tuple
            Final carry state and optimization history.
        """
        scan_fn = Partial(self.scan_step, dataset=dataset)
        if verbose:
            fn = scan_tqdm(n_iterations)(scan_fn)
        else:
            fn = scan_fn

        # Run the optimization loop
        return jax.lax.scan(
            f=fn,
            init=carry,
            xs=jnp.arange(n_iterations),
            length=n_iterations,
        )


# =============================================================================


def for_loop(carry, scan_fn, n_iterations: int):
    """Execute optimization loop using Python for loop with progress bar.

    Parameters
    ----------
    carry : tuple
        Initial carry state.
    scan_fn : Callable
        Scan function to execute at each iteration.
    n_iterations : int
        Number of iterations to perform.

    Returns
    -------
    tuple
        Final carry state and accumulated history buffer.
    """
    jitted_scan_fn = jax.jit(scan_fn)
    carry, history = jitted_scan_fn(carry, None)
    buffer = jax.tree.map(
        lambda x: jnp.tile(x[None], (n_iterations, *([1] * x.ndim))),
        history,
    )

    # Run the optimization loop
    for i in tqdm(range(1, n_iterations)):
        carry, history = jitted_scan_fn(carry, None)
        buffer = jax.tree.map(
            lambda buf, new, i=i: buf.at[i].set(new),
            buffer,
            history,
        )
    return carry, buffer
