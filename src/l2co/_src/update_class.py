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
from jaxtyping import PRNGKeyArray, PyTree

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
    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    stop_fn: StopFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int

    def __init__(
        self,
        init_fn: Callable,
        step_fn: Callable,
        popsize: int,
        hash: int,
        stop_fn: Callable | None = None,
    ):
        self.init_fn = init_fn
        self.step_fn = step_fn
        self.popsize = popsize
        self.hash = hash
        self.stop_fn = stop_fn or (lambda *_, **__: False)

    def init(self, update_state: UpdateState, key: PRNGKeyArray) -> PyTree:
        """Initialize the optimizer state."""
        return self.init_fn(update_state, key)

    def step(
        self,
        opt_state: PyTree,
        update_state: UpdateState,
        history_state: HistoryState,
        batch_state: BatchState,
        n_iterations: int,
        key: PRNGKeyArray,
    ) -> tuple[UpdateState, PyTree, BatchState, HistoryState]:
        # Retrieve the last parameters from the update state
        params = jax.tree.map(
            lambda x: x[-self.popsize :], update_state.last_params
        )

        window_size = 20

        def scan_step(
            carry: tuple[
                PyTree,
                tuple[PyTree],
                PRNGKeyArray,
                BatchState,
                jax.Array,
                OptHistory,
            ],
            _,
        ):
            params, opt_state, key, batch_state, done, recent_history = carry
            stop_condition = self.stop_fn(recent_history, opt_state)

            done = done | stop_condition

            def do_step(
                carry: tuple[
                    PyTree,
                    tuple[PyTree],
                    PRNGKeyArray,
                    BatchState,
                    jax.Array,
                    OptHistory,
                ],
            ):
                (params, opt_state, key, batch_state, done, recent_history) = (
                    carry
                )
                sample, batch_state = batch_state.next(key)
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
                carry: tuple[
                    PyTree,
                    tuple[PyTree],
                    PRNGKeyArray,
                    BatchState,
                    jax.Array,
                    OptHistory,
                ],
            ):
                (params, opt_state, key, batch_state, done, recent_history) = (
                    carry
                )
                dummy_history = recent_history.init(
                    params, self.popsize, self.hash
                )

                return (
                    params,
                    opt_state,
                    key,
                    batch_state,
                    done,
                    recent_history,
                ), dummy_history

            (new_carry, history_entry) = jax.lax.cond(
                done, skip_step, do_step, carry
            )

            return new_carry, history_entry

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
        (_, opt_state, _, batch_state, _, _), history = jax.lax.scan(
            f=scan_step,
            init=(
                params,
                opt_state,
                key,
                batch_state,
                init_done,
                recent_history_init,
            ),
            xs=None,
            length=n_iterations + 1,
        )

        # Remove first element for dim=0 history
        history = jax.tree.map(lambda x: x[1:, :], history)

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


# =============================================================================
