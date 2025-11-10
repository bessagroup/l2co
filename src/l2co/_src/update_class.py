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

from ._typing import InitFunction, OptStates, StepFunction, StopFunction

# Local
from .handshake import SAMPLING_STEP
from .history_state import HistoryState
from .opt_history import OptHistory
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
    handshake_fn: Callable = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    index: int = eqx.field(static=True)
    hash: int
    stop_fn: StopFunction = eqx.field(static=True)
    handshake_fn_opt_state: Callable = eqx.field(static=True)

    def __init__(
        self,
        init_fn: Callable,
        handshake_fn: Callable,
        step_fn: Callable,
        popsize: int,
        index: int,
        hash: int,
        stop_fn: Callable | None = None,
        handshake_fn_opt_state: Callable | None = None,
    ):
        self.init_fn = init_fn
        self.handshake_fn = handshake_fn
        self.step_fn = step_fn
        self.popsize = popsize
        self.index = index
        self.hash = hash
        self.stop_fn = stop_fn or (lambda *_, **__: False)
        self.handshake_fn_opt_state = handshake_fn_opt_state or (
            lambda opt_state, *_, **__,: opt_state
        )

    def init(self, update_state: UpdateState, key: PRNGKeyArray) -> PyTree:
        """Initialize the optimizer state."""
        return self.init_fn(update_state, key)

    def handshake(
        self,
        update_state: UpdateState,
        opt_states: OptStates,
        key: PRNGKeyArray,
    ) -> tuple[UpdateState, OptStates]:
        # Determine whether to skip handshake
        do_handshake = jnp.logical_not(
            (update_state.last_step == SAMPLING_STEP)
            | (update_state.last_step == self.hash)
        )

        # Conditionally perform handshake
        new_update_state = jax.lax.cond(
            do_handshake,
            lambda s: self.handshake_fn(update_state=s, key=key),
            lambda s: s,
            update_state,
        )

        # Conditionally perform opt state handshake
        new_opt_state = jax.lax.cond(
            do_handshake,
            lambda o: self.handshake_fn_opt_state(
                opt_state=o,
                update_state=update_state,
                update_class=self,
                key=key,
            ),
            lambda o: o,
            opt_states[self.index],
        )

        # Update the opt states tuple
        new_opt_states = eqx.tree_at(
            lambda t: t[self.index], opt_states, new_opt_state
        )

        return new_update_state, new_opt_states

    def step(
        self,
        opt_states: tuple[PyTree],
        update_state: UpdateState,
        history_state: HistoryState,
        n_iterations: int,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, UpdateState]:
        update_state, opt_states = self.handshake(
            update_state=update_state, opt_states=opt_states, key=key
        )

        # Retrieve the last parameters from the update state
        params = jax.tree.map(
            lambda x: x[-self.popsize :], update_state.last_params
        )

        window_size = 20

        def scan_step(
            carry: tuple[
                PyTree, tuple[PyTree], PRNGKeyArray, jax.Array, OptHistory
            ],
            _,
        ):
            params, opt_states, key, done, recent_history = carry
            stop_condition = self.stop_fn(
                recent_history, opt_states, self.index
            )

            done = done | stop_condition

            def do_step(
                carry: tuple[
                    PyTree,
                    tuple[PyTree],
                    PRNGKeyArray,
                    jax.Array,
                    OptHistory,
                ],
            ):
                (params, opt_states, key, done, recent_history) = carry
                new_carry, history = self.step_fn(
                    (params, opt_states, key), sample={}
                )

                new_recent_history = recent_history.update(history)

                return (*new_carry, done, new_recent_history), history

            def skip_step(
                carry: tuple[
                    PyTree,
                    tuple[PyTree],
                    PRNGKeyArray,
                    jax.Array,
                    OptHistory,
                ],
            ):
                (params, opt_states, key, done, recent_history) = carry
                dummy_history = recent_history.init(
                    params, self.popsize, self.hash
                )

                return (
                    params,
                    opt_states,
                    key,
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
        (_, opt_states, _, _, _), history = jax.lax.scan(
            f=scan_step,
            init=(params, opt_states, key, init_done, recent_history_init),
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

        return update_state, opt_states, new_history_state


# =============================================================================
