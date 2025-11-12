"""
Module for the update class.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Third-party
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from ._typing import OptHistoryType

# Local
from .task import Task
from .update_state import UpdateState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# =============================================================================


class OptHistory(OptHistoryType):
    loss: Float[Array, " popsize"]
    params: Float[PyTree, " popsize dim"]
    grads: Float[PyTree, " popsize dim"]
    update_step: Int[Array, " popsize"]
    fevals: Int[Array, " popsize"]
    iterations: Int[Array, " popsize"]

    @classmethod
    def init(cls, params: PyTree, popsize: int, last_step: int) -> OptHistory:
        loss = jnp.full((popsize,), jnp.nan)
        params = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), params)
        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), params)
        update_step = jnp.full((popsize,), last_step)
        fevals = jnp.zeros((popsize,), dtype=jnp.int32)
        iterations = jnp.zeros((popsize,), dtype=jnp.int32)

        return cls(
            loss=loss,
            params=params,
            grads=grads,
            update_step=update_step,
            fevals=fevals,
            iterations=iterations,
        )

    @classmethod
    def from_init_sample(
        cls, update_state: UpdateState, task: Task, key: PRNGKeyArray
    ):
        # Evaluate the initial parameters
        init_params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        init_loss, init_grads = jax.tree.map(
            lambda x: x[-1], update_state.evaluate_with_grads(task, key)
        )

        loss = jax.tree.map(lambda x: jnp.expand_dims(x, (0, 1)), init_loss)
        params = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), init_params
        )
        grads = jax.tree.map(lambda x: jnp.expand_dims(x, (0, 1)), init_grads)
        update_step = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), update_state.last_step
        )
        fevals = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), jnp.array(0)
        )
        iterations = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), jnp.array(0)
        )

        return cls(
            loss=loss,
            params=params,
            grads=grads,
            update_step=update_step,
            fevals=fevals,
            iterations=iterations,
        )

    def update(self, history: OptHistory) -> OptHistory:
        """Update the recent history with the current history."""

        loss = jnp.concatenate([self.loss[1:], history.loss[None]], axis=0)
        params = jax.tree.map(
            lambda p, new: jnp.concatenate([p[1:], new[None]], axis=0),
            self.params,
            history.params,
        )
        grads = jax.tree.map(
            lambda g, new: jnp.concatenate([g[1:], new[None]], axis=0),
            self.grads,
            history.grads,
        )
        update_step = jnp.concatenate(
            [self.update_step[1:], history.update_step[None]], axis=0
        )
        fevals = jnp.concatenate(
            [self.fevals[1:], history.fevals[None]], axis=0
        )

        iterations = jnp.concatenate(
            [self.iterations[1:], history.iterations[None]], axis=0
        )
        return OptHistory(
            loss=loss,
            params=params,
            grads=grads,
            update_step=update_step,
            fevals=fevals,
            iterations=iterations,
        )
