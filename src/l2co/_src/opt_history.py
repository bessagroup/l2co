"""
Module for the opt history.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import equinox as eqx

# Third-party
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

# Local
from .task import Task
from .typing import OptHistoryType
from .update_state import UpdateState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptHistory(OptHistoryType):
    """Container for optimization history data.

    Stores the history of losses, parameters, gradients, and evaluation
    metrics for a population of optimization candidates.

    Attributes
    ----------
    loss : Float[Array, " popsize"]
        Loss values for each population member.
    params : Float[PyTree, " popsize dim"]
        Parameter values for each population member.
    grads : Float[PyTree, " popsize dim"]
        Gradient values for each population member.
    update_step : Int[Array, " popsize"]
        Update step number for each population member.
    fevals : Int[Array, " popsize"]
        Number of function evaluations for each population member.
    iterations : Int[Array, " popsize"]
        Iteration count for each population member.
    """

    loss: Float[Array, " popsize"]
    params: Float[PyTree, " popsize dim"]
    grads: Float[PyTree, " popsize dim"]
    update_step: Int[Array, " popsize"]
    fevals: Int[Array, " popsize"]
    iterations: Int[Array, " popsize"]

    @classmethod
    def init(cls, params: PyTree, popsize: int, last_step: int) -> OptHistory:
        """Initialize optimization history with NaN values.

        Parameters
        ----------
        params : PyTree
            Parameter tree structure.
        popsize : int
            Population size.
        last_step : int
            Last update step number.

        Returns
        -------
        OptHistory
            Initialized optimization history.
        """
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
    @eqx.filter_jit
    def from_init_sample(
        cls, update_state: UpdateState, task: Task, key: PRNGKeyArray
    ):
        """Create optimization history from initial sample evaluation.

        Parameters
        ----------
        update_state : UpdateState
            Current update state containing initial parameters.
        task : Task
            Task to evaluate against.
        key : PRNGKeyArray
            PRNG key for random operations.

        Returns
        -------
        OptHistory
            Optimization history initialized with evaluation results.
        """
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
        """Update the recent history with the current history.

        Maintains a rolling window by removing the oldest entry and appending
        the new history.

        Parameters
        ----------
        history : OptHistory
            New history to append.

        Returns
        -------
        OptHistory
            Updated optimization history.
        """

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
