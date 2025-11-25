"""
Module for defining the update state of the optimization process.
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
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from .batching import BatchState
from .task import Task
from .tree_utils import get_best, tree_where
from .typing import InputParameters, OptHistoryType

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

SAMPLING_STEP = 1

# =============================================================================


class UpdateState(eqx.Module):
    last_params: Float[Array, "memory_size dim"]
    last_loss: Float[Array, " memory size"]
    best_params: Float[Array, " dim"]
    best_loss: Float[Array, 1]
    last_step: Int[Array, 1]

    @property
    def memory_size(self) -> int:
        # Assumes all leaves have same leading dimension
        return jax.tree.leaves(self.last_params)[0].shape[0]

    @classmethod
    def init(
        cls,
        params: InputParameters,
        memory_size: int,
    ) -> UpdateState:
        # Efficiently add leading axis by broadcasting
        last_params = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (memory_size,) + x.shape),
            params,
        )

        # initialize last_loss as a 1D float array of NaNs
        last_loss = jnp.full((memory_size,), jnp.nan, dtype=jnp.float32)

        return cls(
            last_params=last_params,
            best_params=params,
            last_loss=last_loss,
            best_loss=jnp.array(jnp.inf, dtype=jnp.float32),
            last_step=jnp.array(0, dtype=jnp.int32),
        )

    @eqx.filter_jit
    def reset(self) -> UpdateState:
        return self.init(params=self.best_params, memory_size=self.memory_size)

    @classmethod
    def from_history(
        cls, history: OptHistoryType, memory_size: int, last_step: int
    ) -> UpdateState:
        params = jax.tree.map(lambda x: x[-memory_size:], history.params)
        losses = history.loss[-memory_size:]

        # Get the best parameters and loss from the history
        best_params, best_loss = get_best(params, losses)

        return cls(
            last_params=params,
            last_loss=losses,
            best_params=best_params,
            best_loss=best_loss,
            last_step=jnp.array(last_step, dtype=jnp.int32),
        )

    @eqx.filter_jit
    def sample(
        self, task: Task, sampler: Callable, key: PRNGKeyArray
    ) -> UpdateState:
        n_samples = self.memory_size
        _, static = eqx.partition(task.model, eqx.is_inexact_array)

        # Sample new parameters using the provided sampler
        new_params = sampler(key, self.best_params, n_samples)

        batch_state = BatchState.init(
            dataset=task.loaded_dataset, batch_size=task.batch_size, key=key
        )
        losses = evaluate(
            eqx.combine(new_params, static),
            batch_state,
            task.loaded_dataset,
            task.loss_fn,
            key,
        )

        best_params, best_loss = get_best(new_params, losses)

        return UpdateState(
            last_params=new_params,
            last_loss=losses,
            best_params=best_params,
            best_loss=best_loss,
            last_step=jnp.array(SAMPLING_STEP, dtype=jnp.int32),
        )

    def merge(self, other: UpdateState) -> UpdateState:
        pred = self.best_loss < other.best_loss
        best_params = tree_where(pred, self.best_params, other.best_params)
        best_loss = jnp.where(pred, self.best_loss, other.best_loss)

        return UpdateState(
            last_params=other.last_params,
            last_loss=other.last_loss,
            best_params=best_params,
            best_loss=best_loss,
            last_step=other.last_step,
        )

    def evaluate_with_grads(
        self, task: Task, key: PRNGKeyArray
    ) -> tuple[jax.Array, PyTree]:
        _, static = eqx.partition(task.model, eqx.is_inexact_array)

        batch_state = BatchState.init(
            dataset=task.loaded_dataset, batch_size=task.batch_size, key=key
        )
        return evaluate_with_grads(
            eqx.combine(self.last_params, static),
            batch_state,
            task.loaded_dataset,
            task.loss_fn,
            key,
        )


# =============================================================================


def _evaluate_one(
    model: PyTree,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    key: PRNGKeyArray,
) -> jax.Array:
    batch_idxs, _ = batch_state.next(key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
    return loss_fn(model, **sample)


evaluate = eqx.filter_vmap(
    eqx.filter_jit(_evaluate_one),
    in_axes=(0, None, None, None, None),
)

evaluate_with_grads = eqx.filter_vmap(
    eqx.filter_value_and_grad(_evaluate_one),
    in_axes=(0, None, None, None, None),
)
