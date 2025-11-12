"""
Module for defining the update state of the optimization process.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from functools import partial

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

# Local
from ._typing import InputParameters, OptHistoryType
from .batching import batch_iter
from .task import Task
from .tree_utils import combined_loss, get_best, tree_where

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
        return jax.tree_util.tree_leaves(self.last_params)[0].shape[0]

    @classmethod
    def init(
        cls,
        params: InputParameters,
        memory_size: int,
    ) -> UpdateState:
        # Add leading axis and tile to memory_size
        last_params = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None, ...], (memory_size,) + x.shape),
            params,
        )

        last_loss = jnp.broadcast_to(jnp.array([jnp.nan]), (memory_size,))

        return cls(
            last_params=last_params,
            best_params=params,
            last_loss=last_loss,
            best_loss=jnp.array([jnp.inf]),
            last_step=jnp.array(0, dtype=jnp.uint32),
        )

    @classmethod
    def from_history(
        cls, history: OptHistoryType, memory_size: int, last_step: int
    ) -> UpdateState:
        params = history.params[-memory_size:]
        losses = history.loss[-memory_size:]

        # Get the best parameters and loss from the history
        best_params, best_loss = get_best(params, losses)

        return cls(
            last_params=params,
            last_loss=losses,
            best_params=best_params,
            best_loss=best_loss,
            last_step=jnp.array(last_step, dtype=jnp.uint32),
        )

    def sample(
        self, task: Task, sampler: Callable, key: PRNGKeyArray
    ) -> UpdateState:
        n_samples = self.memory_size
        params, static = eqx.partition(task.model, eqx.is_inexact_array)

        # Sample new parameters using the provided sampler
        new_params = sampler(key, params, n_samples)

        data_iter = batch_iter(
            dataset=task.loaded_dataset,
            n_evals=n_samples,
            batch_size=task.batch_size,
            key=task.dataset.get("seed", None),
        )

        if task.pass_rng:
            rng_sample = {"key": jr.split(key, num=n_samples)}
        else:
            rng_sample = {}

        losses = jax.vmap(
            partial(
                combined_loss,
                static=static,
                loss_fn=task.loss_fn,
                **next(data_iter),
            )
        )(new_params, **rng_sample)

        best_params, best_loss = get_best(new_params, losses)

        return UpdateState(
            last_params=new_params,
            last_loss=losses,
            best_params=best_params,
            best_loss=best_loss,
            last_step=jnp.array(SAMPLING_STEP, dtype=jnp.uint32),
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

    def evaluate(self, task: Task, key: PRNGKeyArray) -> jax.Array:
        return evaluate(params=self.last_params, task=task, key=key)

    def evaluate_with_grads(
        self, task: Task, key: PRNGKeyArray
    ) -> tuple[jax.Array, PyTree]:
        return evaluate_with_grads(params=self.last_params, task=task, key=key)


# =============================================================================


def evaluate(params: PyTree, task: Task, *, key: PRNGKeyArray) -> jax.Array:
    _, static = eqx.partition(task.model, eqx.is_inexact_array)
    n_samples = jax.tree_util.tree_leaves(params)[0].shape[0]

    data_iter = batch_iter(
        dataset=task.loaded_dataset,
        n_evals=n_samples,
        batch_size=task.batch_size,
        key=task.dataset.get("seed", None),
    )

    if task.pass_rng:
        rng_sample = {"key": jr.split(key, num=n_samples)}
    else:
        rng_sample = {}

    return jax.vmap(
        partial(
            combined_loss,
            static=static,
            loss_fn=task.loss_fn,
            **next(data_iter),
        )
    )(params, **rng_sample)


def evaluate_with_grads(
    params: PyTree, task: Task, *, key: PRNGKeyArray
) -> tuple[jax.Array, PyTree]:
    _, static = eqx.partition(task.model, eqx.is_inexact_array)
    n_samples = jax.tree_util.tree_leaves(params)[0].shape[0]

    data_iter = batch_iter(
        dataset=task.loaded_dataset,
        n_evals=n_samples,
        batch_size=task.batch_size,
        key=task.dataset.get("seed", None),
    )

    if task.pass_rng:
        rng_sample = {"key": jr.split(key, num=n_samples)}
    else:
        rng_sample = {}

    # Define a single-sample loss function
    def loss_fn_single(params_i, **rng_or_data):
        return combined_loss(
            params=params_i,
            static=static,
            loss_fn=task.loss_fn,
            **rng_or_data,
        )

    # Compute both loss and gradient per sample
    loss_and_grad_fn = jax.value_and_grad(loss_fn_single)

    losses, grads = jax.vmap(lambda p, **rng: loss_and_grad_fn(p, **rng))(
        params, **rng_sample, **next(data_iter)
    )

    return losses, grads


def new(
    update_state: UpdateState,
    task: Task,
    sampler: Callable,
    key: PRNGKeyArray,
) -> UpdateState:
    n_samples = update_state.memory_size
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    # Sample new parameters using the provided sampler
    new_params = sampler(key, params, n_samples)

    data_iter = batch_iter(
        dataset=task.loaded_dataset,
        n_evals=n_samples,
        batch_size=task.batch_size,
        key=task.dataset.get("seed", None),
    )

    if task.pass_rng:
        rng_sample = {"key": jr.split(key, num=n_samples)}
    else:
        rng_sample = {}

    losses = jax.vmap(
        partial(
            combined_loss,
            static=static,
            loss_fn=task.loss_fn,
            **next(data_iter),
        )
    )(new_params, **rng_sample)

    best_params, best_loss = get_best(new_params, losses)

    return UpdateState(
        last_params=new_params,
        last_loss=losses,
        best_params=best_params,
        best_loss=best_loss,
        last_step=jnp.array(SAMPLING_STEP, dtype=jnp.uint32),
    )
