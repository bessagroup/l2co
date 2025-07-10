"""
Module for defining the update state of the optimization process.
"""

#                                                                       Modules
# =============================================================================


# Standard
from __future__ import annotations

from functools import partial
from typing import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree

# Local
from ._typing import InputParameters
from .batching import batch_iter
from .task import Task
from .tree_utils import combined_loss, get_best, tree_where

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


SAMPLING_STEP = 1


class UpdateState(eqx.Module):
    last_params: PyTree  # shape: (memory_size, dim)
    key: jax.Array
    best_params: PyTree  # shape: (1, dim)
    best_loss: jax.Array  # shape: (1,)
    last_step: jax.Array

    @property
    def memory_size(self) -> int:
        # Assumes all leaves have same leading dimension
        return jax.tree_util.tree_leaves(self.last_params)[0].shape[0]

    @classmethod
    def init(cls, params: InputParameters, memory_size: int,
             key: PRNGKeyArray) -> UpdateState:

        # Add leading axis and tile to memory_size
        last_params = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None, ...], (memory_size,) + x.shape),
            params
        )

        return cls(
            last_params=last_params,
            key=key,
            best_params=params,
            best_loss=jnp.array([jnp.inf]),
            last_step=jnp.array(0, dtype=jnp.uint32),
        )

    @classmethod
    def from_history(cls, history: PyTree,
                     memory_size: int, last_step: int,
                     key: InputParameters) -> UpdateState:

        params = history['params'][-memory_size:]
        losses = history['loss'][-memory_size:]

        # Get the best parameters and loss from the history
        best_params, best_loss = get_best(params, losses)

        return cls(
            last_params=params,
            key=key,
            best_params=best_params,
            best_loss=best_loss,
            last_step=last_step,
        )

    def merge(self, other: UpdateState) -> UpdateState:
        pred = self.best_loss < other.best_loss
        best_params = tree_where(pred, self.best_params, other.best_params)
        best_loss = jnp.where(pred, self.best_loss, other.best_loss)

        return UpdateState(
            last_params=other.last_params,
            key=other.key,
            best_params=best_params,
            best_loss=best_loss,
            last_step=other.last_step,
        )


def new(
    update_state: UpdateState,
    task: Task,
    sampler: Callable,
) -> UpdateState:

    n_samples = update_state.memory_size
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    new_key, sample_key = jr.split(update_state.key)

    # Sample new parameters using the provided sampler
    new_params = sampler(sample_key, params, n_samples)

    data_iter = batch_iter(dataset=task.loaded_dataset,
                           n_evals=n_samples,
                           batch_size=task.batch_size,
                           key=task.dataset.get('seed', None))

    if task.pass_rng:
        rng_sample = {'key': jr.split(sample_key, num=n_samples)}
    else:
        rng_sample = {}

    losses = jax.vmap(partial(combined_loss, static=static,
                              loss_fn=task.loss_fn,
                              **next(data_iter))
                      )(new_params, **rng_sample)

    best_params, best_loss = get_best(new_params, losses)

    return UpdateState(
        last_params=new_params,
        key=new_key,
        best_params=best_params,
        best_loss=best_loss,
        last_step=SAMPLING_STEP
    )


def best(update_state: UpdateState, **kwargs) -> UpdateState:
    new_last_params = jax.tree.map(
        lambda lp, bp: lp.at[-1].set(bp[0]),  # bp has shape (1, dim)
        update_state.last_params,
        update_state.best_params,
    )
    return UpdateState(
        last_params=new_last_params,
        key=update_state.key,
        best_params=update_state.best_params,
        best_loss=update_state.best_loss,
        last_step=update_state.last_step,
    )


def best_one(update_state: UpdateState, task: Task, sampler: Callable,
             **kwargs) -> UpdateState:

    samples = new(
        update_state=update_state,
        task=task,
        sampler=sampler,
    )

    new_last_params = jax.tree.map(
        lambda lp, sp: lp.at[-1].set(sp[0]),  # sp
        samples.last_params,
        update_state.last_params,
    )

    new_update_state = UpdateState(
        last_params=new_last_params,
        key=samples.key,
        best_params=samples.best_params,
        best_loss=samples.best_loss,
        last_step=update_state.last_step,
    )

    return samples.merge(new_update_state)
