"""
Module for defining the handshake function between optimizer states
and update states.
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
from jaxtyping import PRNGKeyArray

from ._typing import OptState

# Local
from .batching import batch_iter
from .task import Task
from .tree_utils import combined_loss, get_best
from .update_state import UpdateState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

SAMPLING_STEP = 1

#                                                         Between Update States
# =============================================================================


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
        best_params=best_params,
        best_loss=best_loss,
        last_step=jnp.array(SAMPLING_STEP, dtype=jnp.uint32),
    )


def best(update_state: UpdateState, **kwargs) -> UpdateState:
    new_last_params = jax.tree.map(
        lambda lp, bp: lp.at[-1].set(bp[0]),  # bp has shape (1, dim)
        update_state.last_params,
        update_state.best_params,
    )
    return UpdateState(
        last_params=new_last_params,
        best_params=update_state.best_params,
        best_loss=update_state.best_loss,
        last_step=update_state.last_step,
    )


def best_one(
    update_state: UpdateState,
    task: Task,
    sampler: Callable,
    key: PRNGKeyArray,
    **kwargs,
) -> UpdateState:
    samples = new(
        update_state=update_state,
        task=task,
        sampler=sampler,
        key=key,
    )

    new_last_params = jax.tree.map(
        lambda lp, sp: lp.at[-1].set(sp[0]),  # sp
        samples.last_params,
        update_state.last_params,
    )

    new_update_state = UpdateState(
        last_params=new_last_params,
        best_params=samples.best_params,
        best_loss=samples.best_loss,
        last_step=update_state.last_step,
    )

    return samples.merge(new_update_state)


#                                           Between similar optimization states
# =============================================================================


def reset_opt_state(
    opt_state: OptState,
    update_state: UpdateState,
    update_class,
    key: PRNGKeyArray,
) -> OptState:
    """Reset the optimizer state."""
    return update_class.init(update_state, key)


# =============================================================================
