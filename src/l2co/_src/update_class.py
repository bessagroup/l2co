"""
The task-side helpers around :class:`~l2co_optimizers.UpdateClass`.

The run loop itself -- ``init_state`` / ``step`` / ``run`` /
``batch_run`` -- is a set of ``UpdateClass`` methods in
``l2co_optimizers`` (ADR 0017). What stays here is :func:`init_`, the
public free-function spelling of :meth:`UpdateClass.init_state`, and
:func:`evaluate`, which scores models through a task's loss and has no
optimizer in it.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import BatchState, InputParameters, UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "UpdateClass",
    "evaluate",
    "init_",
]


def init_(
    update_class: UpdateClass,
    params: InputParameters,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    key: PRNGKeyArray,
) -> PyTree:
    """Initialize the optimizer state.

    A delegate to :meth:`UpdateClass.init_state`, kept because
    ``l2co.init_`` is public.

    Parameters
    ----------
    update_class : UpdateClass
        Optimizer to initialize.
    params : InputParameters
        Current input parameters.
    batch_state : BatchState
        Current batch state.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    PyTree
        Initialized optimizer state.
    """
    return update_class.init_state(params, batch_state, dataset, key)


# =============================================================================


def _evaluate_one(
    model: PyTree,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    pass_rng: bool,
    batch_key: PRNGKeyArray,
    loss_key: PRNGKeyArray,
) -> jax.Array:
    """Evaluate a single model on a batch of data.

    Parameters
    ----------
    model : PyTree
        Model to evaluate.
    batch_state : BatchState
        Batch state for data sampling.
    dataset : dict[str, jax.Array]
        Full dataset.
    loss_fn : Callable
        Loss function.
    pass_rng : bool
        Whether to pass RNG key to the loss function.
    batch_key : PRNGKeyArray
        PRNG key for batch sampling.
    loss_key : PRNGKeyArray
        PRNG key for the loss function.

    Returns
    -------
    jax.Array
        Loss value.
    """
    batch_idxs, _ = batch_state.next(batch_key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
    if pass_rng:
        return loss_fn(model, key=loss_key, **sample)
    else:
        return loss_fn(model, **sample)


evaluate = eqx.filter_vmap(
    _evaluate_one, in_axes=(0, None, None, None, None, None, 0)
)
