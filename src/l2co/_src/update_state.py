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
    """State container for optimizer updates.

    Stores the current and best parameters, losses, and step information
    for the optimization process.

    Attributes
    ----------
    last_params : Float[Array, "memory_size dim"]
        Last evaluated parameter values.
    last_loss : Float[Array, " memory size"]
        Last evaluated loss values.
    best_params : Float[Array, " dim"]
        Best parameters found so far.
    best_loss : Float[Array, 1]
        Best loss value found so far.
    last_step : Int[Array, 1]
        Last update step number.
    """

    last_params: Float[Array, "memory_size dim"]
    last_loss: Float[Array, " memory size"]
    best_params: Float[Array, " dim"]
    best_loss: Float[Array, 1]
    last_step: Int[Array, 1]

    @property
    def memory_size(self) -> int:
        """Get the memory size (number of stored parameter sets).

        Returns
        -------
        int
            Memory size.
        """
        # Assumes all leaves have same leading dimension
        return jax.tree.leaves(self.last_params)[0].shape[0]

    @classmethod
    def init(
        cls,
        params: InputParameters,
        memory_size: int,
    ) -> UpdateState:
        """Initialize a new UpdateState.

        Parameters
        ----------
        params : InputParameters
            Initial parameter values.
        memory_size : int
            Number of parameter sets to store in memory.

        Returns
        -------
        UpdateState
            Initialized update state.
        """
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
        """Reset state to best parameters found so far.

        Returns
        -------
        UpdateState
            Reset update state.
        """
        return self.init(params=self.best_params, memory_size=self.memory_size)

    @classmethod
    def from_history(
        cls, history: OptHistoryType, memory_size: int, last_step: int
    ) -> UpdateState:
        """Create UpdateState from optimization history.

        Parameters
        ----------
        history : OptHistoryType
            Optimization history to extract state from.
        memory_size : int
            Number of recent entries to retain in memory.
        last_step : int
            Last update step number.

        Returns
        -------
        UpdateState
            Update state initialized from history.
        """
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
        self,
        batch_state: BatchState,
        static: PyTree,
        dataset: dict[str, jax.Array],
        loss_fn: Callable,
        sampler: Callable,
        key: PRNGKeyArray,
    ) -> UpdateState:
        """Sample new parameters and evaluate them.

        Parameters
        ----------
        batch_state : BatchState
            Current batch state for data sampling.
        static : PyTree
            Static (non-trainable) parts of the model.
        dataset : dict[str, jax.Array]
            Dataset for evaluation.
        loss_fn : Callable
            Loss function for evaluation.
        sampler : Callable
            Function to sample new parameters.
        key : PRNGKeyArray
            PRNG key for random operations.

        Returns
        -------
        UpdateState
            Updated state with newly sampled and evaluated parameters.
        """
        n_samples = self.memory_size

        # Sample new parameters using the provided sampler
        new_params = sampler(key, self.best_params, n_samples)

        losses = evaluate(
            eqx.combine(new_params, static),
            batch_state,
            dataset,
            loss_fn,
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
        """Merge with another UpdateState, keeping the better parameters.

        Parameters
        ----------
        other : UpdateState
            Other update state to merge with.

        Returns
        -------
        UpdateState
            Merged update state with best parameters from both states.
        """
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
        """Evaluate parameters and compute gradients.

        Parameters
        ----------
        task : Task
            Task to evaluate against.
        key : PRNGKeyArray
            PRNG key for random operations.

        Returns
        -------
        tuple[jax.Array, PyTree]
            Loss values and gradients for the last parameters.
        """
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
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    jax.Array
        Loss value.
    """
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
