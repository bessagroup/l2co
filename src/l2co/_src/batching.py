"""
Module for batching data for optimization processes
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Int, PRNGKeyArray, PyTree

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class BatchState(eqx.Module):
    """State object for iterating over a dataset in fixed-size batches.

    This class maintains the permutation of dataset indices, the current
    position in that permutation, and provides logic to return the next
    batch of indices while seamlessly handling wrap-around by generating
    a new permutation when the end of the dataset is reached.

    Attributes
    ----------
    perm : Int[Array, " N"]
        1D array containing a permutation of the dataset indices
        (shape: ``(dataset_size,)``). Determines the iteration order.
    pos : int
        Current position (start index) inside ``perm`` for the next batch.
    batch_size : int
        Number of samples per batch. Static field (not traced).

    Class Methods
    -------------
    init(key, dataset_size, batch_size)
        Construct an initial ``BatchState`` with a fresh permutation and
        position set to zero.

    Properties
    ----------
    dataset_size : int
        Total number of samples in the dataset (``perm.shape[0]``).

    Methods
    -------
    next_batch(dataset)
        Return a batch extracted from the PyTree ``dataset`` and the updated
        ``BatchState``. Wraps around and re-permutes when at end.

    Notes
    -----
    The batching logic chooses between two paths:
    ``take_no_wrap`` when the remaining portion of ``perm`` is large enough
    for a full batch, otherwise ``take_wrap`` which concatenates the tail of
    the current permutation with the head of a newly generated permutation.

    Examples
    --------
    >>> key = jr.PRNGKey(0)
    >>> state = BatchState.init(key, dataset_size=10, batch_size=4)
    >>> data = jnp.arange(10)
    >>> batch, state = state.next_batch(data)
    >>> batch.shape
    (4,)
    """

    perm: Int[Array, " N"]
    pos: int
    batch_size: int = eqx.field(static=True)
    dataset: PyTree

    @classmethod
    def init(
        cls,
        dataset: PyTree,
        batch_size: int | None,
        key: PRNGKeyArray,
    ):
        """Initialize a ``BatchState`` with a fresh permutation.

        Parameters
        ----------
        dataset : PyTree
            PyTree dataset from which to determine the dataset size.
        batch_size : int | None
            Number of items per batch.
        key : PRNGKeyArray
            PRNG key used to generate the initial permutation and to seed
            subsequent wrap-around permutations.

        Notes
        -----
        If ``dataset_size`` is ``None``, it defaults to zero. If
        ``batch_size`` is ``None``, it defaults to ``dataset_size``.

        Returns
        -------
        BatchState
            New batching state with position set to zero and ``perm`` a
            random permutation of ``range(dataset_size)``.

        Examples
        --------
        >>> key = jr.PRNGKey(0)
        >>> state = BatchState.init(key, dataset_size=10, batch_size=4)
        >>> state.pos
        0
        """

        dataset_size = jax.tree.leaves(dataset)[0].shape[0] if dataset else 0
        batch_size = batch_size if batch_size is not None else dataset_size

        return cls(
            perm=jr.permutation(key, dataset_size),
            pos=0,
            batch_size=batch_size,
            dataset=dataset,
        )

    @property
    def dataset_size(self) -> int:
        """Total number of samples in the dataset.

        Returns
        -------
        int
            Size of the dataset, equal to ``perm.shape[0]``.
        """
        return self.perm.shape[0]

    def next(self, key: PRNGKeyArray) -> tuple[PyTree, BatchState]:
        """Return a batch from a PyTree dataset and the updated state.

        Parameters
        ----------
        key : PRNGKeyArray
            PRNG key used to generate a new permutation if wrap-around occurs.

        Returns
        -------
        tuple[PyTree, BatchState]
            A tuple ``(batch, new_state)`` where ``batch`` mirrors the
            structure of ``dataset`` and each leaf has leading dimension
            ``batch_size``; ``new_state`` reflects the advanced position or a
            re-permuted state on wrap-around.

        Notes
        -----
        Uses ``jax.lax.cond`` to choose between no-wrap and wrap-around paths,
        and ``jax.tree.map`` to index into the PyTree leaves.

        Examples
        --------
        >>> key = jr.PRNGKey(0)
        >>> state = BatchState.init(key, dataset_size=10, batch_size=4)
        >>> data = {"x": jnp.arange(10)}
        >>> batch, state = state.next_batch(data)
        >>> jax.tree.map(lambda x: x.shape, batch)
        {'x': (4,)}
        """
        new_state, batch_idxs = jax.lax.cond(
            (self.pos + self.batch_size) <= self.dataset_size,
            take_no_wrap,
            take_wrap,
            (self, key),
        )
        # Index into the PyTree
        batch = jax.tree.map(lambda x: x[batch_idxs], self.dataset)
        return batch, new_state


def take_no_wrap(carry: tuple[BatchState, PRNGKeyArray]):
    """Return next batch indices without wrap-around.

    Extracts ``batch_size`` consecutive indices from ``state.perm`` starting
    at ``state.pos`` and advances the position by ``batch_size``. No new
    permutation is generated.

    Parameters
    ----------
    state : BatchState
        Current batching state.
    key : PRNGKeyArray
        PRNG key used to generate a new permutation.

    Returns
    -------
    tuple[BatchState, jax.Array]
        A tuple ``(new_state, batch_idxs)`` where ``new_state`` has an
        advanced ``pos`` and ``batch_idxs`` is a 1D array of length
        ``batch_size`` containing dataset indices.
    """
    state, _ = carry
    batch = jax.lax.dynamic_slice(
        state.perm,
        (state.pos,),
        (state.batch_size,),
    )
    new_state = BatchState(
        perm=state.perm,
        pos=state.pos + state.batch_size,
        batch_size=state.batch_size,
        dataset=state.dataset,
    )
    return new_state, batch


def take_wrap(carry: tuple[BatchState, PRNGKeyArray]):
    """Return next batch indices with wrap-around and re-permutation.

    When the remaining portion of ``state.perm`` is insufficient for a full
    batch, this function takes the tail from the current permutation and the
    head from a freshly generated permutation to form a complete batch. It
    also updates the PRNG key and resets the position within the new
    permutation.

    Parameters
    ----------
    state : BatchState
        Current batching state.
    key : PRNGKeyArray
        PRNG key used to generate a new permutation.

    Returns
    -------
    tuple[BatchState, jax.Array]
        A tuple ``(new_state, batch_idxs)`` where ``new_state`` contains the
        split key, a new permutation, and the position set to the number of
        wrapped elements; ``batch_idxs`` is a 1D array of length
        ``batch_size`` containing dataset indices.

    Notes
    -----
    The returned indices are constructed by concatenating the remaining tail
    of ``state.perm`` with the head of the new permutation (conceptually).
    """
    state, key = carry
    # Indices for current batch
    idx = jnp.arange(state.batch_size)
    mask = idx < (state.dataset_size - state.pos)

    # Slice the remaining elements from current permutation
    raw = jax.lax.dynamic_slice(state.perm, (state.pos,), (state.batch_size,))

    # Generate new permutation for wrap-around
    new_perm = jr.permutation(key, state.dataset_size)
    second = jax.lax.dynamic_slice(new_perm, (0,), (state.batch_size,))

    # Combine slices using mask: first valid part from raw, rest from second
    batch = jnp.where(mask[::-1], raw, second)

    # Compute new position in new permutation
    new_pos = state.batch_size - (state.dataset_size - state.pos)

    new_state = BatchState(
        perm=new_perm,
        pos=new_pos,
        batch_size=state.batch_size,
        dataset=state.dataset,
    )
    return new_state, batch
