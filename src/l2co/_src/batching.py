"""
Module for batching data for optimization processes
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import math
from collections.abc import Iterator

# Third-party
import equinox as eqx
import grain.python as grain
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


class DictSource:
    """
    A data source wrapper for dictionary-based datasets.

    Parameters
    ----------
    data : dict
        Dictionary containing dataset arrays.

    Methods
    -------
    __len__()
        Returns the number of records in the dataset.
    __getitem__(idx)
        Retrieves a specific record by index.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class NoneIterator:
    """
    An iterator that always returns the same data.

    Parameters
    ----------
    data : Optional[Dict[str, jnp.ndarray]], optional
        Data to return on each iteration, by default an empty dictionary.

    Methods
    -------
    __iter__()
        Returns the iterator itself.
    __next__()
        Returns the stored data.
    """

    def __init__(self, data: dict[str, jnp.ndarray] | None = None):
        if data is None:
            data = {}

        self.data = data

    def __iter__(self):
        return self

    def __next__(self):
        return self.data


# =============================================================================


def batch_iter(
    dataset: dict[str, jnp.ndarray] | None,
    n_evals: int,
    batch_size: int | None,
    *,
    key: int | None,
) -> Iterator:
    """
    Create an iterator for batching data.

    Parameters
    ----------
    dataset : Optional[Dict[str, jnp.ndarray]]
        Dataset to batch, where each key corresponds to a feature.
    n_evals : int
        Number of evaluations to perform.
    batch_size : Optional[int]
        Size of each batch.
    key : Optional[int]
        Random seed for shuffling.

    Returns
    -------
    Iterator
        Iterator that yields batches of data.

    Raises
    ------
    ValueError
        If `key` is None when `dataset` is provided.
    """

    # Check if empty dictionary or None value
    if not dataset or not batch_size:
        return NoneIterator(dataset)

    if key is None:
        raise ValueError(
            f"Key must be provided when dataset is not None{key=}, {dataset=}"
        )

    source = DictSource(dataset)

    transformation = grain.Batch(batch_size=batch_size, drop_remainder=True)

    num_records = len(source)
    # Note: Add one since we are doing one scan more
    num_epochs = math.ceil(((n_evals + 1) * batch_size) / num_records)

    index_sampler = grain.IndexSampler(
        num_records=num_records,
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        shuffle=True,
        seed=int(key),
    )

    return iter(
        grain.DataLoader(
            data_source=source,
            operations=[transformation],
            sampler=index_sampler,
        )
    )


# =============================================================================


class BatcherState(eqx.Module):
    """State object for iterating over a dataset in fixed-size batches.

    This class maintains the permutation of dataset indices, the current
    position in that permutation, and provides logic to return the next
    batch of indices while seamlessly handling wrap-around by generating
    a new permutation when the end of the dataset is reached.

    Attributes
    ----------
    key : PRNGKeyArray
        JAX pseudo-random number generator key used to (re)generate
        permutations for wrap-around batches.
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
        Construct an initial ``BatcherState`` with a fresh permutation and
        position set to zero.

    Properties
    ----------
    dataset_size : int
        Total number of samples in the dataset (``perm.shape[0]``).

    Methods
    -------
    next_batch(dataset)
        Return a batch extracted from the PyTree ``dataset`` and the updated
        ``BatcherState``. Wraps around and re-permutes when at end.

    Notes
    -----
    The batching logic chooses between two paths:
    ``take_no_wrap`` when the remaining portion of ``perm`` is large enough
    for a full batch, otherwise ``take_wrap`` which concatenates the tail of
    the current permutation with the head of a newly generated permutation.

    Examples
    --------
    >>> key = jr.PRNGKey(0)
    >>> state = BatcherState.init(key, dataset_size=10, batch_size=4)
    >>> data = jnp.arange(10)
    >>> batch, state = state.next_batch(data)
    >>> batch.shape
    (4,)
    """

    key: PRNGKeyArray
    perm: Int[Array, " N"]
    pos: int
    batch_size: int = eqx.field(static=True)

    @classmethod
    def init(cls, key: PRNGKeyArray, dataset_size: int, batch_size: int):
        """Initialize a ``BatcherState`` with a fresh permutation.

        Parameters
        ----------
        key : PRNGKeyArray
            PRNG key used to generate the initial permutation and to seed
            subsequent wrap-around permutations.
        dataset_size : int
            Total number of items in the dataset.
        batch_size : int
            Number of items per batch.

        Returns
        -------
        BatcherState
            New batching state with position set to zero and ``perm`` a
            random permutation of ``range(dataset_size)``.

        Examples
        --------
        >>> key = jr.PRNGKey(0)
        >>> state = BatcherState.init(key, dataset_size=10, batch_size=4)
        >>> state.pos
        0
        """
        return cls(
            key=key,
            perm=jr.permutation(key, dataset_size),
            pos=0,
            batch_size=batch_size,
        )

    @property
    def dataset_size(self):
        """Total number of samples in the dataset.

        Returns
        -------
        int
            Size of the dataset, equal to ``perm.shape[0]``.
        """
        return self.perm.shape[0]

    def next_batch(self, dataset: PyTree) -> tuple[PyTree, BatcherState]:
        """Return a batch from a PyTree dataset and the updated state.

        Parameters
        ----------
        dataset : PyTree
            PyTree of arrays from which to take a batch. All leaves must be
            indexable by a 1D integer array of indices and have leading
            dimension equal to ``dataset_size``.

        Returns
        -------
        tuple[PyTree, BatcherState]
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
        >>> state = BatcherState.init(key, dataset_size=10, batch_size=4)
        >>> data = {"x": jnp.arange(10)}
        >>> batch, state = state.next_batch(data)
        >>> jax.tree.map(lambda x: x.shape, batch)
        {'x': (4,)}
        """
        new_state, batch_idxs = jax.lax.cond(
            (self.pos + self.batch_size) <= self.dataset_size,
            take_no_wrap,
            take_wrap,
            self,
        )
        # Index into the PyTree
        batch = jax.tree.map(lambda x: x[batch_idxs], dataset)
        return batch, new_state


def take_no_wrap(state: BatcherState):
    """Return next batch indices without wrap-around.

    Extracts ``batch_size`` consecutive indices from ``state.perm`` starting
    at ``state.pos`` and advances the position by ``batch_size``. No new
    permutation is generated.

    Parameters
    ----------
    state : BatcherState
        Current batching state.

    Returns
    -------
    tuple[BatcherState, jax.Array]
        A tuple ``(new_state, batch_idxs)`` where ``new_state`` has an
        advanced ``pos`` and ``batch_idxs`` is a 1D array of length
        ``batch_size`` containing dataset indices.
    """
    batch = jax.lax.dynamic_slice(
        state.perm,
        (state.pos,),
        (state.batch_size,),
    )
    new_state = BatcherState(
        key=state.key,
        perm=state.perm,
        pos=state.pos + state.batch_size,
        batch_size=state.batch_size,
    )
    return new_state, batch


def take_wrap(state: BatcherState):
    """Return next batch indices with wrap-around and re-permutation.

    When the remaining portion of ``state.perm`` is insufficient for a full
    batch, this function takes the tail from the current permutation and the
    head from a freshly generated permutation to form a complete batch. It
    also updates the PRNG key and resets the position within the new
    permutation.

    Parameters
    ----------
    state : BatcherState
        Current batching state.

    Returns
    -------
    tuple[BatcherState, jax.Array]
        A tuple ``(new_state, batch_idxs)`` where ``new_state`` contains the
        split key, a new permutation, and the position set to the number of
        wrapped elements; ``batch_idxs`` is a 1D array of length
        ``batch_size`` containing dataset indices.

    Notes
    -----
    The returned indices are constructed by concatenating the remaining tail
    of ``state.perm`` with the head of the new permutation (conceptually).
    """
    # Indices for current batch
    idx = jnp.arange(state.batch_size)
    mask = idx < (state.dataset_size - state.pos)

    # Slice the remaining elements from current permutation
    raw = jax.lax.dynamic_slice(state.perm, (state.pos,), (state.batch_size,))

    # Generate new permutation for wrap-around
    key1, key2 = jr.split(state.key)
    new_perm = jr.permutation(key1, state.dataset_size)
    second = jax.lax.dynamic_slice(new_perm, (0,), (state.batch_size,))

    # Combine slices using mask: first valid part from raw, rest from second
    batch = jnp.where(mask[::-1], raw, second)

    # Compute new position in new permutation
    new_pos = state.batch_size - (state.dataset_size - state.pos)

    new_state = BatcherState(
        key=key2,
        perm=new_perm,
        pos=new_pos,
        batch_size=state.batch_size,
    )
    return new_state, batch
