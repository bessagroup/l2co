"""
Module for batching data for optimization processes
"""

#                                                                       Modules
# =============================================================================

# Standard
import math
from typing import Dict, Iterator, Optional

# Third-party
import grain.python as grain
import jax.numpy as jnp
import jax.tree_util as jtu

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class DictSource:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class NoneIterator:
    def __init__(self, data: Optional[Dict[str, jnp.ndarray]] = None):
        if data is None:
            data = {}

        self.data = data

    def __iter__(self):
        return self

    def __next__(self):
        return self.data

# =============================================================================


def batch_iter(dataset: Optional[Dict[str, jnp.ndarray]],
               n_evals: int, batch_size: Optional[int], *,
               key: Optional[int]) -> Iterator:

    # Check if empty dictionary or None value
    if not dataset or not batch_size:
        return NoneIterator(dataset)

    if key is None:
        raise ValueError((
            f"Key must be provided when dataset is not None"
            f"{key=}, {dataset=}")
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
        seed=int(key))

    return iter(grain.DataLoader(
        data_source=source,
        operations=[transformation],
        sampler=index_sampler,
    ))


def unroll_batches(data_iter: Iterator, inner_steps: int):
    # Collect samples while preserving dictionary structure
    data = [next(data_iter) for _ in range(inner_steps)]

    if not data:
        return {}, {}

    # Stack the batch data directly
    samples = jtu.tree_map(
        lambda *vals: jnp.stack(vals, axis=0), *data)

    return samples
