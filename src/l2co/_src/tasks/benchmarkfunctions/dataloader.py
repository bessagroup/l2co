import math
from typing import Dict, Iterator, Optional

import grain.python as grain
import jax.numpy as jnp

# =============================================================================


class DictSource:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class NoneIterator:
    def __iter__(self):
        return self

    def __next__(self):
        return {}


# =============================================================================
def batch_iter(dataset: Optional[Dict[str, jnp.ndarray]],
               n_evals: int, batch_size: Optional[int], *,
               key: int) -> Iterator:

    if dataset is None:
        return NoneIterator()

    source = DictSource(dataset)

    if batch_size is None:
        batch_size = len(dataset)

    transformation = grain.Batch(batch_size=batch_size, drop_remainder=True)

    num_records = len(source)
    num_epochs = math.ceil((n_evals * batch_size) / num_records)

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
