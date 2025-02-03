import math
from typing import Dict, Iterator, Optional

import jax.numpy as jnp
import jax.random as jrd

# =============================================================================


def batch_iter(dataset: Optional[Dict[str, jnp.ndarray]], batch_size: int, *,
               key: jrd.PRNGKey) -> Iterator:
    if dataset is None:
        return NoneIterator()

    dataset_size = next(iter(dataset.values())).shape[0]

    def shuffle_data(dataset, key):
        indices = jrd.permutation(key, dataset_size)
        for k, v in dataset.items():
            dataset[k] = v[indices]
        return dataset

    def batch_generator(key):
        while True:
            shuffled_data = shuffle_data(dataset, key)
            key, subkey = jrd.split(key)
            for i in range(0, dataset_size, batch_size):
                batch = {k: v[i:i + batch_size]
                         for k, v in shuffled_data.items()}
                # Incomplete batch
                if batch[next(iter(batch.keys()))].shape[0] < batch_size:
                    extra_needed = batch_size - \
                        batch[next(iter(batch.keys()))].shape[0]
                    extra_indices = jnp.arange(0, extra_needed)
                    batch = {
                        k: jnp.concatenate(
                            [v[i:i + batch_size], v[extra_indices]], axis=0)
                        for k, v in shuffled_data.items()
                    }
                yield batch

    return batch_generator(key)


class NoneIterator:
    def __iter__(self):
        return self

    def __next__(self):
        return {}

# =============================================================================


def spiral_dataset(dataset_size, *, key: jrd.PRNGKey):
    t = jnp.linspace(0, 2 * math.pi, 16)
    offset = jrd.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return {'x': x, 'y': y}
