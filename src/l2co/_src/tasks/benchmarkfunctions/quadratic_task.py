from functools import partial
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import jax.random as jr

from l2co import Task


def generate_dataset(num_samples_quadratic: int, *, key: jr.PRNGKey):
    key_W, key_y = jr.split(key)
    Ws = jr.normal(key_W, shape=(num_samples_quadratic, 10, 10))
    ys = jr.normal(key_y, shape=(num_samples_quadratic, 10))
    return {'W': Ws, 'y': ys}


def save_dataset(dataset: Dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)

# =============================================================================


def compute_quadratic_loss(model, W, y):
    residual = jnp.dot(W, model) - y
    return jnp.sum(residual ** 2)


def create_quadratic_task(
        seed: int, num_samples_quadratic: int,
) -> Task:
    tag = {}
    tag['seed'] = seed

    key = jr.key(int(seed))

    model_key, dataset_key, batch_key = jr.split(key, 3)

    tag['task_name'] = 'quadratic_functions'
    tag['num_samples_quadratic'] = num_samples_quadratic

    model = jnp.zeros(10,)

    tag['dimensionality'] = 10

    dataset = generate_dataset(
        num_samples_quadratic=num_samples_quadratic,
        key=dataset_key)

    loss_fn = partial(compute_quadratic_loss, W=dataset['W'], y=dataset['y'])

    return Task(
        model=model,
        loss_fn=loss_fn,
        dataset=None,
        tag=tag
    )
