# Local
import math
from pathlib import Path

# Third-party
import jax.numpy as jnp
import jax.random as jrd
import jax.tree_util as jtu

# Local
from ..task import Task
from .loss_fn import mse_loss_fn
from .models import rnn

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

    return {"x": x, "y": y}


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def create_spiral_task(
    hidden_size: int,
    dataset_size: int,
    dataset_path: str,
    seed: int,
    batch_size: int,
) -> Task:
    key = jrd.key(int(seed))

    model_key, dataset_key, batch_key = jrd.split(key, 3)

    tag = {"task_name": "spiral"}

    # model
    model, model_tags = rnn(
        in_size=2, out_size=1, hidden_size=hidden_size, key=model_key
    )

    # Count trainable parameters
    num_params = sum(
        p.size for p in jtu.tree_leaves(model) if isinstance(p, jnp.ndarray)
    )
    tag["dimensionality"] = num_params

    tag.update(model_tags)

    if not Path(dataset_path).with_suffix(".npz").exists():
        dataset = spiral_dataset(dataset_size=dataset_size, key=key)
        save_dataset(dataset, dataset_path)

    # loss function
    loss_fn = mse_loss_fn

    tag.update({"loss_fn": "mse"})

    return Task(
        model=model,
        loss_fn=loss_fn,
        dataset={
            "dataset_path": dataset_path,
            "batch_size": batch_size,
            "seed": seed,
        },
        tag=tag,
    )
