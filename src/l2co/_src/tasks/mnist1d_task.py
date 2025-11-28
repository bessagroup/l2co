from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from mnist1d.data import get_dataset_args, make_dataset

from ..task import Task
from .loss_fn import mean_categorical_cross_entropy_loss_fn
from .models import mlp


def mnist1d_dataset(dataset_size, seed: int):
    defaults = get_dataset_args()
    defaults.num_samples = dataset_size
    defaults.seed = seed
    data = make_dataset(defaults)
    return {"x": data["x"], "y": data["y"]}


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def create_mnist1d_task(
    dataset_path: str, dataset_size: int, seed: int, batch_size: int
) -> Task:
    tag = {"task_name": "mnist1d"}
    tag["seed"] = seed

    if not Path(dataset_path).with_suffix(".npz").exists():
        dataset = mnist1d_dataset(dataset_size=dataset_size, seed=seed)
        save_dataset(dataset, dataset_path)

    model, model_tags = mlp(
        in_size=40, out_size=10, hidden_size=16, num_layers=2, key=jr.key(seed)
    )

    # Count trainable parameters
    num_params = sum(
        p.size for p in jtu.tree_leaves(model) if isinstance(p, jnp.ndarray)
    )
    tag["dimensionality"] = num_params

    # Loss fn
    loss_fn = mean_categorical_cross_entropy_loss_fn

    tag.update({"loss_fn": "mean_categorical_cross_entropy_loss_fn"})

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
