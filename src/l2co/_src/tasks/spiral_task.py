"""Spiral dataset classification task using RNN models."""

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
    """Generate a two-class spiral dataset.

    Parameters
    ----------
    dataset_size : int
        Total number of samples (split evenly between classes).
    key : jrd.PRNGKey
        Random key for offset sampling.

    Returns
    -------
    dict[str, jnp.ndarray]
        ``'x'`` features ``(N, 16, 2)`` and ``'y'`` binary
        labels ``(N, 1)``.
    """
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
    """Save a dataset dictionary to an ``.npz`` file.

    Parameters
    ----------
    dataset : dict[str, jnp.ndarray]
        Dictionary of arrays to persist.
    path : str or Path
        Destination file path.
    """
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
    """Create a spiral binary classification task using an RNN.

    Parameters
    ----------
    hidden_size : int
        RNN hidden state size.
    dataset_size : int
        Number of spiral samples.
    dataset_path : str
        Path prefix for the dataset file.
    seed : int
        Random seed for reproducibility.
    batch_size : int
        Mini-batch size for training.

    Returns
    -------
    Task
        Configured spiral classification task.
    """
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
