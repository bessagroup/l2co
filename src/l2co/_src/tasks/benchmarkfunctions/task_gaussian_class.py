from functools import partial
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as random
import jax.random as jrd
import jax.tree_util as jtu

from l2co import Task

from .loss_fn import mean_categorical_cross_entropy_loss_fn_l2
from .models import mlp

# =============================================================================


def generate_one_gaussian_dataset(
    num_gaussians=4,
        num_samples_per_gaussian=25, dim=2, *, key: jrd.PRNGKey):

    # total_samples = num_gaussians * num_samples_per_gaussian

    keys = random.split(key, num_gaussians * 2 + 1)
    dataset_key = keys[0]

    mean_keys = keys[1:1+num_gaussians]
    cov_keys = keys[1+num_gaussians:1+2*num_gaussians]
    label_key = keys[-1]

    # Random means in given dimension
    means = jnp.stack([random.normal(k, (dim,)) * 5.0 for k in mean_keys])
    covs = jnp.stack([random.uniform(k, (dim, dim)) for k in cov_keys])
    # Ensure positive semi-definite covariance
    covs = covs @ jnp.transpose(covs, (0, 2, 1))

    # Assign labels to Gaussians ensuring at least one of each label
    gaussian_labels = random.choice(
        label_key, jnp.array([0, 1]), shape=(num_gaussians,))
    if jnp.all(gaussian_labels == 0) or jnp.all(gaussian_labels == 1):
        gaussian_labels = gaussian_labels.at[0].set(1 - gaussian_labels[0])

    samples = []
    labels = []

    for i in range(num_gaussians):
        gaussian_key = random.fold_in(dataset_key, i)
        points = random.multivariate_normal(
            gaussian_key, means[i], covs[i], (num_samples_per_gaussian,))
        samples.append(points)
        labels.append(
            jnp.full((num_samples_per_gaussian,), gaussian_labels[i]))

    samples = jnp.vstack(samples)
    labels = jnp.concatenate(labels)

    return {'x': samples, 'y': labels}

# =========================================================================================


def save_dataset(dataset: Dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def create_gaussian_task(
        seed: int,
        dataset_path: str,
        num_gaussians: int,
        num_samples_per_gaussian: int,
        dim_points: int,
        l2_regularization: float,
) -> Task:
    tag = {}

    # add the seed to the tags (this is the identifier for the task)
    tag['seed'] = seed

    # add '_seed' to dataset_path
    _path = Path(dataset_path + f'_{seed}').with_suffix('.npz')

    key = jrd.key(int(seed))

    model_key, dataset_key, batch_key = jrd.split(key, 3)

    tag['task_name'] = 'gaussian_classification'
    tag['num_gaussians'] = num_gaussians
    tag['num_samples_per_gaussian'] = num_samples_per_gaussian
    tag['dim_points'] = dim_points

    model, model_tags = mlp(
        in_size=dim_points, out_size=2, hidden_size=2,
        num_layers=2, hidden_activation=jax.nn.relu,
        output_activation=jax.nn.softmax, key=model_key)

    # Count trainable parameters
    num_params = sum(p.size for p in jtu.tree_leaves(
        model) if isinstance(p, jnp.ndarray))
    tag["dimensionality"] = num_params

    tag.update(model_tags)

    if not _path.exists():
        dataset = generate_one_gaussian_dataset(
            num_gaussians=num_gaussians,
            num_samples_per_gaussian=num_samples_per_gaussian,
            dim=dim_points,
            key=dataset_key)
        save_dataset(dataset, _path)

    loss_fn = partial(mean_categorical_cross_entropy_loss_fn_l2,
                      l2_lambda=l2_regularization)

    tag['loss_fn'] = 'cross_entropy_loss'
    tag['l2_regularization'] = l2_regularization

    return Task(
        model=model, loss_fn=loss_fn,
        dataset={'dataset_path': _path,
                 'batch_size': 32,
                 'seed': seed},
        tag=tag)

# =============================================================================
