from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import PyTree

from ..task import Task
from .task_gaussian_class import generate_one_gaussian_dataset, save_dataset


def scale_adam_params(unit_params: jax.Array) -> PyTree:

    learning_rate = 10**(-5 + unit_params[0] * (2 - (-5)))

    # Scale b1 and b2 linearly between 0.85 and 1.0
    b1 = 0.85 + unit_params[1] * (1.0 - 0.85)
    b2 = 0.85 + unit_params[2] * (1.0 - 0.85)

    return {
        "learning_rate": learning_rate,
        "b1": b1,
        "b2": b2,
    }


def create_gaussian_meta_task(
    seed: int, dataset_path: str,
        num_gaussians: int, num_samples_per_gaussian: int, dim_points: int,
        inner_steps: int, l2_regularization: float,
) -> Task:

    tag = {}
    tag['task_name'] = 'gaussian_classification_meta'
    tag['num_gaussians'] = num_gaussians
    tag['num_samples_per_gaussian'] = num_samples_per_gaussian
    tag['dim_points'] = dim_points
    tag['seed'] = seed

    key = jr.key(seed)
    model_key, data_key = jr.split(key, 2)

    outer_model = jnp.array([0.0, 0.0, 0.0])
    tag['dimensionality'] = 3

    _path = Path(dataset_path + f'_{seed}').with_suffix('.npz')

    if not _path.exists():
        dataset = generate_one_gaussian_dataset(
            num_gaussians=num_gaussians,
            num_samples_per_gaussian=num_samples_per_gaussian,
            dim=dim_points, key=data_key
        )

        save_dataset(dataset, _path)

    def loss_fn(model, x, y, l2_lambda=l2_regularization):
        # Predict
        y_pred = jax.nn.softmax(jax.vmap(model)(x), axis=-1)

        # Cross-entropy loss
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(
            y_pred, y).mean()

        params, _ = eqx.partition(model, eqx.is_array_like)

        # L2 regularization: only apply to parameters that are arrays
        l2_penalty = sum(jnp.sum(jnp.square(p))
                         for p in jax.tree_util.tree_leaves(params))
        loss += l2_lambda * l2_penalty

        return loss

    def inner_loop(outer_x: float, key: jax.Array, x: jax.Array, y: jax.Array,
                   inner_steps: int):
        model = eqx.nn.MLP(
            in_size=2,
            out_size=2,
            width_size=2,
            depth=2,
            activation=jnn.relu,
            key=key
        )

        adam_params = scale_adam_params(outer_x)

        optimizer = optax.adam(learning_rate=adam_params['learning_rate'],
                               b1=adam_params['b1'],
                               b2=adam_params['b2'])

        params, static = eqx.partition(model, eqx.is_array_like)
        opt_state = optimizer.init(params)

        def update_step(_, carry):
            inner_params, opt_state, _ = carry
            model = eqx.combine(inner_params, static)
            loss, grads = eqx.filter_value_and_grad(
                loss_fn)(model, x=x, y=y)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = eqx.apply_updates(inner_params, updates)

            return new_params, opt_state, loss

        _, _, final_loss = jax.lax.fori_loop(
            lower=0, upper=inner_steps,
            body_fun=update_step,
            init_val=(params, opt_state, jnp.array(jnp.inf)))

        return final_loss

    return Task(
        model=outer_model,
        loss_fn=partial(inner_loop, key=model_key, inner_steps=inner_steps),
        dataset={'dataset_path': _path,
                 'batch_size': None,
                 'seed': None,
                 },
        tag=tag,)
