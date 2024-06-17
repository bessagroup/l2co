import logging
from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from flax.core import freeze, unfreeze
from flax.training import train_state
from tensorflow.keras.datasets import mnist

RESOLUTION = 28

OPTIMIZERS_MAPPING = {
    'adam': optax.adam,
    'sgd': optax.sgd,
    'amsgrad': optax.amsgrad
}


# Step 1: Define the model using Flax
class SimpleMLP(nn.Module):
    n_features: int
    n_layers: int
    n_labels: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input

        for _ in range(self.n_layers):
            x = nn.Dense(features=self.n_features)(x)
            x = nn.relu(x)

        x = nn.Dense(features=self.n_labels)(x)
        return nn.log_softmax(x)


# Step 2: Initialize and set weights manually
def create_initial_params(rng, input_shape, model: nn.Module):
    # init_rng, dropout_rng = jax.random.split(rng)
    dummy_input = jnp.ones(input_shape, jnp.float32)
    # variables = model.init({'params': rng, 'dropout': dropout_rng}, dummy_input)
    variables = model.init({'params': rng}, dummy_input)

    return variables['params']


# Manually initialize weights (this example uses random initialization)
def create_weights(shape, rng):
    return jax.random.normal(rng, shape) * 0.01


def initialize_params(params, model, rng):
    params = unfreeze(params)

    # Initialize weights for each layer
    for i in range(model.n_layers):
        layer_key = f'Dense_{i}'
        input_dim = RESOLUTION * RESOLUTION if i == 0 else model.n_features
        params[layer_key]['kernel'] = create_weights(
            (input_dim, model.n_features), rng)
        params[layer_key]['bias'] = create_weights((model.n_features,), rng)

    # Initialize weights for the output layer
    output_layer_key = f'Dense_{model.n_layers}'
    params[output_layer_key]['kernel'] = create_weights(
        (model.n_features, model.n_labels), rng)
    params[output_layer_key]['bias'] = create_weights((model.n_labels,), rng)

    return freeze(params)


# Step 3: Define a TrainState to hold the model parameters and optimizer state
class TrainState(train_state.TrainState):
    dropout_rng: jax.random.PRNGKey

# Define the loss function


def compute_metrics(logits, labels, n_classes):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=n_classes)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

# @jax.jit


def train_step(state, batch, labels, model: SimpleMLP):
    def loss_fn(params):
        logits = model.apply({'params': params}, batch, rngs={
                             'dropout': state.dropout_rng})
        one_hot_labels = jax.nn.one_hot(labels, num_classes=model.n_labels)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_labels))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, labels, n_classes=model.n_labels)
    return state, metrics

# @jax.jit


def eval_step(state, params, batch, y_test, model: SimpleMLP):
    logits = model.apply({'params': params}, batch, rngs={
                         'dropout': state.dropout_rng})
    return compute_metrics(logits, y_test, n_classes=model.n_labels)


# =============================================================================

def mnist_loop(optimizer, model: SimpleMLP, seed: Optional[int] = None, epochs: int = 5, batch_size: int = 32, **hyperparameters) -> np.ndarray:
    if not hyperparameters:
        hyperparameters = {'learning_rate': 0.001}

    if seed is None:
        seed = np.random.randint(1e6)

    rng = jax.random.PRNGKey(seed)
    input_shape = (1, RESOLUTION, RESOLUTION)
    params = create_initial_params(rng, input_shape, model)

    # Initialize the weights manually
    params = initialize_params(params, model, rng)

    # Create the optimizer
    tx = optimizer(**hyperparameters)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=rng
    )

    # JIT-compiled versions with partial application
    jit_train_step = jax.jit(partial(train_step, model=model))
    jit_eval_step = jax.jit(partial(eval_step, model=model))

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # List to store the losses
    training_losses = []

    for epoch in range(epochs):
        logging.debug(f"Epoch {epoch+1}/{epochs}")
        # Shuffle the training data
        permutation = jax.random.permutation(rng, len(x_train))
        x_train, y_train = x_train[permutation], y_train[permutation]

        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            state, metrics = jit_train_step(state, batch_x, batch_y)

            # Store the loss
            training_losses.append(metrics['loss'])

            if i % 100 == 0:
                logging.debug(
                    f"Step {i // batch_size}: loss = {metrics['loss']:.4f}, accuracy = {metrics['accuracy']:.4f}")

    # # Evaluate the model
    # metrics = jit_eval_step(state, state.params, x_test, y_test)
    # logging.debug(f"\nTest accuracy: {metrics['accuracy']:.4f}")

    # # Print the collected training losses
    # logging.debug("Training losses over steps:", jnp.array(training_losses))
    xarr = xr.DataArray(
        np.array(training_losses),
        coords={'iterations': np.arange(len(training_losses))},
        dims=('iterations',)
    )

    xarr = xarr.expand_dims({'output_dim': ['y']})
    xarr = xarr.to_dataset(name='output')
    return xarr


def mnist_multiple_realizations(
        optimizer,
        model: SimpleMLP,
        seed: Optional[int] = None,
        epochs: int = 5,
        batch_size: int = 32,
        n_realizations: int = 10,
        optimizer_name: Optional[str] = None,
        **hyperparameters) -> jnp.ndarray:

    all_losses = []
    for i in range(n_realizations):
        logging.debug(f"Realization {i+1}/{n_realizations}")
        all_losses.append(mnist_loop(optimizer, model, seed=seed+i,
                          epochs=epochs, batch_size=batch_size, **hyperparameters))

    xarr = xr.concat(all_losses, dim=xr.DataArray(np.arange(n_realizations), dims='realization')
                     ).transpose('realization', 'iterations', 'output_dim')

    if optimizer_name is not None:
        name = optimizer_name
    else:
        name = optimizer.__name__

    return xarr.assign_coords({'optimizer': [name]})


# =============================================================================


class MNIST(DataGenerator):
    def __init__(self, config):
        self.config = config

    def execute(self):
        self.config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=self.experiment_sample)

        for optimizer in self.config.optimization.optimizers:
            loss = mnist_multiple_realizations(
                optimizer=OPTIMIZERS_MAPPING[optimizer['optimizer']],
                model=SimpleMLP(n_features=self.config.model.n_features,
                                n_layers=self.config.model.n_layers,
                                n_labels=self.config.model.n_labels),
                seed=self.config.seed,
                epochs=self.config.optimization.epochs,
                batch_size=self.config.optimization.batch_size,
                n_realizations=self.config.optimization.realizations,
                optimizer_name=optimizer['name'],
                **optimizer['hyperparameters'],
            )

            self.experiment_sample.store(
                name=optimizer['name'], object=loss, to_disk=True)

        # loss = mnist_multiple_optimizers(
        #     optimizers=[OPTIMIZERS_MAPPING[optimizer]
        #                 for optimizer in self.config.optimization.optimizers],
        #     model=SimpleMLP(n_features=self.config.model.n_features,
        #                     n_layers=self.config.model.n_layers,
        #                     n_labels=self.config.model.n_labels),
        #     seed=self.config.seed,
        #     epochs=self.config.optimization.epochs,
        #     batch_size=self.config.optimization.batch_size,
        #     n_realizations=self.config.optimization.realizations,
        #     hyperparameters=[
        #         {'learning_rate': self.config.optimization.learning_rate}
        #         for _ in range(self.config.optimization.realizations)],
        # )

        # self.experiment_sample.store(name='loss', object=loss, to_disk=True)
