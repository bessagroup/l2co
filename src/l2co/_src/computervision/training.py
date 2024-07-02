"""
Functions for training models
"""

#
#                                                                       Modules
# =============================================================================

# Standard
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

# Third-party
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from flax.training.train_state import TrainState
from optax import softmax_cross_entropy
from optax._src.base import GradientTransformation

# Local
from .data import Dataset
from .models import SimpleMLP

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def create_initial_params(rng: jax.random.KeyArray,
                          input_shape: Tuple[int, int],
                          model: nn.Module) -> Dict[str, Any]:
    dummy_input = jnp.ones(input_shape, jnp.float32)
    variables = model.init({'params': rng}, dummy_input)
    return variables['params']


def compute_metrics(logits: jnp.ndarray,
                    labels: jnp.ndarray,
                    n_classes: int) -> Dict[str, jnp.ndarray]:
    one_hot_labels = jax.nn.one_hot(labels, num_classes=n_classes)
    loss = jnp.mean(softmax_cross_entropy(logits, one_hot_labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


def train_step(state: TrainState, batch_x: jnp.ndarray,
               batch_y: jnp.ndarray,
               model: nn.Module) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:

    def loss_fn(params):
        logits = model.apply({'params': params}, batch_x)
        one_hot_labels = jax.nn.one_hot(batch_y, num_classes=model.n_labels)
        loss = jnp.mean(softmax_cross_entropy(logits, one_hot_labels))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch_y, n_classes=model.n_labels)
    return state, metrics


def evaluate_step(
        state: TrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        model: nn.Module) -> Dict[str, jnp.ndarray]:

    logits = model.apply({'params': state.params}, batch_x)
    return compute_metrics(logits, batch_y, n_classes=model.n_labels)

# =============================================================================


def training_loop(
        optimizer: GradientTransformation,
        model: nn.Module,
        data: Dataset,
        iterations: int,
        seed: Optional[int] = None,
        batch_size: int = 32,
) -> xr.Dataset:

    if seed is None:
        seed = np.random.randint(1e6)

    rng = jax.random.PRNGKey(seed)
    input_shape = data.x_train[0:1].shape  # (1, RESOLUTION, RESOLUTION, 1)
    params = create_initial_params(rng, input_shape, model)

    # Create the TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # JIT-compiled versions with partial application
    jit_train_step = jax.jit(partial(train_step, model=model))
    jit_evaluate_step = jax.jit(partial(evaluate_step, model=model))

    # List to store the losses
    training_losses, training_accuracy = [], []
    testing_losses, testing_accuracy = [], []

    for iteration, (batch_x, batch_y) in zip(
            range(iterations), data.get_batch(
                batch_size=batch_size, seed=seed)):
        logging.debug(f"Epoch {iteration+1}/{iterations}")

        state, metrics_train = jit_train_step(
            state, batch_x, batch_y)

        # Store the loss
        training_losses.append(metrics_train['loss'])
        training_accuracy.append(metrics_train['accuracy'])

        if iteration % 100 == 0:
            logging.debug((
                f"Step {iteration}: loss = {metrics_train['loss']:.4f}"
                f", accuracy = {metrics_train['accuracy']:.4f}"))

        metrics_test = jit_evaluate_step(
            state, data.x_test, data.y_test)

        testing_losses.append(metrics_test['loss'])
        testing_accuracy.append(metrics_test['accuracy'])

    return (ExperimentData(output_data=np.array(training_losses)).to_xarray(),
            ExperimentData(output_data=np.array(
                training_accuracy)).to_xarray(),
            ExperimentData(output_data=np.array(testing_losses)).to_xarray(),
            ExperimentData(output_data=np.array(testing_accuracy)).to_xarray())


def training_multiple_realizations(
        optimizer: GradientTransformation,
        model: SimpleMLP,
        data: Dataset,
        iterations: int,
        seed: Optional[int] = None,
        batch_size: int = 32,
        n_realizations: int = 10,
        optimizer_name: Optional[str] = None,) -> xr.Dataset:

    all_losses = []
    for i in range(n_realizations):
        logging.debug(f"Realization {i+1}/{n_realizations}")
        all_losses.append(
            training_loop(optimizer, model, data, seed=seed+i,
                          iterations=iterations,
                          batch_size=batch_size))

    xarr = xr.concat(
        all_losses,
        dim=xr.DataArray(
            np.arange(n_realizations), dims='realization')
    )

    if optimizer_name is not None:
        name = optimizer_name
    else:
        name = optimizer.__name__

    return xarr.assign_coords({'optimizer': [name]})


# =============================================================================
