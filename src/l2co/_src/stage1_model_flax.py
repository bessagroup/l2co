"""
Flax based neural network implementations
"""

#                                                                       Modules
# =============================================================================
# Standard
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

# Third-party
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from flax.training.train_state import TrainState
from optax import GradientTransformation, softmax_cross_entropy

# Local
from l2co._src.computervision.data import Dataset
from l2co.data import DataLoader

#                                                                        Models
# =============================================================================


class SimpleLSTM(nn.Module):
    features: int
    n_labels: int

    def setup(self):
        self.scan_cell = nn.transforms.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast='params',
            in_axes=1, out_axes=1,
            split_rngs={'params': False})(self.features)

    @nn.compact
    def __call__(self, x):
        carry = self.scan_cell.initialize_carry(
            jax.random.key(0), x[:, 0].shape)
        x = self.scan_cell(carry, x)[1]  # only return the output

        # Flatten the output if needed (taking the last output in the sequence)
        x = x.reshape((x.shape[0], -1))  # flatten

        # Add a dense layer to map to n_labels
        x = nn.Dense(features=self.n_labels)(x)

        return nn.sigmoid(x)

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
    # one_hot_labels = jax.nn.one_hot(labels, num_classes=n_classes)
    loss = jnp.mean(softmax_cross_entropy(logits, labels))
    accuracy = jnp.mean(jnp.argmin(logits, -1) == jnp.argmin(labels, -1))
    return {'loss': loss, 'accuracy': accuracy}

#                                                                Loss functions
# =============================================================================


def loss_fn_classification(params, model, batch_x, batch_y):
    logits = model.apply({'params': params}, batch_x)
    one_hot_labels = jax.nn.one_hot(batch_y, num_classes=model.n_labels)
    loss = jnp.mean(softmax_cross_entropy(logits, one_hot_labels))
    return loss, logits


def loss_fn_regression(params, model, batch_x, batch_y):
    logits = model.apply({'params': params}, batch_x)
    loss = jnp.mean(softmax_cross_entropy(logits, batch_y))
    return loss, logits

# =============================================================================


def train_step(
    state: TrainState, batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        model: nn.Module,
        loss_fn=loss_fn_classification
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:

    _loss_fn = partial(loss_fn, model=model, batch_x=batch_x, batch_y=batch_y)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
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
        seed: Optional[int],
        batch_size: int,
        loss_fn: Callable,
        track_loss: bool,
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
    jit_train_step = jax.jit(partial(train_step, model=model, loss_fn=loss_fn))
    jit_evaluate_step = jax.jit(partial(evaluate_step, model=model))

    if track_loss:
        # List to store the losses
        metrics = {'training_losses': [], 'training_accuracy': [],
                   'testing_losses': [], 'testing_accuracy': []}

    else:
        metrics = None

    for iteration, (batch_x, batch_y) in zip(
            range(iterations), data.get_batch(
                batch_size=batch_size, seed=seed)):
        logging.debug(f"Epoch {iteration+1}/{iterations}")

        state, metrics_train = jit_train_step(
            state, batch_x, batch_y)

        if track_loss:
            # Store the loss
            metrics['training_losses'].append(metrics_train['loss'])
            metrics['training_accuracy'].append(metrics_train['accuracy'])

        # if iteration % 100 == 0:
        #     logging.debug((
        #         f"Step {iteration}: loss = {metrics_train['loss']:.4f}"
        #         f", accuracy = {metrics_train['accuracy']:.4f}"))

            metrics_test = jit_evaluate_step(
                state, data.x_test, data.y_test)

            metrics['testing_losses'].append(metrics_test['loss'])
            metrics['testing_accuracy'].append(metrics_test['accuracy'])

    return state, metrics

# =============================================================================


class SingleStage1Model(Protocol):
    def train(self, data_train: Dataset, iterations: int, seed: int,
              batch_size: int) -> None:
        ...

    def predict(self, data_test: Dataset) -> np.ndarray:
        ...


class Stage1Model_new():
    def __init__(self, optimizers: xr.DataArray, model: SingleStage1Model,
                 t_start: int, t_switch: int, rel_tolerance: float,
                 name: str = 'stage1_model', **kwargs):
        self.optimizers = optimizers
        self.model = model
        self.models = self.create_models()
        self.name = name

        self.t_start = t_start
        self.t_switch = t_switch
        self.rel_tolerance = rel_tolerance

    def train(self, data_train: DataLoader, batch_size: int, iterations: int,
              save_model: bool = False):

        for optimizer in self.optimizers.values:
            data = Dataset.from_optimization_trajectory(
                d=data_train, optimizer=optimizer,
                t=self.t_switch, t_start=self.t_start)

            self.models[optimizer].train(
                data_train=data, iterations=iterations,
                batch_size=batch_size)

    def predict(self, data_test: DataLoader,
                stage0_predictions: xr.DataArray) -> xr.DataArray:

        list_y_pred_stage1 = []
        stage0_predictions = stage0_predictions.sortby('problem')

        for optimizer in self.optimizers.values:
            problems = stage0_predictions.where(
                stage0_predictions == optimizer, drop=True).problem

            data_selected = data_test.select_problems(problems)

            data = Dataset.from_optimization_trajectory(
                d=data_selected, optimizer=optimizer,
                t=self.t_switch, t_start=self.t_start,
                train_test_split=False)

            predictions = xr.DataArray(
                self.models[optimizer].predict(data_test=data),
                dims=['problem', 'optimizer'],
                coords={'optimizer': self.optimizers,
                        'problem': data_selected.problem}
            )

            list_y_pred_stage1.append(predictions)

            # list_y_pred_stage1.append(xr.DataArray(
            #     self.models[optimizer].predict(data_selected),
            #     dims=['problem', 'optimizer'],
            #     coords={'optimizer': self.optimizers,
            #             'problem': data_selected.problem})
            # )

        stage1_predictions = xr.concat(
            list_y_pred_stage1, dim='problem').sortby('problem')

        # Select the optimizer with the highest probability
        # Select the optimizer with the best performance value
        stage1_predictions = stage1_predictions.idxmin(
            dim='optimizer').sortby('problem')

        def concat_with_plus(opt1, opt2):

            if str(opt2) == 'nan':
                return str(opt1)

            if opt1 == opt2:
                return str(opt1)
            return str(opt1) + '+' + str(opt2)

        combined_predictions = xr.apply_ufunc(
            concat_with_plus, stage0_predictions, stage1_predictions,
            vectorize=True,
            dask='parallelized',
            output_dtypes=[object])

        return combined_predictions.sortby('problem')

    def create_models(self) -> Dict[str, SingleStage1Model]:
        """
        Create the Stage1 models

        Returns
        -------
        Dict[str, SingleStage1Model]
            The Stage1 models

        Notes
        -----
        The Stage1 models are created for each optimizer in the optimizers
        """
        return {optimizer: deepcopy(self.model)
                for optimizer in self.optimizers.values}


class LSTMFLax():
    def __init__(
        self, n_outputs: int, length_sequence: int,
            optimizer: GradientTransformation = optax.adam(
                learning_rate=1e-3), latent_dim: int = 32, **kwargs):
        self.loss = loss_fn_regression
        self.optimizer = optimizer
        self.n_outputs = n_outputs
        self.latent_dim = latent_dim
        self.model = SimpleLSTM(features=self.latent_dim,
                                n_labels=self.n_outputs)

    def train(self, data_train: Dataset, iterations: int,
              seed: Optional[int] = None,
              batch_size: int = 32,
              track_loss: bool = False):
        self.state, self.metrics = training_loop(
            optimizer=self.optimizer,
            model=self.model,
            data=data_train,
            iterations=iterations,
            seed=seed,
            batch_size=batch_size,
            loss_fn=self.loss,
            track_loss=track_loss)

    def predict(self, data_test: Dataset) -> np.array:
        logits = self.model.apply(
            {'params': self.state.params}, data_test.x_test)

        return np.array(logits)
