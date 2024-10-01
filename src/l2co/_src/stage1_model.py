"""
Flax based neural network implementations for the Stage 1 model

This module contains the Flax based neural network implementations for the
Stage 1 model. The Stage 1 model is a neural network that predicts the best
optimizer for a given problem based on the predictions of the Stage 0 model.

Classes
-------
SimpleLSTM
    Simple LSTM model
SimpleMLP
    Simple Multi-Layer Perceptron model
HybridModel
    Hybrid model combining an LSTM and an MLP model
TrainState
    Custom TrainState class
Stage1Model
    Stage 1 model
SingleStage1Model
    Protocol class for the learned model that is used in 'Stage 1'

Functions
---------
create_initial_params
    Create initial parameters for the model
compute_metrics
    Compute the softmax cross-entropy loss and accuracy
loss_fn_regression
    Loss function for regression problems
train_step
    Training step function
evaluate_step
    Evaluation step function
training_loop
    Training loop
"""

import logging
import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

# Third-party
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
import xarray as xr
#                                                                       Modules
# =============================================================================
# Standard
from flax import linen as nn
from flax.training import train_state
from optax import GradientTransformation, softmax_cross_entropy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset
from tqdm import tqdm

# Local
from .data_loader import DataLoader, OptimizationDynamics
from .stage0_model import Stage0Model

#                                                                        Models
# =============================================================================


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model
    """
    hidden_size: int
    n_labels: int
    n_layers: int = 1

    def setup(self):
        """
        Setup function, initializes the LSTM cell
        """
        self.lstm_cells = [nn.transforms.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast='params',
            in_axes=1, out_axes=1,
            split_rngs={'params': False})(self.hidden_size)
            for _ in range(self.n_layers)]

    @nn.compact
    def __call__(self, x, train: bool, return_latent: bool = False):
        carry = [cell.initialize_carry(
            jax.random.key(i), x[:, 0].shape)
            for i, cell in enumerate(self.lstm_cells)]

        # Pass the input through each LSTM layer
        for i in range(self.n_layers):
            carry[i], x = self.lstm_cells[i](carry[i], x)

        # Optionally return the latent space directly after the LSTM layers
        if return_latent:
            return x

        # Flatten the output if needed (taking the last output in the sequence)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.Dropout(rate=0.3, deterministic=not train)(x)

        # Add a dense layer to map to n_labels
        x = nn.Dense(features=self.n_labels)(x)

        return nn.sigmoid(x)


def normal_initializer(
        rng: jax.random.KeyArray, shape: Tuple[int, int],
        dtype: jnp.dtype) -> jax.Array:
    """
    Normal initializer for weights and biases

    Parameters
    ----------
    rng : jax.random.KeyArray
        Random key
    shape : Tuple
        Shape of the weights or biases
    dtype : jnp.dtype
        Data type

    Returns
    -------
    jax.Array
        The initialized weights or biases
    """
    return jax.random.normal(rng, shape, dtype) * 0.01


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron model
    """
    n_features: int
    n_layers: int
    n_labels: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = x.reshape((x.shape[0], -1))  # Flatten the input

        for _ in range(self.n_layers):
            x = nn.Dense(features=self.n_features,
                         kernel_init=normal_initializer,
                         bias_init=normal_initializer)(x)
            x = nn.relu(x)

        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.Dropout(rate=0.3, deterministic=not train)(x)

        x = nn.Dense(features=self.n_labels, kernel_init=normal_initializer,
                     bias_init=normal_initializer)(x)
        return nn.sigmoid(x)


class HybridModelWithClf(nn.Module):
    """
    Hybrid model combining an LSTM and an MLP model
    """
    stage1_model: nn.Module
    n_labels: int
    n_layers: int = 1

    @nn.compact
    def __call__(self, y_features, x_traj, train: bool):

        # Pass through stage 1 model
        lstm_output = self.stage1_model(x_traj, train)

        # Concatenate the outputs
        combined = jnp.concatenate([lstm_output, y_features], axis=-1)

        for _ in range(self.n_layers):
            # Final dense layer to map to the desired number of labels
            combined = nn.Dense(features=self.n_labels)(combined)
            combined = nn.sigmoid(combined)

        return combined

# =============================================================================


class Encoder(nn.Module):
    latent_dim: int
    hidden_size: int
    n_layers: int = 1

    def setup(self):
        self.lstm_cells = [
            nn.transforms.scan(
                nn.OptimizedLSTMCell,
                variable_broadcast='params',
                in_axes=1, out_axes=1,
                split_rngs={'params': False})(self.hidden_size)
            for _ in range(self.n_layers)
        ]

        self.latent_dense = nn.Dense(self.latent_dim)

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]

        # Initialize carry state with correct shape: (batch_size, hidden_size)
        carry = [
            cell.initialize_carry(
                jax.random.PRNGKey(i), (batch_size, self.hidden_size))
            for i, cell in enumerate(self.lstm_cells)
        ]

        # print(f"input shape x {x.shape=}")

        for i in range(self.n_layers):
            carry[i], x = self.lstm_cells[i](carry[i], x)

        # Map the final LSTM output to the latent space
        z = self.latent_dense(x)

        # Pool over sequence length to get a single vector per batch
        z = jnp.mean(x, axis=1)  # (batch_size, latent_dim)
        return z


class Decoder(nn.Module):
    output_dim: int
    latent_dim: int
    hidden_size: int
    n_layers: int = 1

    def setup(self):
        self.lstm_cells = [
            nn.transforms.scan(
                nn.OptimizedLSTMCell,
                variable_broadcast='params',
                in_axes=1, out_axes=1,
                split_rngs={'params': False})(self.hidden_size)
            for _ in range(self.n_layers)
        ]

        # Output projection layer
        self.hidden_dense = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.output_dim)

    @nn.compact
    def __call__(self, z, seq_len):
        batch_size = z.shape[0]
        # print(f"{batch_size=}, {seq_len=}")

        # Map latent vector to the LSTM's hidden size
        z = self.hidden_dense(z)

        carry = [
            cell.initialize_carry(
                jax.random.PRNGKey(i + 1), (batch_size, self.hidden_size))
            for i, cell in enumerate(self.lstm_cells)
        ]

        # Use latent vector z as the initial input for the decoder
        # (batch_size, seq_length, latent_dim)
        x = jnp.tile(z[:, jnp.newaxis, :], (1, seq_len, 1))

        # Initialize output storage
        outputs = []

        for i in range(seq_len):
            for j in range(self.n_layers):
                carry[j], x = self.lstm_cells[j](carry[j], x)

            # Generate output at each time step
            output = self.output_layer(x[:, i, :])  # (batch_size, output_dim)
            outputs.append(output)

        # Stack outputs to form the final sequence
        # (batch_size, seq_length, output_dim)
        return jnp.stack(outputs, axis=1)


class LSTMAutoencoder(nn.Module):
    output_dim: int
    latent_dim: int
    hidden_size: int
    n_layers: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        # input shape = (batch_size, sequence_length, output_dim)

        sequence_length = x.shape[1]
        encoder = Encoder(latent_dim=self.latent_dim,
                          hidden_size=self.hidden_size,
                          n_layers=self.n_layers)
        decoder = Decoder(latent_dim=self.latent_dim,
                          hidden_size=self.hidden_size,
                          n_layers=self.n_layers,
                          output_dim=self.output_dim)

        x = nn.BatchNorm(use_running_average=not train)(x)

        latent_space = encoder(x)
        reconstruction = decoder(latent_space, sequence_length)
        return reconstruction


# =============================================================================

class TrainState(train_state.TrainState):
    """
    Custom TrainState class
    """
    batch_stats: Any
    key: jax.Array


# def create_initial_params_old(rng: jax.random.KeyArray,
#                               input_shape_features: Tuple[int, int],
#                               input_shape_trajectory: Tuple[int, int],
#                               model: nn.Module) -> Dict[str, Any]:
#     """
#     Create initial parameters for the model

#     Parameters
#     ----------
#     rng : jax.random.KeyArray
#         Random key
#     input_shape_features : Tuple[int, int]
#         Shape of the input features
#     input_shape_trajectory : Tuple[int, int]
#         Shape of the input trajectory
#     model : nn.Module
#         The model

#     Returns
#     -------
#     Dict[str, Any]
#         Initial parameters
#     """
#     dummy_input_features = jnp.ones(input_shape_features, jnp.float32)
#     dummy_input_trajectory = jnp.ones(input_shape_trajectory, jnp.float32)
#     variables = model.init({'params': rng}, dummy_input_features,
#                            dummy_input_trajectory, train=False)
#     return variables


def create_initial_params(rng: jax.random.KeyArray,
                          model: nn.Module,
                          **input_shapes) -> Dict[str, Any]:
    """
    Create initial parameters for the model

    Parameters
    ----------
    rng : jax.random.KeyArray
        Random key
    model : nn.Module
        The model
    input_shapes : Dict[str, Tuple[int, int]]
        The shapes of the input tensors
    """

    # Create dummy input tensors for each shape provided in input_shapes
    dummy_inputs = [jnp.ones(shape, jnp.float32)
                    for shape in input_shapes.values()]

    # Initialize model parameters using the dummy inputs
    variables = model.init({'params': rng}, *dummy_inputs, train=False)

    return variables


def compute_metrics(
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        n_classes: int) -> Dict[str, jnp.ndarray]:
    """
    Compute the softmax cross-entropy loss and accuracy

    Parameters
    ----------
    logits : jnp.ndarray
        The logits
    labels : jnp.ndarray
        The labels
    n_classes : int
        The number of classes

    Returns
    -------
    Dict[str, jnp.ndarray]
        The loss and accuracy in a dictionary

    Notes
    -----
    The loss is computed as the mean softmax cross-entropy loss.
    The accuracy is computed as the mean number of correct predictions
    """
    loss = jnp.mean(softmax_cross_entropy(logits, labels))
    accuracy = jnp.mean(jnp.argmin(logits, -1) == jnp.argmin(labels, -1))
    return {'loss': loss, 'accuracy': accuracy}

#                                                                Loss functions
# =============================================================================


def loss_fn(
        params,
        state: TrainState,
        batch_y_features: jnp.ndarray,
        batch_x_trajectory: jnp.ndarray,
        batch_y: np.ndarray) -> Tuple[jax.Array, Tuple[Any, Any]]:
    """
    Loss function for regression problems

    Parameters
    ----------
    params : Any
        The parameters
    state : TrainState
        The TrainState
    batch_x_features : np.ndarray
        The input features
    batch_x_trajectory : np.ndarray
        The input trajectories
    batch_y : np.ndarray
        The labels

    Returns
    -------
    Tuple[jax.Array, Tuple[Any, Any]]
        The loss and the logits and updates
    """
    batch_y_pred, updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch_y_features, batch_x_trajectory,
        train=True, mutable=['batch_stats'],
        rngs={'dropout': state.key})
    # loss = jnp.mean(softmax_cross_entropy(y_pred, batch_y))
    # MSE Error
    loss = jnp.mean(jnp.square(batch_y_pred - batch_y))
    return loss, (batch_y_pred, updates)

# =============================================================================


def train_step(
    state: TrainState,
    batch_y_features: jnp.ndarray,
    batch_x_trajectory: jnp.ndarray,
    batch_y: jnp.ndarray,
    model: nn.Module,
    loss_fn: Callable = loss_fn,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Training step function

    Parameters
    ----------
    state : TrainState
        The TrainState
    batch_x_features : jnp.ndarray
        The input features
    batch_x_trajectory : jnp.ndarray
        The input trajectory
    batch_y : jnp.ndarray
        The labels
    model : nn.Module
        The model
    loss_fn : Callable, optional
        The loss function, by default loss_fn_regression

    Returns
    -------
    Tuple[TrainState, Dict[str, jnp.ndarray]]
        The updated TrainState and the metrics
    """

    _loss_fn = partial(loss_fn, state=state, batch_y_features=batch_y_features,
                       batch_x_trajectory=batch_x_trajectory, batch_y=batch_y)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, (batch_y_pred, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    # metrics = compute_metrics(batch_y_pred, batch_y, n_classes=model.n_labels)
    metrics = {}
    metrics['loss'] = loss
    return state, metrics


def evaluate_step(
        state: TrainState,
        batch_y_features: jnp.ndarray,
        batch_x_trajectory: jnp.ndarray,
        batch_y: jnp.ndarray,
        model: nn.Module) -> Dict[str, jnp.ndarray]:
    """
    Evaluation step function

    Parameters
    ----------
    state : TrainState
        The TrainState
    batch_x_features : jnp.ndarray
        The input features
    batch_x_trajectory : jnp.ndarray
        The input trajectory
    batch_y : jnp.ndarray
        The labels
    model : nn.Module
        The model

    Returns
    -------
    Dict[str, jnp.ndarray]
        The metrics of the evaluation on the testing set
    """
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch_y_features, batch_x_trajectory, train=False)

    return compute_metrics(logits, batch_y, n_classes=model.n_labels)

# =============================================================================


def training_loop(
        optimizer: GradientTransformation,
        model_stage0: Stage0Model,
        model_stage1: nn.Module,
        data_optimization: TorchDataLoader,
        num_epochs: int,
        seed: Optional[int],
        batch_size: int,
        loss_fn: Callable,
        track_loss: bool,
        verbose: bool = False
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Training loop

    Parameters
    ----------
    optimizer : GradientTransformation
        optimizer to use
    model : nn.Module
        neural network model
    data_optimization : TorchDataLoader
        learning data
    num_epochs : int
        number of epochs
    seed : Optional[int]
        seed for random number generation
    batch_size : int
        batch size
    loss_fn : Callable
        loss function
    track_loss : bool
        if True, track the loss
    verbose : bool, optional
        if True, show progress bar, by default False

    Returns
    -------
    Tuple[TrainState, Dict[str, jnp.ndarray]]
        The final TrainState and the metrics
    """
    if seed is None:
        seed = np.random.randint(1e6)

    main_key = jax.random.PRNGKey(seed)
    rng, dropout_rng = jax.random.split(main_key, 2)
    # (1, RESOLUTION, RESOLUTION, 1)

    input_shape_features = data_optimization.Y[0].shape
    # input_shape_features = data_optimization.X_features[0:1].shape
    input_shape_trajectory = data_optimization.X_trajectory[0:1].shape
    variables = create_initial_params(
        rng=rng, model=model_stage1,
        input_shape_features=input_shape_features,
        input_shape_trajectory=input_shape_trajectory,
    )

    params = variables['params']
    batch_stats = variables['batch_stats']

    # Create the TrainState
    state = TrainState.create(
        apply_fn=model_stage1.apply,
        params=params,
        key=dropout_rng,
        batch_stats=batch_stats,
        tx=optimizer,
    )

    # JIT-compiled versions with partial application
    jit_train_step = jax.jit(
        partial(train_step, model=model_stage1, loss_fn=loss_fn))
    # jit_evaluate_step = jax.jit(partial(evaluate_step, model=model_stage1))

    if track_loss:
        # List to store the losses
        metrics = {'training_losses': [], 'training_accuracy': [],
                   'testing_losses': [], 'testing_accuracy': []}

    else:
        metrics = None

    train_indices, validation_indices = train_test_split(
        range(len(data_optimization)), test_size=0.2, random_state=42)

    train_split = Subset(data_optimization, train_indices)
    validation_split = Subset(data_optimization, validation_indices)

    train_batches = TorchDataLoader(
        train_split, batch_size=batch_size, shuffle=True)

    validation_batches = TorchDataLoader(
        validation_split, batch_size=len(validation_split), shuffle=False)

    (eval_features, eval_trajectory), eval_y = next(
        iter(validation_batches))

    # subtract the mean of eval_y over the second axis
    eval_y = (eval_y - eval_y.mean(dim=2, keepdim=True))

    # eval_y_features = predict_from_stage0(
    #     stage0_model=model_stage0,
    #     batch_x_features=eval_features,
    #     columns=data_optimization._X_features.features)

    for epoch in range(num_epochs):
        # logging.debug(f"Epoch {epoch+1}/{num_epochs}")

        # Initialize tqdm for batches
        batch_iterator = tqdm(
            train_batches, desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not verbose, leave=False, position=0)

        for (batch_x_features, batch_x_trajectory), batch_y in batch_iterator:

            batch_y_features = predict_from_stage0(
                stage0_model=model_stage0,
                batch_x_features=batch_x_features,
                columns=data_optimization._X_features.features)

            # substract the mean of batch_y over the second axis
            batch_y = (batch_y - batch_y.mean(dim=2, keepdim=True))

            state, metrics_train = jit_train_step(
                state, batch_y_features=jnp.array(batch_y_features),
                batch_x_trajectory=jnp.array(batch_x_trajectory),
                batch_y=jnp.array(batch_y))

            if track_loss:
                # Store the loss
                metrics['training_losses'].append(metrics_train['loss'])
                # metrics['training_accuracy'].append(metrics_train['accuracy'])

            # Update tqdm description with the current loss
            batch_iterator.set_description((
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {metrics_train['loss']: .4f} - "
                # f"Accuracy: {metrics_train['accuracy']: .4f}"
            ))

            # metrics_test = jit_evaluate_step(
            #     state, batch_y_features=jnp.array(eval_y_features),
            #     batch_x_trajectory=jnp.array(eval_trajectory),
            #     batch_y=jnp.array(eval_y))

            # metrics['testing_losses'].append(metrics_test['loss'])
            # metrics['testing_accuracy'].append(metrics_test['accuracy'])

    return state, metrics


def predict_from_stage0(
    stage0_model: Stage0Model, batch_x_features: torch.Tensor,
        columns: np.array) -> torch.Tensor:
    df = pd.DataFrame(batch_x_features.squeeze(), columns=columns)
    y_pred = stage0_model.clf.predict(df)
    return torch.Tensor(y_pred)  # shape = (batch_size, n_optimizers)

# =============================================================================


class SingleStage1Model(Protocol):
    """
    Protocol class for the learned model that is used in 'Stage 1'
    """

    def train(self, optimization_dynamics: Dict[str, OptimizationDynamics],
              iterations: int, seed: int,
              batch_size: int) -> None:
        ...

    def predict(self, data_test: DataLoader) -> np.ndarray:
        ...

    def save(self, path: Path) -> None:
        ...


class Stage1Model():
    def __init__(self, optimizers: xr.DataArray,
                 model_stage1: SingleStage1Model, model_stage0: Stage0Model,
                 t_start: int, t_switch: int, rel_tolerance: float,
                 name: str = 'stage1_model',
                 **kwargs):
        """
        Stage1 model

        Parameters
        ----------
        optimizers : xarray.DataArray
            The optimizers
        model : SingleStage1Model
            The model
        t_start : int
            The start time
        t_switch : int
            The switch time
        rel_tolerance : float
            The relative tolerance
        name : str, optional
            The name of the model, by default 'stage1_model'
        """
        self.optimizers = optimizers
        self.models_stage1 = self.create_models_stage1(model_stage1)
        self.models_stage0 = self.create_models_stage0(model_stage0)
        self.states = self.create_states()
        self.name = name
        self.t_start = t_start
        self.t_switch = t_switch

    def train(self, optimization_dynamics: Dict[str, OptimizationDynamics],
              batch_size: int, num_epochs: int,
              optimizer: GradientTransformation,
              seed: int, save_model: bool = False,
              track_loss: bool = False,
              verbose: bool = False):
        """
        Train the model

        Parameters
        ----------
        data_train : DataLoader
            The training data
        batch_size : int
            The batch size
        num_epochs : int
            The number of epochs
        optimizer : GradientTransformation
            The optimizer
        seed : int
            Seed for the random number generator
        save_model : bool, optional
            If True, save the model, by default False
        """
        for opt in self.optimizers.values:
            self.models_stage0[opt].train(
                feature_data=optimization_dynamics[opt].to_feature_data()
            )

        for opt in self.optimizers.values:
            self.states[opt], metrics = training_loop(
                model_stage0=self.models_stage0[opt],
                model_stage1=self.models_stage1[opt],
                data_optimization=optimization_dynamics[opt],
                num_epochs=num_epochs,
                seed=seed,
                batch_size=batch_size,
                loss_fn=loss_fn,
                track_loss=track_loss,
                optimizer=optimizer,
                verbose=verbose
            )

        if track_loss:
            return metrics

    def predict(self, data_test: Dict[str, OptimizationDynamics],
                stage0_predictions: xr.DataArray) -> xr.DataArray:
        """
        Predict the best optimizer on the testing data

        Parameters
        ----------
        data_test : Dict[str, OptimizationDynamics]
            The testing data optimizationdynamics
        stage0_predictions : xarray.DataArray
            The predictions of the Stage0 model

        Returns
        -------
        xarray.DataArray
            The predictions of the Stage
        """

        list_y_pred_stage1 = []
        stage0_predictions = stage0_predictions.sortby('problem')

        for optimizer in self.optimizers.values:
            problems = stage0_predictions.where(
                stage0_predictions == optimizer, drop=True).problem

            if len(problems) == 0:
                continue

            data_optimization = data_test[optimizer].select_problems(problems)

            # data_optimization = data_selected.to_learning_data(
            #     optimizer=optimizer, t=self.t_switch,
            #     t_start=self.t_start
            # )

            y_features = self.models_stage0[optimizer].clf.predict(
                data_optimization._X_features.to_pandas())

            logits = self.states[optimizer].apply_fn(
                {'params': self.states[optimizer].params,
                 'batch_stats': self.states[optimizer].batch_stats},
                jnp.array(y_features), data_optimization.X_trajectory,
                train=False)

            predictions = xr.DataArray(
                np.array(logits),
                dims=['problem', 'optimizer'],
                coords={'optimizer': self.optimizers,
                        'problem': data_optimization.problem}
            )

            list_y_pred_stage1.append(predictions)

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

    def create_models_stage0(
            self, model: Stage0Model) -> Dict[str, Stage0Model]:
        """
        Create the Stage0 models

        Returns
        -------
        Dict[str, Stage0Model]
            The Stage0 models
        """
        return {optimizer: deepcopy(model)
                for optimizer in self.optimizers.values}

    def create_models_stage1(
            self, model: SingleStage1Model) -> Dict[str, SingleStage1Model]:
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
        return {optimizer: deepcopy(model)
                for optimizer in self.optimizers.values}

    def create_states(self) -> Dict[str, TrainState]:
        """
        Create the TrainStates for the models

        Returns
        -------
        Dict[str, TrainState]
            The TrainStates for the models
        """
        return {optimizer: None for optimizer in self.optimizers.values}

    def save(self, path: Path) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        path : Path
            The path to the model
        """
        for optimizer, model in self.models_stage1.items():
            model.save(Path(path) / f"stage1_{optimizer}")


def load(path: Path) -> Stage1Model:
    """
    Load the model from a file

    Parameters
    ----------
    path : Path
        The path to the model
    """
    filename = (path / 'stage1_model').with_suffix('.pkl')
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model
