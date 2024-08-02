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

#                                                                       Modules
# =============================================================================
# Standard
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
import xarray as xr
from flax.training import train_state
from optax import GradientTransformation, softmax_cross_entropy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset

# Local
from .data_loader import DataLoader

#                                                                        Models
# =============================================================================


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model
    """
    features: int
    n_labels: int

    def setup(self):
        """
        Setup function, initializes the LSTM cell
        """
        self.scan_cell = nn.transforms.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast='params',
            in_axes=1, out_axes=1,
            split_rngs={'params': False})(self.features)

    @nn.compact
    def __call__(self, x, train: bool):
        carry = self.scan_cell.initialize_carry(
            jax.random.key(0), x[:, 0].shape)
        x = self.scan_cell(carry, x)[1]  # only return the output

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


class HybridModel(nn.Module):
    """
    Hybrid model combining an LSTM and an MLP model
    """
    lstm_features: int
    mlp_features: int
    mlp_layers: int
    n_labels: int

    def setup(self):
        self.lstm = SimpleLSTM(
            features=self.lstm_features, n_labels=self.n_labels)
        self.mlp = SimpleMLP(n_features=self.mlp_features,
                             n_layers=self.mlp_layers,
                             n_labels=self.n_labels)

    @nn.compact
    def __call__(self, lstm_input, mlp_input, train: bool):
        # Pass through LSTM
        lstm_output = self.lstm(lstm_input, train)

        # Pass through MLP
        mlp_output = self.mlp(mlp_input, train)

        # Concatenate the outputs
        combined = jnp.concatenate([lstm_output, mlp_output], axis=-1)

        # Final dense layer to map to the desired number of labels
        combined = nn.Dense(features=self.n_labels)(combined)

        return nn.sigmoid(combined)

# =============================================================================


class TrainState(train_state.TrainState):
    """
    Custom TrainState class
    """
    batch_stats: Any
    key: jax.Array


def create_initial_params(rng: jax.random.KeyArray,
                          input_shape_features: Tuple[int, int],
                          input_shape_trajectory: Tuple[int, int],
                          model: nn.Module) -> Dict[str, Any]:
    """
    Create initial parameters for the model

    Parameters
    ----------
    rng : jax.random.KeyArray
        Random key
    input_shape_features : Tuple[int, int]
        Shape of the input features
    input_shape_trajectory : Tuple[int, int]
        Shape of the input trajectory
    model : nn.Module
        The model

    Returns
    -------
    Dict[str, Any]
        Initial parameters
    """
    dummy_input_features = jnp.ones(input_shape_features, jnp.float32)
    dummy_input_trajectory = jnp.ones(input_shape_trajectory, jnp.float32)
    variables = model.init({'params': rng}, dummy_input_features,
                           dummy_input_trajectory, train=False)
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


def loss_fn_regression(
        params,
        state: TrainState,
        batch_x_features: jnp.ndarray,
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
    logits, updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch_x_features, batch_x_trajectory,
        train=True, mutable=['batch_stats'],
        rngs={'dropout': state.key})
    loss = jnp.mean(softmax_cross_entropy(logits, batch_y))
    return loss, (logits, updates)

# =============================================================================


def train_step(
    state: TrainState,
    batch_x_features: jnp.ndarray,
    batch_x_trajectory: jnp.ndarray,
    batch_y: jnp.ndarray,
    model: nn.Module,
    loss_fn: Callable = loss_fn_regression,
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

    _loss_fn = partial(loss_fn, state=state, batch_x_features=batch_x_features,
                       batch_x_trajectory=batch_x_trajectory, batch_y=batch_y)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    metrics = compute_metrics(logits, batch_y, n_classes=model.n_labels)
    return state, metrics


def evaluate_step(
        state: TrainState,
        batch_x_features: jnp.ndarray,
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
        batch_x_features, batch_x_trajectory, train=False)

    return compute_metrics(logits, batch_y, n_classes=model.n_labels)

# =============================================================================


def training_loop(
        optimizer: GradientTransformation,
        model: nn.Module,
        data_optimization: TorchDataLoader,
        num_epochs: int,
        seed: Optional[int],
        batch_size: int,
        loss_fn: Callable,
        track_loss: bool,
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
    input_shape_features = data_optimization.X_features[0:1].shape
    input_shape_trajectory = data_optimization.X_trajectory[0:1].shape
    variables = create_initial_params(
        rng, input_shape_features, input_shape_trajectory, model)

    params = variables['params']
    batch_stats = variables['batch_stats']

    # Create the TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        key=dropout_rng,
        batch_stats=batch_stats,
        tx=optimizer,
    )

    # JIT-compiled versions with partial application
    jit_train_step = jax.jit(
        partial(train_step, model=model, loss_fn=loss_fn))
    jit_evaluate_step = jax.jit(partial(evaluate_step, model=model))

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

    for epoch in range(num_epochs):
        logging.debug(f"Epoch {epoch+1}/{num_epochs}")
        for (batch_x_features, batch_x_trajectory), batch_y in train_batches:
            state, metrics_train = jit_train_step(
                state, jnp.array(batch_x_features),
                jnp.array(batch_x_trajectory), jnp.array(batch_y))

        if track_loss:
            # Store the loss
            metrics['training_losses'].append(metrics_train['loss'])
            metrics['training_accuracy'].append(metrics_train['accuracy'])

            metrics_test = jit_evaluate_step(
                state, eval_features,
                eval_trajectory, eval_y)

            metrics['testing_losses'].append(metrics_test['loss'])
            metrics['testing_accuracy'].append(metrics_test['accuracy'])

    return state, metrics


# =============================================================================


class SingleStage1Model(Protocol):
    """
    Protocol class for the learned model that is used in 'Stage 1'
    """

    def train(self, data_train: DataLoader, iterations: int, seed: int,
              batch_size: int) -> None:
        ...

    def predict(self, data_test: DataLoader) -> np.ndarray:
        ...

    def save(self, path: Path) -> None:
        ...


class Stage1Model():
    def __init__(self, optimizers: xr.DataArray, model: SingleStage1Model,
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
        self.models = self.create_models(model)
        self.states = self.create_states()
        self.name = name
        self.t_start = t_start
        self.t_switch = t_switch
        # self.rel_tolerance = rel_tolerance

    def train(self, data_train: DataLoader, batch_size: int, num_epochs: int,
              optimizer: GradientTransformation,
              seed: int, save_model: bool = False):
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
            data_optimization = data_train.to_learning_data(
                optimizer=opt, t=self.t_switch,
                t_start=self.t_start)

            self.states[opt], _ = training_loop(
                model=self.models[opt],
                data_optimization=data_optimization,
                num_epochs=num_epochs,
                seed=seed,
                batch_size=batch_size,
                loss_fn=loss_fn_regression,
                track_loss=False,
                optimizer=optimizer,
            )

    def predict(self, data_test: DataLoader,
                stage0_predictions: xr.DataArray) -> xr.DataArray:
        """
        Predict the best optimizer on the testing data

        Parameters
        ----------
        data_test : DataLoader
            The testing data
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

            data_selected = data_test.select_problems(problems)

            data_optimization = data_selected.to_learning_data(
                optimizer=optimizer, t=self.t_switch,
                t_start=self.t_start
            )

            logits = self.states[optimizer].apply_fn(
                {'params': self.states[optimizer].params,
                 'batch_stats': self.states[optimizer].batch_stats},
                data_optimization.X_features, data_optimization.X_trajectory,
                train=False)

            predictions = xr.DataArray(
                np.array(logits),
                dims=['problem', 'optimizer'],
                coords={'optimizer': self.optimizers,
                        'problem': data_selected.problem}
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

    def create_models(
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
        # filename = (path / 'stage1_model').with_suffix('.pkl')
        # with open(filename, 'wb') as f:
        #     pickle.dump(self, f)
        for optimizer, model in self.models.items():
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
