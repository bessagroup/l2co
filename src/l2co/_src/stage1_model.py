# """
# Flax based neural network implementations for the Stage 1 model

# This module contains the Flax based neural network implementations for the
# Stage 1 model. The Stage 1 model is a neural network that predicts the best
# optimizer for a given problem based on the predictions of the Stage 0 model.

# Classes
# -------
# SimpleLSTM
#     Simple LSTM model
# SimpleMLP
#     Simple Multi-Layer Perceptron model
# HybridModel
#     Hybrid model combining an LSTM and an MLP model
# TrainState
#     Custom TrainState class
# Stage1Model
#     Stage 1 model
# SingleStage1Model
#     Protocol class for the learned model that is used in 'Stage 1'

# Functions
# ---------
# create_initial_params
#     Create initial parameters for the model
# compute_metrics
#     Compute the softmax cross-entropy loss and accuracy
# loss_fn_regression
#     Loss function for regression problems
# train_step
#     Training step function
# evaluate_step
#     Evaluation step function
# training_loop
#     Training loop
# """

# # Standard
# import pickle
# from copy import deepcopy
# from pathlib import Path
# from typing import Dict, Protocol

# # Third-party
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# import numpy as np
# import pandas as pd
# import torch
# import xarray as xr
# from optax import GradientTransformation

# # Local
# from .data_loader import DataLoader, OptimizationDynamics
# from .stage0_model import Stage0Model
# from .stage1_model_training import TrainState

# #                                                                        Models
# # =============================================================================


# class SimpleMLP(nn.Module):
#     """
#     Simple Multi-Layer Perceptron model
#     """
#     n_features: int
#     n_layers: int
#     n_labels: int

#     @nn.compact
#     def __call__(self, x, train: bool = True):
#         # x = x.reshape((x.shape[0], -1))  # Flatten the input

#         for _ in range(self.n_layers):
#             x = nn.Dense(features=self.n_features)(x)
#             x = nn.relu(x)

#         # x = nn.BatchNorm(use_running_average=not train)(x)

#         # x = nn.Dropout(rate=0.3, deterministic=not train)(x)

#         x = nn.Dense(features=self.n_labels)(x)
#         return nn.sigmoid(x)


# class HybridModelWithClf(nn.Module):
#     """
#     Hybrid model combining an LSTM and an MLP model
#     """
#     stage1_model: nn.Module
#     n_labels: int
#     n_layers: int = 1

#     @nn.compact
#     def __call__(self, y_features, x_traj, train: bool):

#         # Pass through stage 1 model
#         lstm_output = self.stage1_model(x_traj, train)

#         # Concatenate the outputs
#         combined = jnp.concatenate([lstm_output, y_features], axis=-1)

#         for _ in range(self.n_layers):
#             # Final dense layer to map to the desired number of labels
#             combined = nn.Dense(features=self.n_labels)(combined)
#             combined = nn.sigmoid(combined)

#         return combined

# # =============================================================================


# class LSTMEncoder(nn.Module):
#     latent_dim: int
#     hidden_size: int

#     def setup(self):
#         self.lstm_cell = nn.transforms.scan(
#             nn.OptimizedLSTMCell,
#             variable_broadcast='params',
#             in_axes=0, out_axes=0,
#             split_rngs={'params': False})(self.hidden_size)

#         self.latent_dense = nn.Dense(self.latent_dim)

#     @nn.compact
#     def __call__(self, x, train: bool = True):

#         # Initialize carry state with correct shape: (batch_size, hidden_size)
#         carry = self.lstm_cell.initialize_carry(
#             jax.random.PRNGKey(0), (self.hidden_size,))

#         carry, x = self.lstm_cell(carry, x)

#         # Map the final LSTM output to the latent space
#         z = self.latent_dense(x)

#         # Pool over sequence length to get a single vector per batch
#         z = jnp.mean(x, axis=0)  # (batch_size, latent_dim)
#         return z

# # =============================================================================


# class LSTMDecoder(nn.Module):
#     output_dim: int
#     latent_dim: int
#     hidden_size: int

#     def setup(self):
#         self.lstm_cell = nn.transforms.scan(
#             nn.OptimizedLSTMCell,
#             variable_broadcast='params',
#             in_axes=0, out_axes=0,
#             split_rngs={'params': False})(self.hidden_size)

#         # Output projection layer
#         self.hidden_dense = nn.Dense(self.hidden_size)
#         self.output_layer = nn.Dense(self.output_dim)

#     @nn.compact
#     def __call__(self, z, seq_len: int, train: bool = True):
#         # batch_size = z.shape[0]

#         # Map latent vector to the LSTM's hidden size
#         z = self.hidden_dense(z)

#         carry = self.lstm_cell.initialize_carry(
#             jax.random.PRNGKey(1), (self.hidden_size,))

#         # Use latent vector z as the initial input for the decoder
#         # (batch_size, seq_length, latent_dim)
#         x = jnp.tile(z[jnp.newaxis, :], (seq_len, 1))

#         carry, x = self.lstm_cell(carry, x)

#         output = self.output_layer(x)

#         return output

# # =============================================================================


# class LSTMAutoencoder(nn.Module):
#     output_dim: int
#     latent_dim: int
#     hidden_size: int

#     def setup(self):
#         self.encoder = LSTMEncoder(latent_dim=self.latent_dim,
#                                    hidden_size=self.hidden_size)
#         self.decoder = LSTMDecoder(latent_dim=self.latent_dim,
#                                    hidden_size=self.hidden_size,
#                                    output_dim=self.output_dim)

#     @nn.compact
#     def __call__(self, x, train: bool = True):
#         # input shape = (sequence_length, output_dim)

#         sequence_length = x.shape[0]
#         latent_space = self.encoder(x, train)
#         reconstruction = self.decoder(latent_space, sequence_length, train)
#         return reconstruction


# # =============================================================================

# class LSTMAndMLP(nn.Module):
#     mlp_layers: int
#     mlp_neurons_per_layer: int
#     lstm_hidden_size: int
#     lstm_latent_dim: int
#     n_labels: int

#     def setup(self):
#         self.encoder = LSTMEncoder(latent_dim=self.lstm_latent_dim,
#                                    hidden_size=self.lstm_hidden_size)

#         self.mlp = SimpleMLP(n_features=self.mlp_neurons_per_layer,
#                              n_layers=self.mlp_layers,
#                              n_labels=self.n_labels)

#     @nn.compact
#     def __call__(self, x_trajectory, x_features,
#                  train: bool = True, *args, **kwargs):
#         # input shape = (batch_size, sequence_length, output_dim)

#         x_latent_trajectory = self.encoder(x_trajectory, train)
#         x_combined = jnp.concatenate(
#             [x_latent_trajectory, x_features], axis=-1)
#         y = self.mlp(x_combined, train)
#         return y


# # =============================================================================

# def predict_from_stage0(
#     stage0_model: Stage0Model, batch_x_features: torch.Tensor,
#         columns: np.array) -> torch.Tensor:
#     df = pd.DataFrame(batch_x_features.squeeze(), columns=columns)
#     y_pred = stage0_model.clf.predict(df)
#     return torch.Tensor(y_pred)  # shape = (batch_size, n_optimizers)

# # =============================================================================


# class SingleStage1Model(Protocol):
#     """
#     Protocol class for the learned model that is used in 'Stage 1'
#     """

#     def train(self, optimization_dynamics: Dict[str, OptimizationDynamics],
#               iterations: int, seed: int,
#               batch_size: int) -> None:
#         ...

#     def predict(self, data_test: DataLoader) -> np.ndarray:
#         ...

#     def save(self, path: Path) -> None:
#         ...


# class Stage1Model():
#     def __init__(self, optimizers: xr.DataArray,
#                  model_stage1: SingleStage1Model, model_stage0: Stage0Model,
#                  t_start: int, t_switch: int, rel_tolerance: float,
#                  name: str = 'stage1_model',
#                  **kwargs):
#         """
#         Stage1 model

#         Parameters
#         ----------
#         optimizers : xarray.DataArray
#             The optimizers
#         model : SingleStage1Model
#             The model
#         t_start : int
#             The start time
#         t_switch : int
#             The switch time
#         rel_tolerance : float
#             The relative tolerance
#         name : str, optional
#             The name of the model, by default 'stage1_model'
#         """
#         self.optimizers = optimizers
#         self.models_stage1 = self.create_models_stage1(model_stage1)
#         self.models_stage0 = self.create_models_stage0(model_stage0)
#         self.states = self.create_states()
#         self.name = name
#         self.t_start = t_start
#         self.t_switch = t_switch

#     def train(self, optimization_dynamics: Dict[str, OptimizationDynamics],
#               batch_size: int, num_epochs: int,
#               optimizer: GradientTransformation,
#               seed: int, save_model: bool = False,
#               track_loss: bool = False,
#               verbose: bool = False):
#         """
#         Train the model

#         Parameters
#         ----------
#         data_train : DataLoader
#             The training data
#         batch_size : int
#             The batch size
#         num_epochs : int
#             The number of epochs
#         optimizer : GradientTransformation
#             The optimizer
#         seed : int
#             Seed for the random number generator
#         save_model : bool, optional
#             If True, save the model, by default False
#         """
#         for opt in self.optimizers.values:
#             self.models_stage0[opt].train(
#                 feature_data=optimization_dynamics[opt].to_feature_data()
#             )

#         for opt in self.optimizers.values:
#             self.states[opt], metrics = training_loop(
#                 model_stage0=self.models_stage0[opt],
#                 model_stage1=self.models_stage1[opt],
#                 data_optimization=optimization_dynamics[opt],
#                 num_epochs=num_epochs,
#                 seed=seed,
#                 batch_size=batch_size,
#                 loss_fn=loss_fn,
#                 track_loss=track_loss,
#                 optimizer=optimizer,
#                 verbose=verbose
#             )

#         if track_loss:
#             return metrics

#     def predict(self, data_test: Dict[str, OptimizationDynamics],
#                 stage0_predictions: xr.DataArray) -> xr.DataArray:
#         """
#         Predict the best optimizer on the testing data

#         Parameters
#         ----------
#         data_test : Dict[str, OptimizationDynamics]
#             The testing data optimizationdynamics
#         stage0_predictions : xarray.DataArray
#             The predictions of the Stage0 model

#         Returns
#         -------
#         xarray.DataArray
#             The predictions of the Stage
#         """

#         list_y_pred_stage1 = []
#         stage0_predictions = stage0_predictions.sortby('problem')

#         for optimizer in self.optimizers.values:
#             problems = stage0_predictions.where(
#                 stage0_predictions == optimizer, drop=True).problem

#             if len(problems) == 0:
#                 continue

#             data_optimization = data_test[optimizer].select_problems(problems)

#             # data_optimization = data_selected.to_learning_data(
#             #     optimizer=optimizer, t=self.t_switch,
#             #     t_start=self.t_start
#             # )

#             y_features = self.models_stage0[optimizer].clf.predict(
#                 data_optimization._X_features.to_pandas())

#             logits = self.states[optimizer].apply_fn(
#                 {'params': self.states[optimizer].params,
#                  'batch_stats': self.states[optimizer].batch_stats},
#                 jnp.array(y_features), data_optimization.X_trajectory,
#                 train=False)

#             predictions = xr.DataArray(
#                 np.array(logits),
#                 dims=['problem', 'optimizer'],
#                 coords={'optimizer': self.optimizers,
#                         'problem': data_optimization.problem}
#             )

#             list_y_pred_stage1.append(predictions)

#         stage1_predictions = xr.concat(
#             list_y_pred_stage1, dim='problem').sortby('problem')

#         # Select the optimizer with the highest probability
#         # Select the optimizer with the best performance value
#         stage1_predictions = stage1_predictions.idxmin(
#             dim='optimizer').sortby('problem')

#         def concat_with_plus(opt1, opt2):

#             if str(opt2) == 'nan':
#                 return str(opt1)

#             if opt1 == opt2:
#                 return str(opt1)
#             return str(opt1) + '+' + str(opt2)

#         combined_predictions = xr.apply_ufunc(
#             concat_with_plus, stage0_predictions, stage1_predictions,
#             vectorize=True,
#             dask='parallelized',
#             output_dtypes=[object])

#         return combined_predictions.sortby('problem')

#     def create_models_stage0(
#             self, model: Stage0Model) -> Dict[str, Stage0Model]:
#         """
#         Create the Stage0 models

#         Returns
#         -------
#         Dict[str, Stage0Model]
#             The Stage0 models
#         """
#         return {optimizer: deepcopy(model)
#                 for optimizer in self.optimizers.values}

#     def create_models_stage1(
#             self, model: SingleStage1Model) -> Dict[str, SingleStage1Model]:
#         """
#         Create the Stage1 models

#         Returns
#         -------
#         Dict[str, SingleStage1Model]
#             The Stage1 models

#         Notes
#         -----
#         The Stage1 models are created for each optimizer in the optimizers
#         """
#         return {optimizer: deepcopy(model)
#                 for optimizer in self.optimizers.values}

#     def create_states(self) -> Dict[str, TrainState]:
#         """
#         Create the TrainStates for the models

#         Returns
#         -------
#         Dict[str, TrainState]
#             The TrainStates for the models
#         """
#         return {optimizer: None for optimizer in self.optimizers.values}

#     def save(self, path: Path) -> None:
#         """
#         Save the model to a file

#         Parameters
#         ----------
#         path : Path
#             The path to the model
#         """
#         for optimizer, model in self.models_stage1.items():
#             model.save(Path(path) / f"stage1_{optimizer}")


# def load(path: Path) -> Stage1Model:
#     """
#     Load the model from a file

#     Parameters
#     ----------
#     path : Path
#         The path to the model
#     """
#     filename = (path / 'stage1_model').with_suffix('.pkl')
#     with open(filename, 'rb') as f:
#         model = pickle.load(f)

#     return model
