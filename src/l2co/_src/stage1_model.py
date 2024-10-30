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

# Standard
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, Protocol

import equinox as eqx
# Third-party
import jax.numpy as jnp
import numpy as np
import xarray as xr
from optax import GradientTransformation

# Local
from .data_loader import OptimizationDynamics
from .stage0_model import Stage0Model
from .stage1_model_equinox import train_loop

# from .stage1_model_training import TrainState

# =============================================================================


# class SingleStage1Model(eqx.Module, Protocol):
#     """
#     Protocol class for the learned model that is used in 'Stage 1'
#     """
#     ...
#     # def train(self, optimization_dynamics: Dict[str, OptimizationDynamics],
#     #           iterations: int, seed: int,
#     #           batch_size: int) -> None:
#     #     ...

#     # def predict(self, data_test: DataLoader) -> np.ndarray:
#     #     ...

#     # def save(self, path: Path) -> None:
#     #     ...


class Stage1Model():
    def __init__(self, optimizers: xr.DataArray,
                 model_stage1: eqx.Module, model_stage0: Stage0Model,
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
        # self.states = self.create_states()
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
            train_loop(model=self.models_stage1[opt],
                       X_trajectory=optimization_dynamics[opt].X_trajectory,
                       X_features=optimization_dynamics[opt].X_features,
                       Y=optimization_dynamics[opt].Y,
                       optimizer=optimizer,
                       n_steps=num_epochs,
                       verbose=verbose)

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
        ...
        # list_y_pred_stage1 = []
        # stage0_predictions = stage0_predictions.sortby('problem')

        # for optimizer in self.optimizers.values:
        #     problems = stage0_predictions.where(
        #         stage0_predictions == optimizer, drop=True).problem

        #     if len(problems) == 0:
        #         continue

        #     data_optimization = data_test[optimizer].select_problems(problems)

        #     # data_optimization = data_selected.to_learning_data(
        #     #     optimizer=optimizer, t=self.t_switch,
        #     #     t_start=self.t_start
        #     # )

        #     y_features = self.models_stage0[optimizer].clf.predict(
        #         data_optimization._X_features.to_pandas())

        #     logits = self.states[optimizer].apply_fn(
        #         {'params': self.states[optimizer].params,
        #          'batch_stats': self.states[optimizer].batch_stats},
        #         jnp.array(y_features), data_optimization.X_trajectory,
        #         train=False)

        #     predictions = xr.DataArray(
        #         np.array(logits),
        #         dims=['problem', 'optimizer'],
        #         coords={'optimizer': self.optimizers,
        #                 'problem': data_optimization.problem}
        #     )

        #     list_y_pred_stage1.append(predictions)

        # stage1_predictions = xr.concat(
        #     list_y_pred_stage1, dim='problem').sortby('problem')

        # # Select the optimizer with the highest probability
        # # Select the optimizer with the best performance value
        # stage1_predictions = stage1_predictions.idxmin(
        #     dim='optimizer').sortby('problem')

        # def concat_with_plus(opt1, opt2):

        #     if str(opt2) == 'nan':
        #         return str(opt1)

        #     if opt1 == opt2:
        #         return str(opt1)
        #     return str(opt1) + '+' + str(opt2)

        # combined_predictions = xr.apply_ufunc(
        #     concat_with_plus, stage0_predictions, stage1_predictions,
        #     vectorize=True,
        #     dask='parallelized',
        #     output_dtypes=[object])

        # return combined_predictions.sortby('problem')

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
            self, model: eqx.Module) -> Dict[str, eqx.Module]:
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

    # def create_states(self) -> Dict[str, TrainState]:
    #     """
    #     Create the TrainStates for the models

    #     Returns
    #     -------
    #     Dict[str, TrainState]
    #         The TrainStates for the models
    #     """
    #     return {optimizer: None for optimizer in self.optimizers.values}

    # def save(self, path: Path) -> None:
    #     """
    #     Save the model to a file

    #     Parameters
    #     ----------
    #     path : Path
    #         The path to the model
    #     """
    #     for optimizer, model in self.models_stage1.items():
    #         model.save(Path(path) / f"stage1_{optimizer}")


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
