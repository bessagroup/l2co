"""
Module for the stage1 model

Classes
-------
Stage1Model
    Class for creating a stage1 model
LSTMMultiOutputClassifier
    LSTM Multi-output classifier neural network
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Protocol, Tuple

# Third-party
import numpy as np
import xarray as xr
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential, load_model

# Local
from .data_loader import DataLoader

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


# def combine_stage0_stage1_predictions(
#         stage0_predictions: pd.DataFrame,
#         stage1_predictions: pd.DataFrame) -> xr.DataArray:
#     """
#     Combine the stage0 and stage1 predictions

#     Parameters
#     ----------
#     stage0_predictions : pd.DataFrame
#         The stage0 predictions
#     stage1_predictions : pd.DataFrame
#         The stage1 predictions

#     Returns
#     -------
#     xr.DataArray
#         The combined predictions
#     """

#     # Combine the stage0 and stage1 predictions
#     df = pd.DataFrame({'stage0': stage0_predictions,
#                        'stage1': stage1_predictions})

#     # Custom function to apply to each row
#     def string_concatenation(row):
#         if row['stage0'] == row['stage1']:
#             return row['stage0']  # Return the first column name
#         else:
#             # Concatenate values with a + sign
#             return f"{row['stage0']}+{row['stage1']}"

#     # Apply the custom function to each row
#     result_series = df.apply(string_concatenation, axis=1)
#     return xr.DataArray(result_series)

#                                                                Protocol class
# =============================================================================


class SingleStage1Model(Protocol):
    """
    Protocol class for the single Stage1 model
    """

    def create_model(self):
        """
        Protocol method for creating a single model for stage1
        """
        ...

    def train(self, X_train: np.ndarray, X_val: np.ndarray,
              batch_size: int, save_model: bool, epochs: int) -> None:
        """
        Protocol method for training the model
        """
        ...

    def load(self, model_path: str) -> None:
        """
        Protocol method for loading the model
        """
        ...

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Protocol method for predicting
        """
        ...

    def save(self) -> None:
        """
        Protocol method for saving the model
        """
        ...

#                                                             Stage1Model class
# =============================================================================


class Stage1Model:
    def __init__(self, optimizers: xr.DataArray, model: SingleStage1Model,
                 iterations: int, t_start: int,
                 t_switch: int, rel_tolerance: float,
                 name: str = 'stage1_model'):
        self.optimizers = optimizers
        self.model = model
        self.models = self.create_models()
        self.name = name

        self.iterations = iterations
        self.t_start = t_start
        self.t_switch = t_switch
        self.rel_tolerance = rel_tolerance

    def preprocess_data(
        self, data: DataLoader, optimizer: str, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        data_train, data_val = data.train_test_split(
            test_size=0.2, random_state=seed)

        da_train = data_train.X_trajectory(
            iterations=self.t_switch, t_start=self.t_start).dropna('problem')

        da_val = data_val.X_trajectory(
            iterations=self.t_switch, t_start=self.t_start).dropna('problem')

        X_train = np.atleast_3d(da_train)
        X_val = np.atleast_3d(da_val)

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))

        y_train = np.atleast_2d(data_train.y_binary(
            self.rel_tolerance).sel(
                problem=da_train.problem.values).astype(int))
        y_val = np.atleast_2d(data_val.y_binary(
            self.rel_tolerance).sel(
                problem=da_val.problem.values).astype(int))

        return X_train, y_train, X_val, y_val

    def train(self, data_train: DataLoader,
              batch_size: int, epochs: int,
              save_model: bool = False):
        """
        Train the stage1 model

        Parameters
        ----------
        data_train : DataLoader
            The training data
        batch_size : int
            The batch size
        epochs : int
            The number of epochs
        """

        for optimizer in self.optimizers.values:
            _d_train = data_train.query_optimizer(optimizer).query_t(
                self.iterations)

            X_train, y_train, X_val, y_val = self.preprocess_data(
                _d_train, optimizer)

            self.models[optimizer].train(X_train, y_train, X_val, y_val,
                                         epochs=epochs, batch_size=batch_size,
                                         save_model=save_model,
                                         save_name=f'{self.name}_{optimizer}')

    def predict(
            self, data_test: DataLoader,
            stage0_predictions: xr.DataArray) -> xr.DataArray:
        """
        Predict the performance of the optimizers

        Parameters
        ----------
        data_test : DataLoader
            The test data
        """
        list_y_pred_stage1 = []
        stage0_predictions = stage0_predictions.sortby('problem')

        for optimizer in self.optimizers.values:
            problems = stage0_predictions.where(
                stage0_predictions == optimizer, drop=True).problem

            _d_test = data_test.query_optimizer(
                optimizer).query_t(self.iterations).select_problems(problems)

            X_test = np.atleast_3d(_d_test.X_trajectory(
                iterations=self.t_switch, t_start=self.t_start))
            X_test = np.transpose(X_test, (1, 0, 2))

            list_y_pred_stage1.append(xr.DataArray(
                self.models[optimizer].predict(X_test),
                dims=['problem', 'optimizer'],
                coords={'optimizer': _d_test.optimizers,
                        'problem': _d_test.problem})
            )

        stage1_predictions = xr.concat(
            list_y_pred_stage1, dim='problem').sortby('problem')

        # Select the optimizer with the highest probability
        stage1_predictions = stage1_predictions.idxmax(
            dim='optimizer').sortby('problem')

        def concat_with_plus(opt1, opt2):
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

    def load(self) -> None:
        """
        Load the model parameters from a file

        Parameters
        ----------
        model_path : str
            The path to the model
        """
        for optimizer in self.optimizers.values:
            self.models[optimizer].load(f'{self.name}_{optimizer}')


#                                                         Model implementations
# =============================================================================

class LSTMMultiOutputClassifier(SingleStage1Model):
    def __init__(self, n_outputs: int, length_sequence: int,
                 optimizer: str = 'adam', latent_dim: int = 32):
        """
        LSTM Multi-output classifier neural network

        Parameters
        ----------
        n_outputs: int
            The number of outputs
        length_sequence : int
            The length of the sequence
        optimizer : str, optional
            The optimizer, by default 'adam'
        latent_dim : int, optional
            The latent dimension, by default 32
        """

        self.loss: str = 'binary_crossentropy'
        self.optimizer: str = optimizer
        self.latent_dim: int = latent_dim
        self.n_outputs = n_outputs
        self.length_sequence = length_sequence

        self.create_model()

    def create_model(self):
        """
        Instantiate the model
        """
        input_shape = (self.length_sequence, 1)

        self.model = Sequential()

        self.model.add(Input(shape=input_shape, name='input_layer'))
        self.model.add(LSTM(units=self.latent_dim,
                       return_state=False, name='lstm_layer'))
        self.model.add(Dense(units=self.n_outputs, activation='sigmoid'))

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    def train(
            self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray,
            batch_size: int = 32, save_model: bool = False, epochs: int = 3,
            save_name: Optional[str] = None):
        """
        Train the model

        Parameters
        ----------
        X_train : np.ndarray
            The training data
        y_train : np.ndarray
            The training labels
        X_val : np.ndarray
            The validation datalist_y_pred_stage1
        y_val : np.ndarray
            The validation labels
        batch_size : int, optional
            The batch size, by default 32
        save_model : bool, optional
            If the model should be saved, by default False
        epochs : int, optional
            The number of epochs, by default 3
        """
        if save_model:
            checkpoint = ModelCheckpoint(
                f'{save_name}.h5', save_best_only=True)
            callbacks_list = [checkpoint]

        else:
            callbacks_list = None

        self.create_model()

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Parameters
        ----------
        X_test : np.ndarray
            The test data
        """
        return self.model.predict(X_test)

    def load(self, name: str):
        """
        Load the model
        """
        self.model = load_model(f'{name}.h5')
