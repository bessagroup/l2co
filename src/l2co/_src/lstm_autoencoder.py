"""
Module for the stage1 model
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import Dict, Protocol

# Third-party
import numpy as np
import xarray as xr
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model, Sequential

# Local
from .data_loader import DataLoader

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


class Stage1Model:
    def __init__(self, optimizers: xr.DataArray, model: SingleStage1Model):
        self.optimizers = optimizers
        self.model = model
        self.models = self.create_model()

    def train(self, data_train: DataLoader,
              batch_size: int, epochs: int):
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

            _d_train, _d_val = _d_train.train_test_split(test_size=0.2)

            X_train = np.atleast_3d(_d_train.X_trajectory(
                self.t_switch, t_start=self.t_start))
            X_val = np.atleast_3d(_d_val.X_trajectory(
                self.t_switch, t_start=self.t_start))

            X_train = np.transpose(X_train, (1, 0, 2))
            X_val = np.transpose(X_val, (1, 0, 2))

            y_train = np.atleast_2d(_d_train.y_binary(
                self.rel_tolerance).astype(int))
            y_val = np.atleast_2d(_d_val.y_binary(
                self.rel_tolerance).astype(int))

            self.models[optimizer].train(X_train, y_train, X_val, y_val,
                                         epochs=epochs, batch_size=batch_size)

    def predict(self, data_test: DataLoader) -> xr.DataArray:
        """
        Predict the performance of the optimizers

        Parameters
        ----------
        data_test : DataLoader
            The test data
        """
        ...

    def create_model(self) -> Dict[str, SingleStage1Model]:
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
        return {optimizer: self.model.create_model()
                for optimizer in self.optimizers.values}

    def load(self, model_path: str) -> None:
        """
        Load the model parameters from a file

        Parameters
        ----------
        model_path : str
            The path to the model
        """
        for optimizer in self.optimizers.values:
            self.models[optimizer].load(model_path)


class SingleStage1Model(Protocol):
    """
    Protocol class for the single Stage1 model
    """

    def create_model(self) -> None:
        """
        Protocol method for creating a single model for stage1
        """
        ...

    def train(self, X_train: np.ndarray, X_val: np.ndarray,
              batch_size: int = 32,
              save_model: bool = False, epochs: int = 3) -> None:
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


class LSTMMultiOutputClassifier(SingleStage1Model):
    def __init__(self, optimizers: xr.DataArray, length_sequence: int,
                 optimizer: str = 'adam', latent_dim: int = 32):
        """
        LSTM Multi-output classifier neural network

        Parameters
        ----------
        optimizers : xr.DataArray
            The optimizers
        length_sequence : int
            The length of the sequence
        optimizer : str, optional
            The optimizer, by default 'adam'
        """

        self.loss: str = 'binary_crossentropy'
        self.optimizer: str = optimizer
        self.latent_dim: int = latent_dim
        self.optimizers = optimizers
        self.length_sequence = length_sequence

        self.model: Model = None

    def create_model(self):
        """
        Instantiate the model
        """
        input_shape = (self.length_sequence, 1)

        self.model = Sequential()

        self.model.add(Input(shape=input_shape, name='input_layer'))
        self.model.add(LSTM(units=self.latent_dim,
                       return_state=False, name='lstm_layer'))
        self.model.add(Dense(units=len(self.optimizer), activation='sigmoid'))

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    def train(
            self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray,
            batch_size: int = 32, save_model: bool = False, epochs: int = 3):
        """
        Train the model

        Parameters
        ----------
        X_train : np.ndarray
            The training data
        y_train : np.ndarray
            The training labels
        X_val : np.ndarray
            The validation data
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
                'model_weights.h5', save_best_only=True)
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
