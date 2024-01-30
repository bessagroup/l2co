"""
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import Protocol

# Third-party
import numpy as np
import xarray as xr
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model, Sequential

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


class Stage1Model(Protocol):
    def create_model(self) -> None:
        ...

    def train(self, X_train: np.ndarray, X_val: np.ndarray,
              batch_size: int = 32,
              save_model: bool = False, epochs: int = 3) -> None:
        ...


class LSTMMultiOutputClassifier(Stage1Model):
    def __init__(self, optimizers: xr.DataArray, length_sequence: int,
                 optimizer: str = 'adam', latent_dim: int = 32):
        self.loss: str = 'binary_crossentropy'
        self.optimizer: str = optimizer
        self.latent_dim: int = latent_dim
        self.optimizers = optimizers
        self.length_sequence = length_sequence

        self.model: Model = None

    def create_model(self):
        input_shape = (self.length_sequence, 1)

        self.model = Sequential()

        self.model.add(Input(shape=input_shape, name='input_layer'))
        self.model.add(LSTM(units=self.latent_dim,
                       return_state=False, name='lstm_layer'))
        self.model.add(Dense(units=4, activation='sigmoid'))

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    def train(
            self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray,
            batch_size: int = 32, save_model: bool = False, epochs: int = 3):
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
        return self.model.predict(X_test)
