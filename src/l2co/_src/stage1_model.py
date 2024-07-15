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
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Tuple

# Third-party
import numpy as np
import xarray as xr
from joblib import dump, load
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential, load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

# Local
from .data_loader import DataLoader
from .ela import ELA_FEATURE_LABELS
from .preprocessing import preprocessing_factory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================

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

    def load(self, path: Path) -> None:
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
                 name: str = 'stage1_model', **kwargs):
        self.optimizers = optimizers
        self.model = model
        self.models = self.create_models()
        self.name = name

        self.iterations = iterations
        self.t_start = t_start
        self.t_switch = t_switch
        self.rel_tolerance = rel_tolerance

    def preprocess_data(self, data: DataLoader, optimizer: str
                        ) -> Tuple[np.ndarray, np.ndarray]:
        da_X = data.select_optimizer(optimizer).select_t(
            self.t_switch, t_start=self.t_start).X_trajectory(
            self.t_switch, t_start=self.t_start).dropna('problem')
        da_y = data.query_optimizer(optimizer).select_problems(
            da_X.problem).y_binary(self.rel_tolerance)

        X = np.atleast_3d(da_X)
        X = np.transpose(X, (2, 1, 0))

        y = np.atleast_2d(da_y.astype(int))

        return X, y

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
        d_train, d_val = data_train.train_test_split(
            test_size=0.2, random_state=42
        )

        for optimizer in self.optimizers.values:
            X_train, y_train = self.preprocess_data(d_train, optimizer)
            X_val, y_val = self.preprocess_data(d_val, optimizer)

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

            data_selected = data_test.select_problems(problems)

            da_X_train = data_selected.select_optimizer(optimizer).select_t(
                self.t_switch, t_start=self.t_start).X_trajectory(
                    self.t_switch, t_start=self.t_start)

            X_test = np.atleast_3d(da_X_train)
            X_test = np.transpose(X_test, (2, 1, 0))

            list_y_pred_stage1.append(xr.DataArray(
                self.models[optimizer].predict(X_test),
                dims=['problem', 'optimizer'],
                coords={'optimizer': self.optimizers,
                        'problem': data_selected.problem})
            )

        stage1_predictions = xr.concat(
            list_y_pred_stage1, dim='problem').sortby('problem')

        # Select the optimizer with the highest probability
        stage1_predictions = stage1_predictions.idxmax(
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

    def load(self, path: str) -> None:
        """
        Load the model parameters from a file

        Parameters
        ----------
        path : str
            The path to the model
        """
        for optimizer in self.optimizers.values:
            self.models[optimizer].load(
                Path(path) / f'{self.name}_{optimizer}')


#                                                         Model implementations
# =============================================================================

class LSTMMultiOutputClassifier(SingleStage1Model):
    def __init__(self, n_outputs: int, length_sequence: int,
                 optimizer: str = 'adam', latent_dim: int = 32, **kwargs):
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

    def load(self, path: Path):
        """
        Load the model
        """
        self.model = load_model(f'{path}.h5')


class ELAClassifier(Stage1Model):
    def __init__(self, iterations: int, rel_tolerance: float, t_switch: int,
                 name: str = 'stage1_model'):
        self.model = self.create_model()
        self.name = name
        self.iterations = iterations
        self.t_switch = t_switch
        self.rel_tolerance = rel_tolerance

    def create_model(self):
        """
        Instantiate the model
        """
        clf = RandomForestClassifier()
        preprocessing_step = preprocessing_factory(
            feature_types=ELA_FEATURE_LABELS)
        return make_pipeline(preprocessing_step,
                             SimpleImputer(strategy='mean'),
                             MultiOutputClassifier(clf))

    def train(self, data_train: DataLoader, features: xr.DataArray,
              save_model: bool = False,
              ) -> None:
        """
        Protocol method for training the model
        """
        X_train = features.load().stack(problem=('realization', 'itemID'))
        X_train = xr.where(np.isinf(X_train), np.nan, X_train)

        X_train = X_train.sel(problem=data_train.problem)

        y_train = data_train.select_single_optimizers().y_binary(
            self.rel_tolerance)

        y_train = y_train.sel(problem=X_train.problem)

        self.model.fit(X_train.to_pandas().T, y_train.to_pandas())

    def load(self, model_path: str) -> None:
        """
        Protocol method for loading the model
        """
        self.model = load(f'{model_path}.joblib')

    def predict_proba(self, X: Iterable) -> Iterable[float]:
        """
        Method for predicting the probabilities

        Parameters
        ----------
        X : Iterable
            The features in array-like structure

        Returns
        -------
        Iterable[float]
            The probabilities
        """
        return self.model.predict_proba(X)

    def predict(
        self, data_test: DataLoader,
        features: xr.DataArray,
        stage0_predictions: xr.DataArray = None,
    ) -> xr.DataArray:
        """
        Protocol method for predicting
        """
        y_shape = data_test.select_single_optimizers().y_binary()

        X_test = features.load().stack(problem=('realization', 'itemID'))
        X_test = xr.where(np.isinf(X_test), np.nan, X_test)

        X_test = X_test.sel(problem=data_test.problem)
        y_pred = self.model.predict_proba(X_test.to_pandas().T)

        xarr_y_pred = xr.DataArray(
            np.array(y_pred)[:, :, 1].T,
            coords=y_shape.coords).idxmax('optimizer')

        return xr.apply_ufunc(lambda x: 'random_samples+' + x, xarr_y_pred)

    def save(self) -> None:
        """
        Protocol method for saving the model
        """
        dump(self.model, f'{self.name}.joblib')
