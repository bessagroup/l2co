"""
Module containing a trainer class to evaluate the performance of the
optimization dynamics classifier.

Classes
-------
Trainer
    Class for training the optimization dynamics classifier
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from pathlib import Path
from typing import Optional

import optax
import xarray as xr
# Third-party
from optax import GradientTransformation
from sklearn.base import BaseEstimator as SingleStage0Model

# Local
from ._logging import logger
from .data_loader import DataLoader
from .stage0_model import Stage0Model
from .stage0_model import load as stage0_load
from .stage1_model import SingleStage1Model, Stage1Model
from .stage1_model import load as stage1_load

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


class Trainer:
    def __init__(self, data_train: DataLoader,
                 data_test: DataLoader,
                 stage0_clf: SingleStage0Model | str,
                 stage1_clf: SingleStage1Model | str,
                 iterations: int,
                 t_switch: int,
                 t_start: int,
                 rel_tolerance: float,
                 preprocessing_step,
                 epochs: int = 3,
                 batch_size: int = 32,
                 optimizer: GradientTransformation = optax.adam(1e-3),
                 name: str = 'trainer'):
        """
        Trainer class combining the models and the data for the data-driven
        optimization dynamics classifier

        Parameters
        ----------
        data_train : DataLoader
            Data loader containing the training data
        data_test : DataLoader
            Data loader containing the test data
        stage0_clf : sklearn.BaseEstimator
            The stage0 classifier
        stage1_clf : SingleStage1Model
            The stage1 classifier
        rel_tolerance : float, optional
            Relative tolerance, by default 0.0
        preprocessing_step : ColumnTransformer
            Preprocessing step for the categorical input features
        iterations : int
            Number of iterations of the optimization dynamics
        t_switch : int
            Iteration number of the switch
        t_start : int
            Iteration number of the start of the trajectory
        name : str, optional
            Name of the trainer, by default 'trainer'
        epochs : int, optional
            Number of epochs, by default 3
        batch_size : int, optional
            Batch size, by default 32
        optimizer : GradientTransformation, optional
            Optimizer, by default optax.adam(1e-3)
        """

        self.data_train = data_train
        self.data_test = data_test
        self.rel_tolerance = rel_tolerance
        self.iterations = iterations
        self.t_switch = t_switch
        self.t_start = t_start
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.y_pred_stage0 = None
        self.y_pred = None

        if isinstance(stage0_clf, str):
            self.stage0_model = self.load_stage0(stage0_clf)

        else:
            self.stage0_model = Stage0Model(
                model_name='stage0_model',
                preprocessing_step=preprocessing_step,
                clf=stage0_clf)

        if isinstance(stage1_clf, str):
            self.stage1_model = self.load_stage1(stage1_clf)

        else:
            self.stage1_model = Stage1Model(
                optimizers=self.data_train.select_single_optimizers()
                .optimizers,
                model=stage1_clf,
                t_start=t_start,
                t_switch=t_switch,
                rel_tolerance=rel_tolerance,
                optimizer=optimizer,
                name='stage0_model')

    def train(
            self, overwrite_performance_values: Optional[xr.DataArray] = None):
        """
        Train the models with the training data
        """
        self.train_stage0(performance=overwrite_performance_values)
        self.train_stage1()

    def train_stage0(self, performance: Optional[xr.DataArray] = None):
        """
        Train the stage0 model with the training data
        """
        X = self.data_train.X.to_pandas()
        Y = self.data_train.select_single_optimizers().y_binary(
            self.rel_tolerance).to_pandas()

        self.stage0_model.fit(X, Y)

        logger.info(f"Trained stage0 model: {self.stage0_model.name}")

    # def load_stage1(self, path: str):
    #     """
    #     Load the stage1 model
    #     """
    #     self.stage1_model.load(path)

    def train_stage1(self, save_model: bool = False):
        """
        Train the stage1 model with the training data
        """
        self.stage1_model.train(
            data_train=self.data_train,
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            optimizer=self.optimizer,
            seed=42,
            save_model=save_model,
        )

        logger.info(f"Trained stage1 model: {self.stage1_model.name}")

    def predict(self):
        """
        Predict the labels for stage 0 and stage 1 for the test data
        """
        self.stage0_prediction = self.predict_stage0()
        self.stage1_prediction = self.predict_stage1()

    def predict_stage0(self) -> xr.DataArray:
        """
        Predict the labels for stage 0 for the test data

        Returns
        -------
        xarray.DataArray
            The predicted labels
        """
        _d_test = self.data_test.select_single_optimizers()

        predictions = self.stage0_model.predict(_d_test).rename(
            f"{self.name}_stage0")

        logger.info("Predicted stage0 labels")

        return predictions

    def predict_stage1(self) -> xr.DataArray:
        """
        Predict the labels for stage 1 for the test data

        Returns
        -------
        xarray.DataArray
            The predicted labels
        """

        logger.info("Predicting stage1 labels")

        predictions = self.stage1_model.predict(
            self.data_test, self.stage0_prediction).rename(
                f"{self.name}_stage1")

        return predictions

    def save(self, path: str):
        """
        Save the models to a file

        Parameters
        ----------
        path : str
            The path to the model
        """

        _path = Path(path)

        _path.mkdir(parents=True, exist_ok=True)

        self.stage0_model.save(_path)
        self.stage1_model.save(_path)

    def load_stage0(self, path: str):
        """
        Load the stage0 model
        """
        self.stage0_model = stage0_load(path)

    def load_stage1(self, path: str):
        """
        Load the stage1 model
        """
        self.stage1_model = stage1_load(path)
