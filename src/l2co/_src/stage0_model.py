"""
Module containing the model for the stage0 decision.

Classes
-------
Stage0Model
    Class for creating a stage0 model

Functions
---------
load
    Function to load a Stage0Model from a file
"""

#                                                                       Modules
# =============================================================================

# Standard
import pickle
from pathlib import Path
from typing import Literal, Optional

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

# Local
# from .data_loader import FeaturesDataSet

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class Stage0Model():
    def __init__(self,
                 model_name: str,
                 clf: BaseEstimator,
                 preprocessing_step: Optional[ColumnTransformer] = None,
                 impute_nans: bool = False,
                 type: Literal['classifier', 'regressor'] = 'classifier',
                 **parameters):
        """
        Class for creating a stage0 model

        Parameters
        ----------
        model_name : str
            The name of the model
        clf : BaseEstimator
            The scikit-learn classifier object
        preprocessing_step : Optional[ColumnTransformer], optional
            Optional preprocessing step for the features, by default None
        impute_nans : bool, optional
            If True, impute NaNs in the data, by default False
        type : Literal['classifier', 'regressor'], optional
            The type of model, by default 'classifier'


        Returns
        -------
        Stage0Model
            A stage0 model object
        """
        self.model_name = model_name
        self.parameters = parameters
        self.type = type

        if self.type == 'classifier':
            model = MultiOutputClassifier(clf)

        elif self.type == 'regressor':
            model = clf

        if preprocessing_step is None:
            self.clf = make_pipeline(model)

        else:
            if impute_nans:
                self.clf = make_pipeline(preprocessing_step,
                                         SimpleImputer(strategy='mean'),
                                         model)
            else:
                self.clf = make_pipeline(preprocessing_step,
                                         model)

    @property
    def name(self) -> str:
        """
        Property that returns the name of the model

        Returns
        -------
        str
            The name of the model
        """
        return self.create_name()

    def create_name(self) -> str:
        """
        Method that creates a name for the model

        Returns
        -------
        str
            The name of the model

        Notes
        -----
        The name of the model is created by concatenating the model_name and
        the chosen hyperparameters.
        """
        name = self.model_name

        for key, value in self.parameters.items():
            name += f'_{key}={value}'

        return name

    def train(self, feature_data: object):
        """
        Method for fitting the model

        Parameters
        ----------
        X : Iterable
            The features in array-like structure
        y : Any
            The labels in array-like structure
        """
        if self.type == 'classifier':
            y = feature_data.Y.to_pandas(
            ).loc[:, feature_data.Y.to_pandas().any()]
        elif self.type == 'regressor':
            y = feature_data.Y_features.to_pandas()
            # Subtract the mean to emphasize the differences between
            # the performances
            y = y.subtract(y.mean(axis=1), axis=0)

        x = feature_data.X_features.to_pandas()

        self.clf.fit(x, y)

    def predict(self, feature_data: object) -> xr.DataArray:
        if self.type == 'classifier':
            y = feature_data.Y.to_pandas()

        elif self.type == 'regressor':
            y = feature_data.Y_features.to_pandas()

        x = feature_data.X_features.to_pandas()

        if self.type == 'classifier':
            y_pred = self.clf.predict_proba(x)
            df = pd.DataFrame(
                np.array(y_pred)[
                    :, :, 1].T, columns=y.columns,
                index=x.index)

            return xr.DataArray(df.idxmax(axis=1)).assign_coords(
                {'itemID': feature_data.X_features.itemID}
            ).rename('stage0_predictions')

        elif self.type == 'regressor':
            y_pred = self.clf.predict(x)
            df = pd.DataFrame(y_pred, columns=y.columns, index=x.index)

            return xr.DataArray(df.idxmin(axis=1)).assign_coords(
                {'itemID': feature_data.X_features.itemID}
            ).rename('stage0_predictions')

    def predict_raw(self, feature_dataset: object) -> xr.DataArray:
        y = feature_dataset.Y_features.to_pandas()
        x = feature_dataset.X_features.to_pandas()

        y_pred = pd.DataFrame(self.clf.predict(
            x), columns=y.columns, index=x.index)

        # y_pred = y_pred.add(y.mean(axis=1), axis=0)
        return xr.DataArray(y_pred).assign_coords(
            {'itemID': feature_dataset.X_features.itemID})

    def save(self, path: Path) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        path : Path
            The path to the model
        """
        filename = (path / 'stage0_model').with_suffix('.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def load(path: Path) -> Stage0Model:
    """
    Load the model from a file

    Parameters
    ----------
    path : Path
        The path to the model
    """
    filename = (path / 'stage0_model').with_suffix('.pkl')
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model
