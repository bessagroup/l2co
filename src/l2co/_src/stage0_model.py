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
from typing import Optional

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
from .data_loader import FeaturesDataSet

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
        parameters : dict
            Optional parameters for the classifier

        Returns
        -------
        Stage0Model
            A stage0 model object
        """
        self.model_name = model_name
        self.parameters = parameters
        if preprocessing_step is None:
            self.clf = make_pipeline(MultiOutputClassifier(clf))

        else:
            if impute_nans:
                self.clf = make_pipeline(preprocessing_step,
                                         SimpleImputer(strategy='mean'),
                                         MultiOutputClassifier(clf))
            else:
                self.clf = make_pipeline(preprocessing_step,
                                         MultiOutputClassifier(clf))

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

    def train(self, feature_data: FeaturesDataSet):
        """
        Method for fitting the model

        Parameters
        ----------
        X : Iterable
            The features in array-like structure
        y : Any
            The labels in array-like structure
        """
        y = feature_data.Y.to_pandas().loc[:, feature_data.Y.to_pandas().any()]
        x = feature_data.X_features.to_pandas()
        self.clf.fit(x, y)

    def predict(self, feature_data: FeaturesDataSet) -> xr.DataArray:
        y = feature_data.Y.to_pandas()
        x = feature_data.X_features.to_pandas()
        df = pd.DataFrame(
            np.array(self.clf.predict_proba(x))[:, :, 1].T, columns=y.columns,
            index=x.index)

        return xr.DataArray(df.idxmax(axis=1)).assign_coords(
            {'itemID': feature_data.X_features.itemID}
        ).rename('stage0_predictions')

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
