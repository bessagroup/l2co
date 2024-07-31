"""
Module containing various classifiers for stage0

Classes
-------
Stage0Model
    Class for creating a stage0 model

Functions
---------
create_classifier
    Function for creating a classifier
create_decisiontree
    Function for creating a decision tree classifier
create_gpc
    Function for creating a Gaussian process classifier
create_knn
    Function for creating a k-nearest neighbors classifier
create_svc
    Function for creating a support vector classifier
create_gradientboostingclassifier
    Function for creating a gradient boosting classifier
"""

#                                                                       Modules
# =============================================================================

# Standard
import pickle
from pathlib import Path
from typing import Iterable, Optional

# Third-party
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

# Local
from l2co.data import DataLoader

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

    def fit(self, X: Iterable, y: Iterable):
        """
        Method for fitting the model

        Parameters
        ----------
        X : Iterable
            The features in array-like structure
        y : Any
            The labels in array-like structure
        """
        y = y.loc[:, y.any()]
        self.clf.fit(X, y)

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
        return self.clf.predict_proba(X)

    def predict(self, data_test: DataLoader) -> xr.DataArray:
        """
        Predict the labels for stage 0 for the test data
        """

        # y = data_test.y_binary()

        y_pred_proba = xr.DataArray(
            np.array(self.predict_proba(data_test.X.to_pandas()))[
                :, :, 1].T,
            dims=data_test.y_binary().dims,
            coords=data_test.y_binary().coords)

        return y_pred_proba.idxmax(dim='optimizer').sortby('problem')

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
