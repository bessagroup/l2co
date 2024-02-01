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
from typing import Iterable, Optional

import numpy as np
# Third-party
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_loader import DataLoader

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
            self.clf = make_pipeline(preprocessing_step,
                                     MultiOutputClassifier(clf))

    @classmethod
    def from_name(cls, model_name: str,
                  preprocessing_step: Optional[ColumnTransformer] = None,
                  **parameters):
        """
        Create a stage0 model from the name of a model

        Parameters
        ----------
        model_name : str
            The name of the model
        preprocessing_step : Optional[ColumnTransformer], optional
            Optional preprocessing step for the features, by default None
        parameters : dict
            Optional parameters for the classifier

        Returns
        -------
        Stage0Model
            A stage0 model object

        Notes
        -----

        Internally, there is a mapping from strings to scikit-learn classifiers
        """
        return cls(clf=sklearn_classifier_factory(model_name, **parameters),
                   model_name=model_name,
                   preprocessing_step=preprocessing_step,
                   parameters=parameters)

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
        self.clf.fit(X, y)

    def load(self, path: str):
        """
        Load the model from a file

        Parameters
        ----------
        path : str
            The path to the model
        """
        ...

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

        y_pred_proba = xr.DataArray(
            np.array(self.predict_proba(data_test.X.to_pandas()))[
                :, :, 1].T,
            dims=data_test.y_binary().dims,
            coords=data_test.y_binary().coords)

        p = (y_pred_proba / y_pred_proba.sum(dim='optimizer'))

        # Define a function to apply np.random.choice with probabilities from p
        def random_choice_with_probabilities(p_values):
            choice = np.random.choice(p.optimizer.values, p=p_values.ravel())
            return choice

        # Use apply_ufunc to apply the function along the desired coordinates
        result = xr.apply_ufunc(
            random_choice_with_probabilities,
            # p.unstack(),
            p,
            input_core_dims=[['optimizer']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[object],
        )

        return result

# =============================================================================


def sklearn_classifier_factory(
        self, model_name: str, **parameters) -> BaseEstimator:
    """
    Function for creating a classifier from a string

    Parameters
    ----------
    model_name : str
        The name of the model
    parameters : dict
        Optional parameters for the classifier

    Returns
    -------
    BaseEstimator
        The sklearn estimator
    """
    if model_name == 'DecisionTree':
        return create_decisiontree(parameters)

    elif model_name == 'SVC':
        return create_svc(parameters)

    elif model_name == 'KNN':
        return create_knn(parameters)

    elif model_name == 'GPC':
        return create_gpc(parameters)

    elif model_name == 'GradientBoostingClassifier':
        return create_gradientboostingclassifier(parameters)


def create_gradientboostingclassifier(parameters) -> BaseEstimator:
    """
    Function for creating a gradient boosting classifier

    Parameters
    ----------
    parameters : dict
        Hyperparameters for the classifier

    Returns
    -------
    BaseEstimator
        The sklearn estimator
    """
    learning_rate = parameters['learning_rate']
    n_estimators = parameters['n_estimators']
    min_samples_leaf = parameters['min_samples_leaf']
    max_depth = parameters['max_depth']
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,)


def create_decisiontree(parameters) -> BaseEstimator:
    """
    Function for creating a decision tree classifier

    Parameters
    ----------
    parameters : dict
        Hyperparameters for the classifier

    Returns
    -------
    BaseEstimator
        The sklearn estimator
    """
    criterion = parameters['criterion']
    return DecisionTreeClassifier(criterion=criterion)


def create_gpc(parameters) -> BaseEstimator:
    """
    Function for creating a Gaussian Process classifier

    Parameters
    ----------
    parameters : dict
        Hyperparameters for the classifier

    Returns
    -------
    BaseEstimator
        The sklearn estimator
    """
    return GaussianProcessClassifier()


def create_knn(parameters) -> BaseEstimator:
    """
    Function for creating a k-NN classifier

    Parameters
    ----------
    parameters : dict
        Hyperparameters for the classifier

    Returns
    -------
    BaseEstimator
        The sklearn estimator
    """
    n_neighbors = parameters['n_neighbors']
    weights = parameters['weights']
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def create_svc(parameters) -> BaseEstimator:
    """
    Function for creating SVC classifier

    Parameters
    ----------
    parameters : dict
        Hyperparameters for the classifier

    Returns
    -------
    BaseEstimator
        The sklearn estimator
    """
    kernel = parameters['kernel']
    return SVC(kernel=kernel)
