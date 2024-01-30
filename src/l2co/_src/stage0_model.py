"""
Module containing various classifiers for stage0
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import Any

# Third-party
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                              ExpSineSquared, Matern)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class Stage0Model():
    def __init__(self,
                 model_name: str,
                 regressor: bool,
                 clf: Pipeline,
                 **parameters):

        self.model_name = model_name
        self.regressor = regressor
        self.parameters = parameters
        self.clf = clf

    @property
    def name(self) -> str:
        return self.create_name()

    @classmethod
    def from_name(cls, model_name: str, regressor: bool = False, **parameters):
        return cls(regressor=regressor,
                   clf=create_classifier(model_name, **parameters)
                   )

    @classmethod
    def from_sklearn_directly(cls, model_name: str,
                              clf: Any, regressor: bool = False):

        return cls(model_name=model_name, regressor=regressor, clf=clf)

    def create_name(self) -> str:
        name = self.model_name

        if self.regressor:
            name += '_reg'

        for key, value in self.parameters.items():
            name += f'_{key}={value}'

        return name

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

# =============================================================================


def create_classifier(self, model_name: str, **parameters) -> Pipeline:
    if model_name == 'DecisionTree':
        return create_decisiontree(parameters)

    elif model_name == 'SVC':
        return create_svc(parameters)

    elif model_name == 'KNN':
        return create_knn(parameters)

    elif model_name == 'GPC':
        return create_gpc(parameters)

    elif model_name == 'GPR':
        return create_gpr(parameters)

    elif model_name == 'GradientBoostingRegressor':
        return create_gradientboostingregressor(parameters)

    elif model_name == 'GradientBoostingClassifier':
        return create_gradientboostingclassifier(parameters)


def create_gradientboostingclassifier(parameters):
    learning_rate = parameters['learning_rate']
    n_estimators = parameters['n_estimators']
    min_samples_leaf = parameters['min_samples_leaf']
    max_depth = parameters['max_depth']
    return MultiOutputClassifier(
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,))


def create_gpr(parameters):
    kernel = parameters['kernel']

    if kernel == 'RBF':
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            RBF(1.0, length_scale_bounds="fixed")

    elif kernel == 'ExpSineSquared':
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            ExpSineSquared(1.0, length_scale_bounds="fixed",
                           periodicity_bounds="fixed")

    elif kernel == 'Matern':
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            Matern(1.0, length_scale_bounds="fixed", nu=1.5)

    return MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel))


def create_gradientboostingregressor(parameters):
    learning_rate = parameters['learning_rate']
    n_estimators = parameters['n_estimators']
    return MultiOutputRegressor(
        GradientBoostingRegressor(random_state=0,
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate))


def create_decisiontree(parameters):
    criterion = parameters['criterion']
    return DecisionTreeClassifier(criterion=criterion)


def create_gpc(parameters):
    return GaussianProcessClassifier()


def create_knn(parameters):
    n_neighbors = parameters['n_neighbors']
    weights = parameters['weights']
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def create_svc(parameters):
    kernel = parameters['kernel']
    clf = SVC(kernel=kernel)
    return create_multiclass_classifier(parameters, clf)


def create_multiclass_classifier(parameters, single_clf):
    multiclass_mode = parameters['multiclass_mode']

    if multiclass_mode == 'one-vs-rest':
        return OneVsRestClassifier(single_clf)

    if multiclass_mode == 'one-vs-one':
        return OneVsOneClassifier(single_clf)
