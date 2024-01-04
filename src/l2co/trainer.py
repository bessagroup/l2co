"""
Module containing a trainer class to evaluate the performance of a classifier
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Third-party
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from f3dasm.datageneration import DataGenerator
from omegaconf import DictConfig
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

# Local
from ._logging import logger as l2co_logger
from .classifier import Classifier
from .data_loader import DataLoader
from .perf_profile import create_performance_profile, plot_performance_profile

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


# make sure no plots are showing
plt.ioff()

# create a logger
logger = l2co_logger.getChild(__name__)

# =============================================================================


class Trainer():
    def __init__(self, data_train: DataLoader,
                 data_test: DataLoader,
                 clf: Classifier, config: DictConfig,
                 name: str = 'classifier'):
        self.data_train = data_train
        self.data_test = data_test
        self.clf = clf
        self.config = config
        self.name = name

        self.y_pred = None
        self.precision = None
        self.f1 = None
        self.recall = None
        self.support = None
        self.accuracy = None
        self.confusion_matrix = None
        self.performance_profile = None

    def train(self):
        """
        Train the classifier.

        Parameters
        ----------
        clf : tree.DecisionTreeClassifier
            The classifier.
        """
        if self.clf.regressor:
            self.clf.fit(self.data_train.X, self.data_train.y_reg)
        else:
            self.clf.fit(self.data_train.X.T, self.data_train.y)

    def predict(self) -> xr.DataArray:

        # Predict for testing data
        predictions = self.clf.predict(self.data_test.X.T)

        if self.clf.regressor:

            xarr_predictions = xr.DataArray(
                predictions, dims=['problem', 'optimizer'],
                coords={'problem': self.data_test.problem,
                        'optimizer': self.data_test.optimizer},
                name=self.name)

            self.y_pred = xarr_predictions.idxmin(
                dim='optimizer', keep_attrs=True).rename(
                    self.name)

        else:
            self.y_pred = xr.DataArray(predictions, dims='problem', coords={
                'problem': self.data_test.problem},
                name=self.name)

    def create_performance_profile(self):

        self.performance_profile = create_performance_profile(
            ds_raw=self.data_test.data,
            iterations=self.data_test.n_iterations,
            strategies=[self.y_pred])

        self.performance_profile.load()

    def plot_performance_profile(self):
        if self.performance_profile is None:
            self.create_performance_profile()

        fig_perf_profile = plot_performance_profile(
            self.performance_profile,
            strategy_names=[self.name])

        return fig_perf_profile

    def plot_confusion_matrix(self):
        if self.confusion_matrix is None:
            self.calculate_classifier_metrics()

        fig_confusion_matrix, ax_confusion_matrix = plt.subplots(
            figsize=(16, 16))

        ax_confusion_matrix = sns.heatmap(
            self.confusion_matrix,
            annot=True, fmt='d',
            xticklabels=self.data_test.optimizers.values,
            yticklabels=self.data_test.optimizers.values,
            ax=ax_confusion_matrix)

        return fig_confusion_matrix

    def calculate_classifier_metrics(self):
        # CLASSIFIER METRICS
        (self.precision, self.recall,
         self.f1, self.support) = precision_recall_fscore_support(
            self.data_test.y, self.y_pred, average='weighted')
        self.accuracy = accuracy_score(self.data_test.y, self.y_pred)

        # create confusion matrix
        self.confusion_matrix = confusion_matrix(self.data_test.y, self.y_pred)

# =============================================================================


class ClassifierConstructor(DataGenerator):
    def __init__(self, config: DictConfig,
                 data_train: DataLoader,
                 data_test: DataLoader):
        self.config = config
        self.data_train = data_train
        self.data_test = data_test

    def execute(self):
        model_name = self.experiment_sample.get('classifier')
        parameters = self.experiment_sample.input_data
        del parameters['classifier']

        clf = Classifier(model_name=model_name,
                         regressor=self.config.regression, **parameters)
        trainer = Trainer(data_train=self.data_train, data_test=self.data_test,
                          clf=clf, config=self.config)

        trainer.train()
        trainer.predict()

        trainer.calculate_classifier_metrics()
        trainer.create_performance_profile()

        if self.config.debug:
            self.experiment_sample.store(
                object=trainer.y_pred.unstack('problem'),
                name='da_predictions', to_disk=True
            )

        self.experiment_sample.store(
            object=trainer.precision, name='precision')
        self.experiment_sample.store(object=trainer.recall, name='recall')
        self.experiment_sample.store(object=trainer.f1, name='f1')
        self.experiment_sample.store(object=trainer.accuracy, name='accuracy')
        self.experiment_sample.store(
            object=trainer.performance_profile.to_pandas(),
            name='performance_profile_csv', to_disk=True)

        self.experiment_sample.store(object=trainer.plot_confusion_matrix(), name="confusion_matrix",
                                     to_disk=True)

        self.experiment_sample.store(object=trainer.plot_performance_profile(), name="performance_profile",
                                     to_disk=True)
