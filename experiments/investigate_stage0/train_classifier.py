
# Standard
from typing import Optional

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from create_classifier import create_classifier
from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator
from omegaconf import DictConfig
from sklearn import tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

# Local
from l2co._logging import logger as l2co_logger
from l2co.content_based import get_labels
from l2co.perf_profile import (create_performance_profile, load_all_data,
                               plot_performance_profile)
from l2co.performance import (get_performance, query_t,
                              select_optimizers_without_switch)

# make sure no plots are showing
plt.ioff()

# create a logger
logger = l2co_logger.getChild(__name__)


class ClassifierConstructer(DataGenerator):
    def __init__(self, config: DictConfig,
                 training_data: ExperimentData,
                 testing_data: Optional[ExperimentData] = None):

        logger.setLevel(config.log_level)

        self.config = config

        if testing_data is None:

            _training_index, _testing_index = train_test_split(
                training_data.index, test_size=config.test_size)

            self.experimentdata_train = training_data.select(_training_index)
            self.experimentdata_test = training_data.select(_testing_index)

        else:
            self.experimentdata_train = training_data
            self.experimentdata_test = testing_data

        self._data_train = load_all_data(self.experimentdata_train)
        self._data_test = load_all_data(self.experimentdata_test)

        # Get the number of realizations
        self.n_realizations = self._data_train.realization.size
        self.n_iterations = self._data_train.iterations.size - 1

        # Create features
        self.X_train = self.create_features(self.experimentdata_train)
        self.X_test = self.create_features(self.experimentdata_test)

        # Create performance train
        performance_train = query_t(get_performance(
            self.experimentdata_train), t=self.n_iterations)

        self.performance_train = select_optimizers_without_switch(
            performance_train)

        # Create performance test
        performance_test = query_t(get_performance(
            self.experimentdata_test), t=self.n_iterations)

        self.performance_test = select_optimizers_without_switch(
            performance_test)

    def create_features(self, experimentdata: ExperimentData) -> xr.DataArray:
        """
        Create features for each itemID in the experimentdata.

        Parameters
        ----------
        experimentdata : f3dasm.ExperimentData
            The experimentdata object.

        Returns
        -------
        xarray.DataArray
            A dataframe with the features for each itemID.
        """
        # features
        features = get_labels(
            experimentdata, n_realizations=self.n_realizations)

        logger.debug(
            f"Created features for {features.problem.size} problems")
        return features

    def create_labels(self, performance: xr.DataArray) -> xr.DataArray:
        """
        Create labels for each itemID in the experimentdata.

        Parameters
        ----------
        performance : xarray.DataArray
            The performance data.

        Returns
        -------
        xarray.DataArray
            A dataframe with the labels for each itemID.
        """
        labels = performance.idxmin(dim='optimizer').expand_dims(
            dim={'realization': np.arange(self.n_realizations)}
        ).stack(problem=['itemID', 'realization']).astype('U100')

        return labels

    def train_classifier(
            self, clf, y_train: xr.DataArray) -> tree.DecisionTreeClassifier:
        clf.fit(self.X_train, y_train)
        return clf

    def predict(self, clf) -> xr.DataArray:
        _data_test = self._data_test.stack(problem=['itemID', 'realization'])

        # Predict for testing data
        predictions = clf.predict(self.X_test)
        return xr.DataArray(predictions, dims='problem', coords={
            'problem': _data_test.problem}, name=self.create_classifier_name())

    def create_classifier_name(self) -> str:
        return self.config.project_dir

    def execute(self):
        clf = create_classifier(self.experiment_sample)

        y_train = self.create_labels(
            performance=self.performance_train)
        y_test = self.create_labels(
            performance=self.performance_test)

        clf = self.train_classifier(clf, y_train)

        da_predictions = self.predict(clf)

        # CLASSIFIER METRICS
        self.calculate_classifier_metrics(y_pred=da_predictions,
                                          y_test=y_test,
                                          clf=clf)

        performance_profile = create_performance_profile(
            ds_raw=self._data_test,
            iterations=self.n_iterations,
            strategies=[da_predictions])

        performance_profile.load()

        # Plot the performance profile
        fig_perf_profile = plot_performance_profile(
            performance_profile,
            strategy_names=[self.create_classifier_name()])

        self.experiment_sample.store(
            object=fig_perf_profile, name='performance_profile', to_disk=True)

    def calculate_classifier_metrics(
            self, y_pred: xr.DataArray, y_test: xr.DataArray, clf):
        # CLASSIFIER METRICS
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        self.experiment_sample.store(
            object=precision, name='precision', to_disk=False)
        self.experiment_sample.store(
            object=recall, name='recall', to_disk=False)
        self.experiment_sample.store(
            object=f1, name='f1', to_disk=False)
        self.experiment_sample.store(
            object=accuracy, name='accuracy', to_disk=False)

        # create confusion matrix
        confusion_matrix_ = confusion_matrix(y_test, y_pred)

        # plot the confusion matrix, and set the labels
        # to the names of the predictions
        fig_confusion_matrix, ax_confusion_matrix = plt.subplots(
            figsize=(16, 16))

        ax_confusion_matrix = sns.heatmap(confusion_matrix_,
                                          annot=True, fmt='d',
                                          xticklabels=clf.classes_,
                                          yticklabels=clf.classes_,
                                          ax=ax_confusion_matrix)

        self.experiment_sample.store(
            object=fig_confusion_matrix,
            name='confusion_matrix', to_disk=True)
