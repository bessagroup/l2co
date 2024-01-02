
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
from l2co.performance import (get_performance, get_performance_preprocessed,
                              query_t, select_optimizers_without_switch)

# make sure no plots are showing
plt.ioff()

# create a logger
logger = l2co_logger.getChild(__name__)


class ClassifierConstructer(DataGenerator):
    def __init__(self, X_train: xr.DataArray, X_test: xr.DataArray,
                 y_train: xr.DataArray, y_test: xr.DataArray,
                 n_realizations: int, n_iterations: int,
                 optimizers: xr.DataArray, _data_test: xr.Dataset,
                 config: DictConfig):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_realizations = n_realizations
        self.n_iterations = n_iterations
        self.optimizers = optimizers
        self._data_test = _data_test
        self.config = config

        logger.setLevel(config.log_level)

    @classmethod
    def from_experimentdata(cls, config: DictConfig,
                            training_data: ExperimentData,
                            testing_data: Optional[ExperimentData] = None):

        if testing_data is None:
            _training_index, _testing_index = train_test_split(
                training_data.index, test_size=config.test_size)

            experimentdata_train = training_data.select(_training_index)
            experimentdata_test = training_data.select(_testing_index)

        else:
            experimentdata_train = training_data
            experimentdata_test = testing_data

        _data_train = load_all_data(experimentdata_train)
        _data_test = load_all_data(experimentdata_test)

        # Get the number of realizations
        n_realizations = _data_train.realization.size
        n_iterations = _data_train.iterations.size - 1

        # Create features
        X_train = create_features(experimentdata_train, n_realizations)
        X_test = create_features(experimentdata_test, n_realizations)

        # Create performance train
        performance_train = query_t(get_performance(
            experimentdata_train), t=n_iterations)

        performance_train = select_optimizers_without_switch(
            performance_train)

        # Create performance test
        performance_test = query_t(get_performance(
            experimentdata_test), t=n_iterations)

        performance_test = select_optimizers_without_switch(
            performance_test)

        # Create labels
        y_train = create_labels(
            performance=performance_train,
            n_realizations=n_realizations,
            regression=config.regression)
        y_test = create_labels(
            performance=performance_test,
            n_realizations=n_realizations,
        )

        # Optimizers
        optimizers = performance_test.optimizer

        return cls(X_train=X_train, X_test=X_test,
                   y_train=y_train, y_test=y_test,
                   n_realizations=n_realizations,
                   n_iterations=n_iterations,
                   optimizers=optimizers,
                   _data_test=_data_test,
                   config=config)

    @classmethod
    def from_preprocessed_files(cls, config: DictConfig,
                                experimentdata: ExperimentData,
                                ds: xr.Dataset, ds_samples: xr.Dataset):

        n_realizations = len(ds.realization)
        n_iterations = len(ds.iterations)-1

        _training_index, _testing_index = train_test_split(
            experimentdata.index, test_size=config.test_size)

        experimentdata_train = experimentdata.select(_training_index)
        experimentdata_test = experimentdata.select(_testing_index)

        ds_train = ds.sel(itemID=_training_index)
        ds_test = ds.sel(itemID=_testing_index)

        ds_samples_train = ds_samples.sel(itemID=_training_index)
        ds_samples_test = ds_samples.sel(itemID=_testing_index)

        # Performance
        performance_train = query_t(get_performance_preprocessed(
            ds_train, ds_samples_train), t=len(ds_train.iterations)-1)
        performance_train = select_optimizers_without_switch(performance_train)

        performance_test = query_t(get_performance_preprocessed(
            ds_test, ds_samples_test), t=len(ds_test.iterations)-1)

        performance_test = select_optimizers_without_switch(performance_test)

        # Optimizers
        optimizers = performance_test.optimizer

        # Features
        X_train = get_labels(experimentdata_train,
                             n_realizations=n_realizations)

        X_test = get_labels(experimentdata_test,
                            n_realizations=n_realizations)

        # Labels
        y_train = performance_train.idxmin(dim='optimizer').expand_dims(
            dim={'realization': np.arange(n_realizations)}
        ).stack(problem=['itemID', 'realization']).astype('U100')

        y_test = performance_test.idxmin(dim='optimizer').expand_dims(
            dim={'realization': np.arange(n_realizations)}
        ).stack(problem=['itemID', 'realization']).astype('U100')

        return cls(X_train=X_train, X_test=X_test,
                   y_train=y_train, y_test=y_test,
                   n_realizations=n_realizations,
                   n_iterations=n_iterations,
                   optimizers=optimizers,
                   _data_test=ds_test,
                   config=config)

    def train_classifier(
            self, clf) -> tree.DecisionTreeClassifier:
        clf.fit(self.X_train, self.y_train)
        return clf

    def predict(self, clf, optimizers,
                regression: bool = False) -> xr.DataArray:
        _data_test = self._data_test.stack(problem=['itemID', 'realization'])

        # Predict for testing data
        predictions = clf.predict(self.X_test)

        if regression:

            xarr_predictions = xr.DataArray(
                predictions, dims=['problem', 'optimizer'],
                coords={'problem': _data_test.problem,
                        'optimizer': optimizers},
                name=self.create_classifier_name())

            return xarr_predictions.idxmin(
                dim='optimizer', keep_attrs=True).rename(
                    self.create_classifier_name())

        else:
            return xr.DataArray(predictions, dims='problem', coords={
                'problem': _data_test.problem},
                name=self.create_classifier_name())

    def create_classifier_name(self) -> str:
        name = '_'.join(str(v)
                        for v in self.experiment_sample.input_data.values())
        return name

    def execute(self):
        clf = create_classifier(self.experiment_sample)

        clf = self.train_classifier(clf)

        da_predictions = self.predict(
            clf, optimizers=self.optimizers,
            regression=self.config.regression)

        if self.config.debug:
            self.experiment_sample.store(
                object=da_predictions.unstack('problem'),
                name='da_predictions', to_disk=True)

        # CLASSIFIER METRICS
        self.calculate_classifier_metrics(y_pred=da_predictions,
                                          clf=clf)

        performance_profile = create_performance_profile(
            ds_raw=self._data_test,
            iterations=self.n_iterations,
            strategies=[da_predictions])

        performance_profile.load()

        self.experiment_sample.store(
            object=performance_profile.to_pandas(),
            name='performance_profile_csv', to_disk=True)

        # Plot the performance profile
        fig_perf_profile = plot_performance_profile(
            performance_profile,
            strategy_names=[self.create_classifier_name()])

        self.experiment_sample.store(
            object=fig_perf_profile, name='performance_profile', to_disk=True)

    def calculate_classifier_metrics(
            self, y_pred: xr.DataArray, clf):
        # CLASSIFIER METRICS
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted')
        accuracy = accuracy_score(self.y_test, y_pred)

        self.experiment_sample.store(
            object=precision, name='precision', to_disk=False)
        self.experiment_sample.store(
            object=recall, name='recall', to_disk=False)
        self.experiment_sample.store(
            object=f1, name='f1', to_disk=False)
        self.experiment_sample.store(
            object=accuracy, name='accuracy', to_disk=False)

        # create confusion matrix
        confusion_matrix_ = confusion_matrix(self.y_test, y_pred)

        # plot the confusion matrix, and set the labels
        # to the names of the predictions
        fig_confusion_matrix, ax_confusion_matrix = plt.subplots(
            figsize=(16, 16))

        ax_confusion_matrix = sns.heatmap(
            confusion_matrix_,
            annot=True, fmt='d',
            xticklabels=self.optimizers.values,
            yticklabels=self.optimizers.values,
            ax=ax_confusion_matrix)

        self.experiment_sample.store(
            object=fig_confusion_matrix,
            name='confusion_matrix', to_disk=True)


def create_features(
        experimentdata: ExperimentData, n_realizations: int) -> xr.DataArray:
    """
    Create features for each itemID in the experimentdata.

    Parameters
    ----------
    experimentdata : f3dasm.ExperimentData
        The experimentdata object.
    n_realizations : int
        The number of realizations.

    Returns
    -------
    xarray.DataArray
        A dataframe with the features for each itemID.
    """
    # features
    features = get_labels(
        experimentdata, n_realizations=n_realizations)

    logger.debug(
        f"Created features for {features.problem.size} problems")
    return features


def create_labels(performance: xr.DataArray, n_realizations: int,
                  regression: bool = False) -> xr.DataArray:
    """
    Create labels for each itemID in the experimentdata.

    Parameters
    ----------
    performance : xarray.DataArray
        The performance data.
    n_realizations : int
        The number of realizations.
    regression : bool, optional
        Whether to return regression labels instead
        of classification labels, by default False

    Returns
    -------
    xarray.DataArray
        A dataframe with the labels for each itemID.
    """
    if regression:
        return performance.expand_dims(
            dim={'realization': np.arange(n_realizations)}
        ).stack(problem=['itemID', 'realization']).T

    else:
        return performance.idxmin(dim='optimizer').expand_dims(
            dim={'realization': np.arange(n_realizations)}
        ).stack(problem=['itemID', 'realization']).astype('U100')
