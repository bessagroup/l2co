"""
Module containing a trainer class to evaluate the performance of a classifier
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
                 clf: Classifier, config: DictConfig):
        self.data_train = data_train
        self.data_test = data_test
        self.clf = clf
        self.config = config

        self.name = clf.model_name
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

    def plot_performance_profile(
            self, ax=None, ax2=None, show_combinations: bool = True):
        if self.performance_profile is None:
            self.create_performance_profile()

        ax = plot_performance_profile(
            self.performance_profile,
            strategy_names=[self.name],
            ax=ax, ax2=ax2, show_combinations=show_combinations)

        return ax

    def plot_confusion_matrix(self, ax=None, ax2=None):
        if self.confusion_matrix is None:
            self.calculate_classifier_metrics()

        if ax is None:
            fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
                4, 1]},
                figsize=(16, 16))

        text = (f"Precision: {self.precision:.2f} -"
                f"Recall: {self.recall:.2f}\n"
                f"F1: {self.f1:.2f} - Accuracy: {self.accuracy:.2f}")

        ax2.text(0.5, 0.5, text, ha='center', va='center', fontsize=14)
        ax = sns.heatmap(
            self.confusion_matrix,
            annot=True, fmt='d',
            xticklabels=self.data_test.optimizers.values,
            yticklabels=self.data_test.optimizers.values,
            ax=ax)

        ax2.axis('off')

        return ax

    def plot_distribution_labels(self, ax=None, ax2=None):

        xx = self.data_test.X.unstack().stack(
            problem=['itemID', 'realization']).T

        X_df = pd.DataFrame(xx.values, columns=self.data_test.features)
        y_series = pd.Series(self.y_pred.values, name='label')

        Xy_df = pd.concat([X_df, y_series], axis=1)

        n_features = len(self.data_test.features)
        all_labels = list(set(Xy_df['label']))
        all_labels.sort()
        # Create a function to calculate label frequencies

        def get_label_frequencies(group):
            label_counts = group['label'].value_counts(
                normalize=True, sort=False)
            label_freq = {label: label_counts.get(
                label, 0.0) for label in all_labels}
            label_freq['total_count'] = len(group)
            return pd.Series(label_freq)

        df = Xy_df.groupby(list(X_df.columns)).apply(
            get_label_frequencies).reset_index()

        # Plotting horizontal bars for each row
        num_rows = len(df)
        # num_labels = len(df.columns) - 5  # Excluding the feature columns

        if ax is None:
            fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
                4, 1]}, figsize=(10, 0.5 * num_rows))

        colors = ['skyblue', 'salmon', 'lightgreen',
                  'gold', 'red']  # Define colors for each label

        yticklabels = []

        legend_patches = [mpatches.Patch(
            color=colors[i], label=label) for i, label in enumerate(
                df.columns[n_features:-1])]

        for i, (_, row) in enumerate(df.iterrows()):
            labels = row.index[n_features:-1]  # Excluding the feature columns
            freq = row.values[n_features:-1]  # Excluding the feature values
            yticklabels.append(
                ", ".join([f"{feature}: {value}" for
                           feature, value in row[:n_features].items()]))
            left = 0
            for j, (label, value) in enumerate(zip(labels, freq)):
                ax.barh(i, width=value, height=0.3, left=left,
                        color=colors[j], label=label)
                left += value

                if value != 0.0:
                    ax.text(left - value / 2, i,
                            f'{(value*row["total_count"]):.0f}',
                            ha='center', va='center', color='black')

        ax.set_yticks(np.arange(num_rows))
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('Frequency')
        ax2.legend(handles=legend_patches, loc='center',
                   ncol=len(legend_patches), fontsize='small')

        ax2.axis('off')

        ax.set_title(self.clf.name)

        return ax

    def calculate_classifier_metrics(self):
        # CLASSIFIER METRICS
        (self.precision, self.recall,
         self.f1, self.support) = precision_recall_fscore_support(
            self.data_test.y, self.y_pred, average='weighted')
        self.accuracy = accuracy_score(self.data_test.y, self.y_pred)

        # create confusion matrix
        self.confusion_matrix = confusion_matrix(self.data_test.y, self.y_pred)


class TrainerProbabilities(Trainer):
    def __init__(self, data_train: DataLoader,
                 data_test: DataLoader,
                 clf: Classifier, config: DictConfig,
                 rel_tolerance: float = 0.0):
        super().__init__(data_train=data_train,
                         data_test=data_test,
                         clf=clf,
                         config=config)

        self.rel_tolerance = rel_tolerance

    def train(self):
        self.clf.fit(self.data_train.X.T.to_pandas(),
                     self.data_train.y_binary(self.rel_tolerance).to_pandas())

    def predict(self) -> xr.DataArray:

        y_pred_proba = xr.DataArray(
            np.array(self.clf.predict_proba(self.data_test.X.T.to_pandas()))[
                :, :, 1].T,
            dims=self.data_test.y_binary(self.rel_tolerance).dims,
            coords=self.data_test.y_binary(self.rel_tolerance).coords)

        p = (y_pred_proba / y_pred_proba.sum(dim='optimizer'))

        # Define a function to apply np.random.choice with probabilities from p
        def random_choice_with_probabilities(p_values):
            choice = np.random.choice(p.optimizer.values, p=p_values.ravel())
            return choice

        # Use apply_ufunc to apply the function along the desired coordinates
        result = xr.apply_ufunc(
            random_choice_with_probabilities,
            p.unstack(),
            input_core_dims=[['optimizer']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[object],
        )

        self.y_pred = result.stack(problem=('itemID', 'realization')).astype(
            'U100').rename(self.name)

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

        self.experiment_sample.store(object=trainer.plot_confusion_matrix(),
                                     name="confusion_matrix",
                                     to_disk=True)

        self.experiment_sample.store(object=trainer.plot_performance_profile(),
                                     name="performance_profile",
                                     to_disk=True)

        self.experiment_sample.store(object=trainer.plot_distribution_labels(),
                                     name="distribution_labels",
                                     to_disk=True)


class ClassifierConstructorProba(DataGenerator):
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

        clf = Classifier.from_name(model_name=model_name,
                                   regressor=self.config.regression,
                                   **parameters)
        trainer = TrainerProbabilities(
            data_train=self.data_train, data_test=self.data_test,
            clf=clf, config=self.config,
            rel_tolerance=self.config.rel_tolerance)

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

        fig, ax = plt.subplots(4, 2, figsize=(15, 15), gridspec_kw={
                               'height_ratios': [1, 0.25, 1, 0.25]})
        axs = ax
        _ = trainer.plot_confusion_matrix(ax=axs[0, 0], ax2=axs[1, 0])
        _ = trainer.plot_performance_profile(
            ax=axs[0, 1], ax2=axs[1, 1], show_combinations=False)
        _ = trainer.data_test.plot_performance_labels(
            rel_tolerance=self.config.rel_tolerance,
            ax=axs[2, 0], ax2=axs[3, 0],
            y_pred=trainer.y_pred, set_yticklabels=False)
        _ = trainer.data_test.plot_performance_labels(
            rel_tolerance=self.config.rel_tolerance,
            ax=axs[2, 1], ax2=axs[3, 1], set_yticklabels=False)

        self.experiment_sample.store(object=fig, name="fig", to_disk=True)
