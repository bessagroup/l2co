"""
Module containing a trainer class to evaluate the performance of the
optimization dynamics classifier.

Classes
-------
TrainerBase
    Base class to train and evaluate the performance of a classifier
TrainerStage1
    Implementation of the Trainer class for
    stage 0 and stage 1 of the classifier


"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from typing import Iterable, Optional, Tuple

# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

# Local
from .data_loader import DataLoader
from .lstm_autoencoder import SingleStage1Model, Stage1Model
from .perf_profile import create_performance_profile, plot_performance_profile
from .stage0_model import Stage0Model

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


class Trainer:
    def __init__(self, data_train: DataLoader,
                 data_test: DataLoader,
                 stage0_clf: BaseEstimator,
                 stage1_clf: SingleStage1Model,
                 iterations: int,
                 t_switch: int,
                 t_start: int,
                 rel_tolerance: float,
                 preprocessing_step,
                 epochs: int = 3,
                 batch_size: int = 32,
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

        self.y_pred_stage0 = None
        self.y_pred = None
        self.stage1_clf = None

        self.precision = None
        self.f1 = None
        self.recall = None
        self.support = None
        self.accuracy = None
        self.confusion_matrix = None
        self.performance_profile = None

        self.stage0_model = Stage0Model(
            model_name='stage0_model',
            preprocessing_step=preprocessing_step,
            clf=stage0_clf)

        self.stage1_model = Stage1Model(
            optimizers=self.data_train.select_single_optimizers().optimizers,
            model=stage1_clf,
            iterations=iterations,
            t_start=t_start,
            t_switch=t_switch,
            rel_tolerance=rel_tolerance,
            name='stage0_model')

    def train(self):
        """
        Train the models with the training data
        """
        self.train_stage0()
        self.train_stage1()

    def train_stage0(self):
        """
        Train the stage0 model with the training data
        """
        _d_train = self.data_train.select_single_optimizers().query_t(
            self.iterations)

        self.stage0_model.fit(_d_train.X.to_pandas(),
                              _d_train.y_binary(self.rel_tolerance).to_pandas()
                              )

    def train_stage1(self):
        """
        Train the stage1 model with the training data
        """
        self.stage1_model.train(self.data_train, batch_size=self.batch_size,
                                epochs=self.epochs)

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
        _d_test = self.data_test.select_single_optimizers().query_t(
            self.iterations)

        return self.stage0_model.predict(_d_test).rename(
            f"{self.name}_stage0")

    def predict_stage1(self) -> xr.DataArray:
        """
        Predict the labels for stage 1 for the test data

        Returns
        -------
        xarray.DataArray
            The predicted labels
        """
        return self.stage1_model.predict(
            self.data_test, self.stage0_prediction).rename(
                f"{self.name}_stage1")

    def plot_stage0(self):
        """
        Plot the performance profile for the classifier for stage 0
        """
        return self.plot(
            y_true=self.data_test.select_single_optimizers().query_t(
                self.iterations).y,
            y_pred=self.stage0_prediction,
            strategies=self.stage0_prediction,
            optimizers=self.data_test.select_single_optimizers().optimizers,
            data_test=self.data_test.select_single_optimizers().query_t(
                self.iterations),
            strategy_names=[f"{self.name}_stage0"])

    def plot_stage1(self):
        """
        Plot the performance profile for the classifier for stage 1
        """
        return self.plot(
            y_true=self.data_test.query_t(
                self.iterations).y,
            y_pred=self.stage1_prediction,
            strategies=self.stage1_prediction,
            optimizers=self.data_test.optimizers,
            data_test=self.data_test.query_t(self.iterations),
            strategy_names=[f"{self.name}_stage1"])

    def create_performance_profile(
            self, strategies: Iterable[xr.DataArray]):
        """
        Create a performance profile for the classifier and loads it into
        memory

        Parameters
        ----------
        strategies : Optional[Iterable[xr.DataArray]], optional
            Additional strategies to include in the performance profile,
            by default None
        """
        data_test = self.data_test.data.sortby('problem')
        best_perf_profile = self.data_test.query_t(
            self.iterations).y.rename('best_perf_profile')
        worst_perf_profile = self.data_test.select_single_optimizers().query_t(
            self.iterations).performance.idxmax(
                dim='optimizer').rename('worst_perf_profile')

        _strategies = [best_perf_profile, worst_perf_profile, *strategies]

        self.performance_profile = create_performance_profile(
            ds_raw=data_test,
            iterations=self.data_test.n_iterations,
            strategies=_strategies)

        self.performance_profile.load()

    def plot_performance_profile(
            self,
            strategies: Iterable[xr.DataArray],
            strategy_names: Iterable[str],
            axes: Optional[Iterable[plt.Axes, plt.Axes]] = None,
            show_combinations: bool = True,
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Plot the performance profile for the classifier

        Parameters
        ----------
        axes : Optional[Tuple[plt.Axes, plt.Axes]], optional
            Tuple containing two matplotlib axes objects, by default None
        strategy_names : Iterable[str]
            Names of the strategies to be included in the performance profile
        strategies : Iterable[xr.DataArray]
            Additional strategies to include in the performance profile
        show_combinations : bool, optional
            Whether to show the combinations of strategies, by default True

        Returns
        -------
        Tuple[plt.Axes, plt.Axes]
            Matplotlib axes object containing the performance profile

        Notes
        -----

        - If no axes are provided, a new figure will be created
        - If no performance profile is available, it will be created
        """

        if self.performance_profile is None:
            self.create_performance_profile(strategies=strategies)

        ax = plot_performance_profile(
            self.performance_profile,
            strategy_names=[
                'best_perf_profile', 'worst_perf_profile', *strategy_names
            ], axes=axes,
            show_combinations=show_combinations)

        return ax

    def plot_confusion_matrix(
            self,
            y_true: xr.DataArray,
            y_pred: xr.DataArray,
            optimizers: xr.DataArray,
            axes: Optional[Iterable[plt.Axes, plt.Axes]] = None
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Plot the confusion matrix for the classifier

        Parameters
        ----------
        y_true : xarray.DataArray
            True labels
        y_pred : xarray.DataArray
            Predicted labels
        optimizers : xarray.DataArray
            Optimizers
        axes : Optional[Tuple[plt.Axes, plt.Axes]], optional
            Tuple containing two matplotlib axes objects, by default None

        Returns
        -------
        Tuple[plt.Axes, plt.Axes]
            Matplotlib axes object containing the confusion matrix

        Notes
        -----
        - If no axes are provided, a new figure will be created
        - If no confusion matrix is available, it will be created
        """
        if self.confusion_matrix is None:
            self.calculate_classifier_metrics(
                y_true=y_true, y_pred=y_pred, optimizers=optimizers)

        if axes is None:
            fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
                4, 1]},
                figsize=(16, 16))

        text = (f"Precision: {self.precision:.2f} -"
                f"Recall: {self.recall:.2f}\n"
                f"F1: {self.f1:.2f} - Accuracy: {self.accuracy:.2f}")

        axes[1].text(0.5, 0.5, text, ha='center', va='center', fontsize=14)
        axes[0] = sns.heatmap(
            self.confusion_matrix,
            annot=True, fmt='d',
            xticklabels=optimizers.values,
            yticklabels=optimizers.values,
            ax=axes[0])

        axes[1].axis('off')

        return axes

    def plot_distribution_labels(
            self,
            axes: Optional[Iterable[plt.Axes, plt.Axes]] = None
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Plot the distribution of labels for the classifier

        Parameters
        ----------
        axes : Optional[Tuple[plt.Axes, plt.Axes]], optional
            Tuple containing two matplotlib axes objects, by default None

        Returns
        -------
        Tuple[plt.Axes, plt.Axes]
            Matplotlib axes object containing the distribution of labels

        Notes
        -----
        - If no axes are provided, a new figure will be created
        """

        xx = self.data_test.X.unstack().stack(
            problem=['itemID', 'realization']).sortby('problem').T

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

        if axes is None:
            fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
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
                axes[0].barh(i, width=value, height=0.3, left=left,
                             color=colors[j], label=label)
                left += value

                if value != 0.0:
                    axes[0].text(left - value / 2, i,
                                 f'{(value*row["total_count"]):.0f}',
                                 ha='center', va='center', color='black')

        axes[0].set_yticks(np.arange(num_rows))
        axes[0].set_yticklabels(yticklabels)
        axes[0].set_xlabel('Frequency')
        axes[1].legend(handles=legend_patches, loc='center',
                       ncol=len(legend_patches), fontsize='small')

        axes[1].axis('off')

        axes[0].set_title(self.stage0model.name)

        return axes

    def calculate_classifier_metrics(
            self, y_true: xr.DataArray,
            y_pred: xr.DataArray, optimizers: xr.DataArray):
        """
        Calculate the classifier metrics y_pred

        Parameters
        ----------
        y_true : xr.DataArray
            True labels
        y_pred : xr.DataArray
            Predicted labels
        optimizers : xr.DataArray
            Optimizers

        Notes
        -----
        - If no predictions are available, they will be created by calling
            self.predict()
        """
        # CLASSIFIER METRICS
        (self.precision, self.recall,
         self.f1, self.support) = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')
        self.accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        # create confusion matrix
        self.confusion_matrix = confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=optimizers)

    def plot(self, y_true: xr.DataArray, y_pred: xr.DataArray,
             strategies: xr.DataArray, strategy_names: xr.DataArray,
             optimizers: xr.DataArray, data_test: DataLoader,
             show_combinations: bool = False):
        fig, axes = plt.subplots(4, 2, figsize=(15, 15), gridspec_kw={
            'height_ratios': [1, 0.25, 1, 0.25]})

        _ = self.plot_confusion_matrix(y_true=y_true, y_pred=y_pred,
                                       optimizers=optimizers, axes=[
                                           axes[0, 0], axes[1, 0]])
        _ = self.plot_performance_profile(
            strategies=[strategies], strategy_names=strategy_names,
            axes=[axes[0, 1], axes[1, 1]], show_combinations=False)
        _ = data_test.plot_performance_labels(
            rel_tolerance=self.rel_tolerance,
            axes=[axes[2, 0], axes[3, 0]],
            y_pred=y_pred, set_yticklabels=False)
        _ = data_test.plot_performance_labels(
            rel_tolerance=self.rel_tolerance,
            axes=[axes[2, 1], axes[3, 1]],
            y_pred=None, set_yticklabels=False)

        return fig
