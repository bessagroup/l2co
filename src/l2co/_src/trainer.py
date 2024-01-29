"""
Module containing a trainer class to evaluate the performance of a classifier
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from abc import abstractmethod
# from functools import cached_property
from typing import Dict, Iterable, Optional, Tuple

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
from .classifier import Classifier
from .data_loader import DataLoader
from .lstm_autoencoder import LSTMMultiOutputClassifier, Stage1Model
from .perf_profile import create_performance_profile, plot_performance_profile

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


# make sure no plots are showing
plt.ioff()

# =============================================================================


class Trainer():
    """
    Class to train and evaluate the performance of a classifier

    Parameters
    ----------
    data_train : DataLoader
        Data loader containing the training data
    data_test : DataLoader
        Data loader containing the test data
    clf : Classifier
        Classifier to be evaluated
    config : DictConfig
        Configuration object
    """

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

    @abstractmethod
    def train(self):
        """
        Train the classifier
        """
        pass

    @abstractmethod
    def predict(self) -> xr.DataArray:
        """
        Predict the labels for the test data
        """
        pass

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

        self.performance_profile = create_performance_profile(
            ds_raw=data_test,
            iterations=self.data_test.n_iterations,
            strategies=strategies)

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
            strategy_names=strategy_names, axes=axes,
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
                y_true=y_true, y_pred=y_pred, optimizer=optimizers)

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
        # num_labels = len(df.columns) - 5  # Excluding the feature columns

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

        axes[0].set_title(self.clf.name)

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
        # if self.y_pred is None:
        #     self.predict()

        # y = self.data_test.y.sortby('problem')

        # CLASSIFIER METRICS
        (self.precision, self.recall,
         self.f1, self.support) = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')
        self.accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        # create confusion matrix
        self.confusion_matrix = confusion_matrix(
            y_true=y_true, y_pred=y_pred, optimizers=optimizers)

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
            strategies=strategies, strategy_names=strategy_names,
            axes=[axes[0, 1], axes[1, 1]], show_combinations=False)
        _ = data_test.plot_performance_labels(
            rel_tolerance=self.rel_tolerance,
            axes=[axes[2, 0], axes[3, 0]],
            y_pred=self.y_pred, set_yticklabels=False)
        _ = data_test.plot_performance_labels(
            rel_tolerance=self.rel_tolerance,
            axes=[axes[2, 1], axes[3, 1]],
            y_pred=None, set_yticklabels=False)

        return fig


class TrainerStage0(Trainer):
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
        self.clf.fit(self.data_train.X.to_pandas(),
                     self.data_train.y_binary(self.rel_tolerance).to_pandas())

    def predict(self) -> xr.DataArray:
        y_pred_proba = xr.DataArray(
            np.array(self.clf.predict_proba(self.data_test.X.to_pandas()))[
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
            # p.unstack(),
            p,
            input_core_dims=[['optimizer']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[object],
        )

        self.y_pred = result.rename(self.name)

# =============================================================================


class TrainerStage1(Trainer):
    def __init__(self, data_train: DataLoader,
                 data_test: DataLoader,
                 stage0_clf: Classifier, config: DictConfig,
                 iterations: int,
                 t_switch: int,
                 t_start: int,
                 rel_tolerance: float = 0.0,
                 name: str = 'trainer'):

        self.data_train = data_train
        self.data_test = data_test
        self.stage0_clf = stage0_clf
        self.rel_tolerance = rel_tolerance
        self.iterations = iterations
        self.t_switch = t_switch
        self.t_start = t_start
        self.name = name

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

        self.clf = self.stage0_clf

    def train_stage0(self):
        _d_train = self.data_train.select_single_optimizers().query_t(
            self.iterations)

        self.stage0_clf.fit(_d_train.X.to_pandas(),
                            _d_train.y_binary(self.rel_tolerance).to_pandas())

    def train_stage1_optimizer(
            self, optimizer: str,
            epochs: int = 3, batch_size: int = 32) -> Stage1Model:

        _d_train = self.data_train.query_optimizer(optimizer).query_t(
            self.iterations)

        _d_train, _d_val = _d_train.train_test_split(test_size=0.2)

        X_train = np.atleast_3d(_d_train.X_trajectory2(
            self.t_switch, t_start=self.t_start))
        X_val = np.atleast_3d(_d_val.X_trajectory2(
            self.t_switch, t_start=self.t_start))

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))

        y_train = np.atleast_2d(_d_train.y_binary(
            self.rel_tolerance).astype(int))
        y_val = np.atleast_2d(_d_val.y_binary(
            self.rel_tolerance).astype(int))

        model = LSTMMultiOutputClassifier(
            optimizers=_d_train.optimizers,
            length_sequence=self.t_switch-self.t_start)

        model.train(X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=batch_size)

        return model

    def train_stage1(self) -> None:
        self.stage1_clf: Dict[str, Stage1Model] = {
            optimizer: self.train_stage1_optimizer(optimizer)
            for optimizer in
            self.data_train.select_single_optimizers().optimizers.values
        }

    def predict_stage1(self) -> xr.DataArray:
        if self.y_pred_stage0 is None:
            self.predict_stage0()

        if self.stage1_clf is None:
            self.train_stage1()

        list_y_pred_stage1 = []

        for optimizer in self.data_test.select_single_optimizers().optimizers.values:
            problems = self.y_pred_stage0.where(
                self.y_pred_stage0 == optimizer, drop=True).problem
            _d_test = self.data_test.query_optimizer(
                optimizer).query_t(self.iterations).select_problems(problems)

            X_test = np.atleast_3d(_d_test.X_trajectory2(
                self.t_switch, t_start=self.t_start))
            X_test = np.transpose(X_test, (1, 0, 2))

            list_y_pred_stage1.append(xr.DataArray(
                self.stage1_clf[optimizer].predict(X_test),
                dims=['problem', 'optimizer'],
                coords={'optimizer': _d_test.optimizers,
                        'problem': _d_test.problem})
            )

        y_pred_stage1 = xr.concat(
            list_y_pred_stage1, dim='problem').sortby('problem')

        y_pred_stage1 = y_pred_stage1.idxmax(
            dim='optimizer').sortby('problem')
        self.y_pred_stage0 = self.y_pred_stage0.sortby('problem')

        df = pd.DataFrame({'stage0': self.y_pred_stage0.to_pandas(),
                          'stage1': y_pred_stage1.to_pandas()})

        # Custom function to apply to each row
        def custom_function(row):
            if row['stage0'] == row['stage1']:
                return row['stage0']  # Return the first column name
            else:
                # Concatenate values with a + sign
                return f"{row['stage0']}+{row['stage1']}"

        # Apply the custom function to each row
        result_series = df.apply(custom_function, axis=1)
        self.y_pred_stage1 = xr.DataArray(
            result_series).rename(f"{self.name}_stage1")

    def predict_stage0(self) -> None:
        _d_test = self.data_test.select_single_optimizers().query_t(
            self.iterations)

        y_pred_proba = xr.DataArray(
            np.array(self.stage0_clf.predict_proba(_d_test.X.to_pandas()))[
                :, :, 1].T,
            dims=_d_test.y_binary(self.rel_tolerance).dims,
            coords=_d_test.y_binary(self.rel_tolerance).coords)

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

        self.y_pred_stage0 = result.rename(f"{self.name}_stage0")

    def plot_stage0(self):
        return self.plot(
            y_true=self.data_test.select_single_optimizers().query_t(
                self.iterations).y,
            y_pred=self.y_pred_stage0,
            strategies=self.y_pred_stage0,
            optimizers=self.data_test.select_single_optimizers().optimizers,
            data_test=self.data_test.select_single_optimizers().query_t(
                self.iterations),
            strategy_names=[f"{self.name}_stage0"])

    def plot_stage1(self):
        return self.plot(
            y_true=self.data_test.query_t(
                self.iterations).y,
            y_pred=self.y_pred_stage1,
            optimizers=self.data_test.optimizers,
            data_test=self.data_test.query_t(self.iterations),
            strategies=self.y_pred_stage1,
            strategy_names=[f"{self.name}_stage1"])

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
        trainer = TrainerStage0(
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

        fig, axes = plt.subplots(4, 2, figsize=(15, 15), gridspec_kw={
            'height_ratios': [1, 0.25, 1, 0.25]})
        _ = trainer.plot_confusion_matrix(axes=axes[0])
        _ = trainer.plot_performance_profile(
            axes=axes[1], show_combinations=False)
        _ = trainer.data_test.plot_performance_labels(
            rel_tolerance=self.config.rel_tolerance,
            axes=axes[2],
            y_pred=trainer.y_pred, set_yticklabels=False)
        _ = trainer.data_test.plot_performance_labels(
            rel_tolerance=self.config.rel_tolerance,
            axes=axes[3], set_yticklabels=False)

        self.experiment_sample.store(object=fig, name="fig", to_disk=True)
