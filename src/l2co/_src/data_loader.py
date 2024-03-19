"""
Module containing a data loader class for stage0 of the L2CO framework.

Classes
-------

DataLoader
    A data loader class for optimization dynamics data
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Tuple

# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Local
from ._logging import logger as l2co_logger
from .content_based import get_labels
from .perf_profile import create_performance_profile, load_all_data
from .performance import (INVALID_VALUE, get_performance,
                          get_performance_preprocessed, query_t)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

# create a logger
logger = l2co_logger.getChild(__name__)

# =============================================================================


class DataLoader():
    def __init__(self, X: xr.DataArray,
                 n_iterations: int,
                 n_realizations: int,
                 performance: xr.Dataset,
                 data: xr.Dataset,
                 config: DictConfig,
                 experimentdata: ExperimentData,
                 data_samples: xr.Dataset):
        """
        A data loader class for stage0 of the L2CO framework.

        Parameters
        ----------
        X : xarray.DataArray
            The features of the optimization problems.
        n_iterations : int
            The number of training steps of each roll-out.
        n_realizations : int
            The number of realizations for each problem.
        performance : xarray.Dataset
            The performance dataset.
        data : xarray.Dataset
            The raw optimization trajectories.
        config : DictConfig
            The configuration file in hydra format.
        experimentdata : ExperimentData
            The experiment data.
        """
        self.X = X  # X.sortby('problem')
        self.n_iterations = n_iterations
        self.n_realizations = n_realizations
        self.performance = performance  # performance.sortby('problem')
        self.data = data  # data.sortby('problem')
        self.config = config
        self.experimentdata = experimentdata
        self.data_samples = data_samples

    # NOT WORKING!
    @classmethod
    def from_experimentdata(cls, config: DictConfig,
                            experimentdata: ExperimentData):
        """
        Create a data loader from a f3dasm ExperimentData object.

        Parameters
        ----------
        config : DictConfig
            The configuration file in hydra format.
        experimentdata : ExperimentData
            The experiment data object.

        Returns
        -------
        DataLoader
            The resulting data loader.
        """

        _, _df_output = experimentdata.to_pandas()
        # remove all values that say 'ERROR' and 'invalid'
        _df_output = _df_output.replace('ERROR', np.nan)
        _df_output = _df_output.replace('invalid', np.nan)
        _df_output = _df_output.replace('not working', np.nan)

        # remove all experiments where none of the optimizers have finished
        _valid_indices = _df_output.drop(
            columns=['random_samples']).dropna(how='all').index

        experimentdata = experimentdata.select(_valid_indices)

        # get column names where all rows are nan
        _columns_to_drop = _df_output.columns[_df_output.isna().all()]
        experimentdata = experimentdata.drop_output(_columns_to_drop)

        _data = load_all_data(experimentdata)

        # Get the number of realizations
        n_realizations = _data.realization.size
        n_iterations = _data.iterations.size - 1

        # Features
        X = get_labels(experimentdata=experimentdata,
                       n_realizations=n_realizations)

        # Performance
        performance = get_performance(experimentdata)

        # Performance: Expand and stack dimension realization and itemID
        performance_stacked = performance.expand_dims(
            {'realization': np.arange(_data.realization.size)}).stack(
                problem=('realization', 'itemID'))

        # Fill in unvalid runs with 1000.0
        _, df_output = experimentdata.to_pandas()
        df_output = df_output.drop(
            columns=['random_samples']).sort_index(axis=1)
        da = xr.DataArray(
            df_output.T, dims=['optimizer', 'itemID']).expand_dims(
            {'realization': np.arange(2),
             'iterations': np.arange(_data.iterations.size),
             })
        da_mask = xr.where(da == 'invalid', True, False)
        da_mask = da_mask.transpose(
            'optimizer', 'iterations', 'realization', 'itemID')

        performance_stacked = performance_stacked.where(
            ~da_mask,
            other=INVALID_VALUE)
        performance_stacked = performance_stacked.stack(
            problem=('realization', 'itemID'))

        # Stack dimensions realization and itemID
        data_stacked = _data.stack(problem=('realization', 'itemID'))

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   performance=performance_stacked,
                   data=data_stacked,
                   config=config,
                   experimentdata=experimentdata)

    @classmethod
    def from_preprocessed_files(cls, config: DictConfig,
                                experimentdata: ExperimentData,
                                data: xr.Dataset, data_samples: xr.Dataset):
        """
        Create a data loader from preprocessed files of the optimization
        trajectories (data) and the roll-outs of random search (data_samples).

        Parameters
        ----------
        config : DictConfig
            The configuration file in hydra format.
        experimentdata : ExperimentData
            The experiment data object.
        data : xarray.Dataset
            The raw optimization trajectories.
        data_samples : xarray.Dataset
            The raw optimization trajectories.

        Returns
        -------
        DataLoader
            The resulting data loader.
        """

        n_realizations = data.realization.size
        n_iterations = data.iterations.size

        _, _df_output = experimentdata.to_pandas()
        # remove all values that say 'ERROR' and 'invalid'
        _df_output = _df_output.replace('ERROR', np.nan)
        _df_output = _df_output.replace('invalid', np.nan)
        _df_output = _df_output.replace('not working', np.nan)

        # remove all experiments where none of the optimizers have finished
        _valid_indices = _df_output.drop(
            columns=['random_samples']).dropna(how='all').index

        experimentdata = experimentdata.select(_valid_indices)

        # get column names where all rows are nan
        _columns_to_drop = _df_output.columns[_df_output.isna().all()]
        experimentdata = experimentdata.drop_output(_columns_to_drop)

        data = data.sel(itemID=experimentdata.index)
        data_samples = data_samples.sel(itemID=experimentdata.index)

        # Features
        X = get_labels(experimentdata=experimentdata,
                       n_realizations=n_realizations)

        # Performance
        performance = get_performance_preprocessed(
            data, data_samples)

        # Performance: Expand and stack dimension realization and itemID
        performance_stacked = performance.expand_dims(
            {'realization': np.arange(data.realization.size)})

        # Fill in unvalid runs with 1000.0
        _, df_output = experimentdata.to_pandas()
        df_output = df_output.drop(
            columns=['random_samples']).sort_index(axis=1)
        da = xr.DataArray(
            df_output.T, dims=['optimizer', 'itemID']).expand_dims(
            {'realization': np.arange(n_realizations),
             'iterations': np.arange(data.iterations.size),
             })
        da_mask = xr.where(da == 'invalid', True, False)
        da_mask = da_mask.transpose(
            'optimizer', 'iterations', 'realization', 'itemID')

        performance_stacked = performance_stacked.where(
            ~da_mask,
            other=INVALID_VALUE)
        performance_stacked = performance_stacked.stack(
            problem=('realization', 'itemID'))

        # Stack dimensions realization and itemID
        data_stacked = data.stack(problem=('realization', 'itemID'))
        data_samples_stacked = data_samples.stack(
            problem=('realization', 'itemID'))
        # data_samples_stacked = data_samples_stacked.sel(
        #     itemID=np.unique(data_stacked.itemID.values))

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   data=data_stacked,
                   performance=performance_stacked,
                   config=config,
                   experimentdata=experimentdata,
                   data_samples=data_samples_stacked)

    def filter_nan_or_inf(self) -> DataLoader:
        """
        Filter out all problems with NaN or Inf values in the optimization
        raw data.

        Returns
        -------
        DataLoader
            The filtered data loader.
        """
        mask = self.data.where(np.logical_or(
            np.isnan(self.data), np.isnan(self.data)), drop=True)

        faulty_problems = mask.problem

        correct_problems = self.data.problem.where(
            ~self.data.problem.isin(faulty_problems), drop=True)

        return self.select_problems(correct_problems)

    def has_nan_or_inf(self) -> bool:
        return np.logical_or.reduce(
            [np.isnan(self.data.output), np.isnan(self.data.output)]).any()

    # @cached_property
    @property
    def y(self) -> xr.DataArray:
        """
        Returns the minimum performance index along the 'optimizer' dimension,
        expanded with a new dimension 'realization' and stacked with 'itemID'
        and 'realization' dimensions.

        Returns
        -------
        xr.DataArray
            The resulting DataArray with dimensions 'problem',
            'realization', and 'itemID'.
        """
        logger.debug("Creating labels")

        return self.performance.sel(problem=self.problem, drop=True).idxmin(
            dim='optimizer')

    @property
    def features(self) -> xr.DataArray:
        """
        Property method to return the features coordinate

        Returns
        -------
        xarray.DataArray
            The features coordinate.
        """
        return self.X.features

    @property
    def problem(self) -> xr.DataArray:
        """
        Property method to return the problem coordinate

        Returns
        -------
        xarray.DataArray
            The problem coordinate.
        """
        return self.X.problem

    @property
    def optimizers(self) -> xr.DataArray:
        """
        Property method to return the optimizers coordinate

        Returns
        -------
        xarray.DataArray
            The optimizers coordinate.
        """
        return self.performance.optimizer

    @lru_cache(maxsize=1)
    def y_binary(self, rel_tolerance: float = 0.1) -> xr.DataArray:
        if rel_tolerance == 0.0:
            return (self.y == self.performance.optimizer).T

        else:
            threshold = self.performance.min(
                dim='optimizer') + abs(
                    self.performance.min(dim='optimizer') * rel_tolerance)

        return (self.performance <= threshold).sel(
            problem=self.problem, drop=True).T

    @lru_cache(maxsize=1)
    def X_trajectory(self, iterations: int, t_start: int = 0) -> xr.DataArray:
        """
        Select the history of an optimization trajectory.

        Parameters
        ----------
        iterations : int
            The number of iterations to select.
        t_start : int, optional
            The starting iteration number of the requested trajectory,
            by default 0

        Returns
        -------
        xr.DataArray
            The selected trajectory.
        """
        _X_trajectory = self.data.isel(
            output_dim=0, iterations=range(t_start, iterations), drop=True
        ).output.rename('trajectory')

        return xr.apply_ufunc(
            np.log, (_X_trajectory - _X_trajectory.min('iterations')
                     ) + 1.0, dask='parallelized')

    def performance_profile(
            self, strategies: Optional[Iterable[xr.DataArray]] = None
    ) -> xr.Dataset:

        if strategies is None:
            strategies = []

        # self.data = self.data.sortby('problem')
        best_perf_profile = self.query_t(
            self.n_iterations-1).y.rename('best_perf_profile')
        worst_perf_profile = self.query_t(
            self.n_iterations-1).performance.idxmax(
                dim='optimizer').rename('worst_perf_profile')

        _strategies = [best_perf_profile, worst_perf_profile, *strategies]

        performance_profile = create_performance_profile(
            ds_raw=self.data,
            iterations=self.n_iterations,
            strategies=_strategies)

        # performance_profile.load()
        return performance_profile

    def select_single_optimizers(self) -> DataLoader:
        """
        Select only the single optimizers.

        Returns
        -------
        DataLoader
            The data loader with only the single optimizers.
        """
        single_optimizers = [o.values for o in self.performance.optimizer
                             if '+' not in str(o.values)]

        performance = self.performance.sel(
            optimizer=single_optimizers)

        data = self.data.sel(optimizer=single_optimizers)

        return DataLoader(
            X=self.X,
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            performance=performance,
            data=data,
            config=self.config,
            experimentdata=self.experimentdata,
            data_samples=self.data_samples
        )

    def select_problems(self, problem_indices: xr.DataArray) -> DataLoader:
        """
        Select a subset of problems of the data loader.

        Parameters
        ----------
        problem_indices : xr.DataArray
            The problem indices to select.

        Returns
        -------
        DataLoader
            The selected data loader.

        Notes
        -----
        Selects on the problem dimension. Also sorts the problem indices on
        itemID.
        """
        _problem_indices = problem_indices.sortby('itemID')
        _unique_itemID = np.unique(problem_indices.itemID.values)
        return DataLoader(
            X=self.X.sel(problem=_problem_indices),
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            performance=self.performance.sel(
                problem=_problem_indices),
            data=self.data.sel(problem=_problem_indices),
            config=self.config,
            experimentdata=self.experimentdata.select(_unique_itemID),
            data_samples=self.data_samples.sel(problem=_problem_indices)
        )

    def query_t(self, t: int) -> DataLoader:
        """
        Query the data loader at a specific iteration.

        Parameters
        ----------
        t : int
            The iteration to query.

        Returns
        -------
        DataLoader
            The queried data loader.
        """
        performance_t = query_t(self.performance, t=t)
        data = self.data.sel(iterations=range(t))
        return DataLoader(
            X=self.X,
            n_iterations=t,
            n_realizations=self.n_realizations,
            performance=performance_t,
            data=data,
            config=self.config,
            experimentdata=self.experimentdata,
            data_samples=self.data_samples
        )

    def query_optimizer(self, optimizer: str) -> DataLoader:
        """
        Query the data loader for a specific optimizer.

        Parameters
        ----------
        optimizer : str
            The optimizer to query.

        Returns
        -------
        DataLoader
            The queried data loader.
        """
        filtered_optimizer = np.array(
            [o for o in self.data.optimizer.values if o.startswith(optimizer)],
            dtype=object
        )

        data_selected = self.data.sel(optimizer=optimizer)

        n_iterations = self.n_iterations
        n_realizations = self.n_realizations

        # Config
        config = self.config

        # Performance
        performance = self.performance.sel(
            optimizer=filtered_optimizer).assign_coords(
            optimizer=np.array(
                [o.split('+')[-1] if '+' in o else o
                 for o in filtered_optimizer], dtype=object))

        return DataLoader(
            X=self.X,
            n_iterations=n_iterations,
            n_realizations=n_realizations,
            performance=performance,
            data=data_selected,
            config=config,
            experimentdata=self.experimentdata,
            data_samples=self.data_samples
        )

    def train_test_split(
            self,
            test_size: float = 0.2, random_state: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Split the data into train and test sets.

        Parameters
        ----------
        test_size : float, optional
            The size of the test set, by default 0.2
        random_state : int, optional
            The random state, by default None

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The train and test data loaders.
        """
        _X_train_indices, _X_test_indices, = train_test_split(
            self.X.problem, test_size=test_size,
            random_state=random_state)

        return (self.select_problems(_X_train_indices),
                self.select_problems(_X_test_indices))

    def plot_performance_labels(
            self,
            y_pred: Optional[xr.DataArray] = None,
            rel_tolerance: float = 0.0,
            return_df: bool = False,
            axes: Optional[Iterable[plt.Axes, plt.Axes]] = None,
            set_yticklabels: bool = True
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Plot the performance labels.

        Parameters
        ----------
        y_pred : Optional[xr.DataArray], optional
            The predicted labels, by default None
        rel_tolerance : float, optional
            The relative tolerance, by default 0.0
        return_df : bool, optional
            Whether to return the dataframe, by default False
        axes : Optional[Tuple[plt.Axes, plt.Axes]], optional
            The axes to plot on, by default None
        set_yticklabels : bool, optional
            Whether to set the yticklabels, by default True

        Returns
        -------
        Tuple[plt.Axes, plt.Axes]
            The axes.
        """

        if y_pred is None:
            _y = self.y_binary(rel_tolerance=rel_tolerance)

        else:
            _y = (y_pred == self.performance.optimizer).T

        y_pandas = _y.to_pandas()
        x_pandas = self.X.to_pandas()

        y_pandas.index = y_pandas.index.set_names(['realization', 'itemID'])
        x_pandas.index = x_pandas.index.set_names(['realization', 'itemID'])

        grouped_Xy = pd.concat([x_pandas, y_pandas], axis=1)

        df = grouped_Xy.groupby(
            list(self.features.values)).mean().reset_index()

        num_rows = len(df)
        n_features = len(self.features)

        col_optimizers = list(self.performance.optimizer.values)
        df[col_optimizers] = df[col_optimizers].div(
            df[col_optimizers].sum(axis=1), axis=0)

        if axes is None:
            fig, axes = plt.subplots(
                2, 1, gridspec_kw={'height_ratios': [
                    4, 1]}, figsize=(10, 0.5 * num_rows))

        colors = ['skyblue', 'salmon', 'lightgreen',
                  'gold', 'red']  # Define colors for each label

        yticklabels = []

        legend_patches = [mpatches.Patch(
            color=colors[i], label=label) for i, label in enumerate(
                df.columns[n_features:])]

        for i, (_, row) in enumerate(df.iterrows()):
            labels = row.index[n_features:]  # Excluding the feature columns
            freq = row.values[n_features:]  # Excluding the feature values
            if set_yticklabels:
                yticklabels.append(
                    ", ".join([f"{feature}: {value}" for
                               feature, value in row[:n_features].items()]))
            left = 0
            for j, (label, value) in enumerate(zip(labels, freq)):
                axes[0].barh(i, width=value, height=0.3, left=left,
                             color=colors[j], label=label)
                left += value

        axes[0].set_yticks(np.arange(num_rows))
        axes[0].set_yticklabels(yticklabels)
        axes[0].set_xlabel('Frequency')
        axes[1].legend(handles=legend_patches, loc='center',
                       ncol=len(legend_patches))

        axes[1].axis('off')
        text = f"Relative tolerance: {rel_tolerance}"
        if y_pred is None:
            text += "- Test data"

        else:
            text += "- Predicted data"
        axes[0].set_title(text)

        if return_df:
            return axes, df

        return axes

    def has_unknown_performances(self) -> bool:
        """
        Check if the performance dataset contains unknown values.

        Returns
        -------
        bool
            Whether the performance dataset contains unknown values.
        """
        return bool(self.performance.isnull().any())
