"""
Module containing a data loader class for stage0 of the L2CO framework.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from functools import cached_property, lru_cache
from typing import Optional, Tuple

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
from .perf_profile import load_all_data
from .performance import (get_performance, get_performance_preprocessed,
                          query_t, select_optimizers_without_switch)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

# create a logger
logger = l2co_logger.getChild(__name__)

# make sure no plots are showing
plt.ioff()

# =============================================================================


class DataLoader():
    def __init__(self, X: xr.DataArray,
                 n_iterations: int,
                 n_realizations: int,
                 performance: xr.Dataset,
                 data: xr.Dataset,
                 config: DictConfig,
                 experimentdata: ExperimentData):
        """
        A data loader class for stage0 of the L2CO framework.

        Parameters
        ----------
        X : xarray.DataArray
            The features.
        n_iterations : int
            The number of iterations.
        n_realizations : int
            The number of realizations.
        performance : xarray.Dataset
            The performance dataset.
        data : xarray.Dataset
            The data.
        config : DictConfig
            The configuration file in hydra format.
        experimentdata : ExperimentData
            The experiment data.
        """
        self.X = X
        self.n_iterations = n_iterations
        self.n_realizations = n_realizations
        self.performance = performance
        self.data = data
        self.config = config
        self.experimentdata = experimentdata

    @classmethod
    def from_experimentdata(cls, config: DictConfig,
                            experimentdata: ExperimentData,
                            select_single_optimizers: bool = True):

        _data = load_all_data(experimentdata)

        # Get the number of realizations
        n_realizations = _data.realization.size
        n_iterations = _data.iterations.size - 1

        # Features
        X = get_labels(experimentdata=experimentdata,
                       n_realizations=n_realizations)

        # Performance
        performance = query_t(get_performance(
            experimentdata), t=n_iterations)

        if select_single_optimizers:
            performance = select_optimizers_without_switch(performance)

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   performance=performance,
                   data=_data,
                   config=config,
                   experimentdata=experimentdata)

    @classmethod
    def from_preprocessed_files(cls, config: DictConfig,
                                experimentdata: ExperimentData,
                                data: xr.Dataset, data_samples: xr.Dataset,
                                select_single_optimizers: bool = True):
        n_realizations = data.realization.size
        n_iterations = data.iterations.size  # HAS TO BE -1

        # Features
        X = get_labels(experimentdata=experimentdata,
                       n_realizations=n_realizations)

        # Performance
        performance = get_performance_preprocessed(
            data, data_samples)

        if select_single_optimizers:
            performance = select_optimizers_without_switch(performance)

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   data=data,
                   performance=performance,
                   config=config,
                   experimentdata=experimentdata)

    @cached_property
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
        # return self.performance.idxmin(dim='optimizer').expand_dims(
        #     dim={'realization': np.arange(self.n_realizations)}
        # ).stack(problem=['itemID', 'realization']).astype('U100')

        return self.performance.expand_dims(
            dim={'realization': np.arange(self.n_realizations)}
        ).stack(problem=['itemID', 'realization']
                ).sel(problem=self.problem, drop=True).idxmin(dim='optimizer')

    @cached_property
    def y_reg(self) -> xr.DataArray:
        """
        Returns the performance index along the 'optimizer' dimension,
        expanded with a new dimension 'realization' and stacked with 'itemID'
        and 'realization' dimensions. Used for regression models

        Returns
        -------
        xr.DataArray
            The resulting DataArray with dimensions 'problem',
            'realization', and 'itemID'.
        """
        logger.debug("Creating labels")
        return self.performance.expand_dims(
            dim={'realization': np.arange(self.n_realizations)}
        ).stack(problem=['itemID', 'realization']).T

    @property
    def features(self) -> xr.DataArray:
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
        # return self.X.unstack(
        #     'problem').stack(
        #         problem=['itemID', 'realization']).problem

    @property
    def optimizers(self) -> xr.DataArray:
        """
        Property method to return the optimizers coordinate

        Returns
        -------
        xarray.DataArray
            The optimizers coordinate.
        """
        return xr.DataArray(np.unique(self.y),
                            dims='optimizer',
                            coords={'optimizer': np.unique(self.y)},
                            name='optimizer')

    def select_single_optimizers(self) -> DataLoader:
        """
        Select only the single optimizers.

        Returns
        -------
        DataLoader
            The data loader with only the single optimizers.
        """
        # TODO: select appropriate data with only single optimizers!

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
            experimentdata=self.experimentdata
        )

    def select(self, itemID_indices: np.ndarray) -> DataLoader:
        """
        Select a subset of itemIDs of the data loader.

        Parameters
        ----------
        indices : np.ndarray
            The indices to select.

        Returns
        -------
        DataLoader
            The selected data loader.

        Notes
        -----
        Selects on the itemID dimension.
        """
        return DataLoader(
            X=self.X.unstack('problem').sel(
                itemID=itemID_indices).stack(
                    problem=['itemID', 'realization']),
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            performance=self.performance.sel(itemID=itemID_indices),
            data=self.data.sel(itemID=itemID_indices),
            config=self.config,
            experimentdata=self.experimentdata.select(itemID_indices))

    def select_problems(self, problem_indices: xr.DataArray) -> DataLoader:
        _problem_indices = problem_indices.sortby('itemID')
        _unique_itemID = np.unique(problem_indices.itemID.values)
        return DataLoader(
            X=self.X.sel(problem=_problem_indices),
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            performance=self.performance.sel(
                itemID=_unique_itemID),
            data=self.data.sel(itemID=_unique_itemID),
            config=self.config,
            experimentdata=self.experimentdata.select(_unique_itemID)
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
            experimentdata=self.experimentdata
        )

    def query_optimizer(self, optimizer: str) -> DataLoader:
        filtered_optimizer = np.array(
            [o for o in self.data.optimizer.values if o.startswith(optimizer)],
            dtype=object
        )

        _, df_output = self.experimentdata.to_pandas()
        df_selected = df_output[filtered_optimizer]

        selection: pd.Series = df_selected.apply(
            lambda row: next((col for col, val in row.items()
                              if val is not None), None), axis=1)

        # remove all rows that are None
        selection = selection[selection.notnull()]

        # ExperimentData
        experimentdata = self.experimentdata.select(selection.index)

        # Data
        data_selected = xr.concat(
            [self.data.sel(itemID=i, optimizer=p)
             for i, p in selection.items()],
            dim='itemID').drop('optimizer').expand_dims(
                {'optimizer': [optimizer]})

        # Features
        X = self.X.unstack('problem').sel(
            itemID=selection.index).stack(problem=['itemID', 'realization'])

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
            X=X,
            n_iterations=n_iterations,
            n_realizations=n_realizations,
            performance=performance,
            data=data_selected,
            config=config,
            experimentdata=experimentdata
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
            self.X.unstack('problem').itemID, test_size=test_size,
            random_state=random_state)

        return (self.select(_X_train_indices),
                self.select(_X_test_indices))

    @lru_cache(maxsize=1)
    def y_binary(self, rel_tolerance: float = 0.1) -> xr.DataArray:
        if rel_tolerance == 0.0:
            return (self.y == self.performance.optimizer).T

        else:
            threshold = self.performance.min(
                dim='optimizer') + abs(
                    self.performance.min(dim='optimizer') * rel_tolerance)

        return (self.performance <= threshold).expand_dims(
            dim={'realization': np.arange(self.n_realizations)}
        ).stack(problem=['itemID', 'realization']
                ).sel(problem=self.problem, drop=True).T

    def plot_performance_labels(self,
                                y_pred: Optional[xr.DataArray] = None,
                                rel_tolerance: float = 0.1,
                                return_df: bool = False,
                                ax=None,
                                ax2=None,
                                set_yticklabels: bool = True):

        _X = self.X.unstack().stack(problem=('itemID', 'realization'))

        if y_pred is None:
            _y = self.y_binary(rel_tolerance=rel_tolerance)

        else:
            _y = (y_pred == self.performance.optimizer).T

        grouped_Xy = pd.concat([_X.to_pandas(
        ).T, _y.to_pandas()], axis=1)

        df = grouped_Xy.groupby(
            list(self.features.values)).mean().reset_index()

        num_rows = len(df)
        n_features = len(self.features)

        col_optimizers = list(self.performance.optimizer.values)
        df[col_optimizers] = df[col_optimizers].div(
            df[col_optimizers].sum(axis=1), axis=0)

        if ax is None:
            fig, (ax, ax2) = plt.subplots(
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
                ax.barh(i, width=value, height=0.3, left=left,
                        color=colors[j], label=label)
                left += value

        ax.set_yticks(np.arange(num_rows))
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('Frequency')
        ax2.legend(handles=legend_patches, loc='center',
                   ncol=len(legend_patches))

        ax2.axis('off')
        text = f"Relative tolerance: {rel_tolerance}"
        if y_pred is None:
            text += "- Test data"

        else:
            text += "- Predicted data"
        ax.set_title(text)

        if return_df:
            return ax, df

        return ax
