"""
Module containing a data loader class for stage0 of the L2CO framework.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from functools import cached_property, lru_cache
from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Third-party
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
                 config: DictConfig,):
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
        """
        self.X = X
        self.n_iterations = n_iterations
        self.n_realizations = n_realizations
        self.performance = performance
        self.data = data
        self.config = config

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
        return self.performance.idxmin(dim='optimizer').expand_dims(
            dim={'realization': np.arange(self.n_realizations)}
        ).stack(problem=['itemID', 'realization']).astype('U100')

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
        return self.X.unstack(
            'problem').stack(
                problem=['itemID', 'realization']).problem

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

        X_train = self.X.unstack('problem').sel(
            itemID=_X_train_indices).stack(problem=['itemID', 'realization'])
        X_test = self.X.unstack('problem').sel(
            itemID=_X_test_indices).stack(problem=['itemID', 'realization'])

        _data_train = self.data.sel(itemID=_X_train_indices)
        _data_test = self.data.sel(itemID=_X_test_indices)

        _performance_train = self.performance.sel(itemID=_X_train_indices)
        _performance_test = self.performance.sel(itemID=_X_test_indices)

        return (
            DataLoader(
                X=X_train,
                n_iterations=self.n_iterations,
                n_realizations=self.n_realizations,
                performance=_performance_train,
                data=_data_train,
                config=self.config),
            DataLoader(
                X=X_test,
                n_iterations=self.n_iterations,
                n_realizations=self.n_realizations,
                performance=_performance_test,
                data=_data_test,
                config=self.config)
        )

    @classmethod
    def from_experimentdata(cls, config: DictConfig,
                            experimentdata: ExperimentData):

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

        performance = select_optimizers_without_switch(performance)

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   performance=performance,
                   data=_data,
                   config=config)

    @classmethod
    def from_preprocessed_files(cls, config: DictConfig,
                                experimentdata: ExperimentData,
                                data: xr.Dataset, data_samples: xr.Dataset):
        n_realizations = len(data.realization)
        n_iterations = len(data.iterations)-2  # HAS TO BE -1

        # Features
        X = get_labels(experimentdata=experimentdata,
                       n_realizations=n_realizations)

        # Performance
        performance = query_t(get_performance_preprocessed(
            data, data_samples), t=n_iterations)
        performance = select_optimizers_without_switch(performance)

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   data=data,
                   performance=performance,
                   config=config)

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
        ).stack(problem=['itemID', 'realization']).T

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
