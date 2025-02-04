"""
Module containing a data loader class of the L2CO framework.

Classes
-------

DataLoader
    A data loader class for optimization dynamics data. Used for analysing the
    performances of various optimizers as well as creating learning data
OptimizationDynamics
    Learning dataset inherited from pytorch Dataset. Used for learning the
    machine learning models
create_entire_dataset
    Create an entire dataset from a given experiment data object
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split

# Local
from .performance import create_performance_profile, scaling2

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


#                                                                     Constants
# =============================================================================

INVALID_PERFORMANCE_VALUE = 1.0

#                                                                   Data Loader
# =============================================================================


class DataLoader:
    def __init__(
        self,
        data: xr.Dataset,
        task_labels: xr.DataArray,
    ):
        """
        A data loader class for stage0 of the L2CO framework.

        Parameters
        ----------
        data : xarray.Dataset
            The raw optimization trajectories.
        task_labels: pd.DataFrame
            The labels of the optimization problems.
        """
        self.data = data
        self.task_labels = task_labels

#                                                      Alternative Constructors
# =============================================================================

    @classmethod
    def from_file(
        cls, path: str, labels_filename: str = 'labels',
            data_filename: str = 'entire_dataset'):
        """
        Create a data loader from saved files.

        Parameters
        ----------
        path : str
            The path to the saved files.

        Returns
        -------
        DataLoader
            The resulting data loader.

        Notes
        -----
        The saved files should be in the path provided, in the NetCDF format
        and named 'entire_dataset.nc' for the optimization trajectories and
        'entire_dataset_samples.nc' for the roll-outs of random search.
        """
        # Create path object
        _path = Path(path)

        # Retrieve the optimization dynamics data
        data = xr.open_dataset(
            (_path / data_filename).with_suffix('.nc'),
            chunks='auto')

        df_task_labels = pd.read_csv(
            (_path / labels_filename).with_suffix('.csv'),
            index_col=0, header=0)

        task_labels = xr.DataArray(df_task_labels.values,
                                   coords={'itemID': df_task_labels.index,
                                           'features': df_task_labels.columns},
                                   dims=['itemID', 'features'],
                                   )

        # Expand itemID dim to match problem dim
        task_labels = task_labels.sel(itemID=data.itemID)

        return cls(data=data,
                   task_labels=task_labels,
                   )

#                                                         Storing and exporting
# =============================================================================

    def to_learning_data(
            self, optimizer: str,
            t: int, t_start: int,
            rel_tolerance: float = 0.0) -> OptimizationDynamics:
        """
        Convert the data loader to learning data.

        Parameters
        ----------
        optimizer : str
            The optimizer to select.
        t : int
            The iteration to select.
        t_start : int, optional
            The starting iteration number of the requested trajectory,
            by default 0
        rel_tolerance : float, optional
            The relative tolerance to use, by default 0.0

        Returns
        -------
        OptimizationDynamics
            The resulting optimization dynamics.

        Notes
        -----
        OptimizationDynamics is a pytorch Dataset object.
        """
        X_trajectory = self.select_optimizer(optimizer, squash=True).select_t(
            t=t, t_start=t_start).X_trajectory()
        Y = self.query_optimizer(optimizer).performance.isel(
            iterations=-1)

        return OptimizationDynamics(
            X_features=self.task_labels,
            X_trajectory=X_trajectory,
            Y=Y,
            rel_tolerance=rel_tolerance)

    def to_all_learning_data(
            self, t: int, t_start: int, rel_tolerance: float = 0.0
    ) -> Dict[str, OptimizationDynamics]:
        return {optimizer: self.to_learning_data(
            optimizer=optimizer, t=t, t_start=t_start,
            rel_tolerance=rel_tolerance) for
            optimizer in self.select_single_optimizers().optimizers.values}

    def to_feature_data(self, t: int, t_start, rel_tolerance: float = 0.0,
                        optimizer: Optional[str] = None,
                        ) -> FeaturesDataSet:
        """
        Convert the data loader to feature data.

        Parameters
        ----------
        rel_tolerance : float, optional
            The relative tolerance to use, by default 0.0
        optimizer : str, optional
            The optimizer to condition on, by default None

        Returns
        -------
        FeaturesDataSet
            The resulting feature data.

        Notes
        -----
        FeaturesDataSet is a simple class containing the features and labels of
        the optimization problems.
        """
        if optimizer is None:
            Y_features = self.select_single_optimizers(squash=True).select_t(
                t=t, t_start=t_start).performance.isel(
                iterations=-1, output_dim=0, drop=True).T
        else:
            Y_features = self.query_optimizer(optimizer).performance.isel(
                iterations=-1, output_dim=0, drop=True).T

        return FeaturesDataSet(X_features=self.task_labels,
                               Y_features=Y_features,
                               rel_tolerance=rel_tolerance)
#                                                                    Properties
# =============================================================================

    @ property
    def n_iterations(self) -> int:
        return self.data.iterations.size

    @ property
    def n_realization(self) -> int:
        return np.unique(self.data.realization).size

    @ property
    def features(self) -> xr.DataArray:
        """
        Property method to return the features coordinate

        Returns
        -------
        xarray.DataArray
            The features coordinate.
        """
        return self.task_labels.features

    @ property
    def problems(self) -> xr.DataArray:
        """
        Property method to return the problem coordinate

        Returns
        -------
        xarray.DataArray
            The problem coordinate.
        """
        return self.task_labels.problem

    @ property
    def optimizers(self) -> xr.DataArray:
        """
        Property method to return the optimizers coordinate

        Returns
        -------
        xarray.DataArray
            The optimizers coordinate.
        """
        return self.data.optimizer

    @ property
    def performance(self) -> xr.DataArray:
        t_start = perf_counter()
        print("Triggered performance")
        performance = scaling2(self.data)

        performance = performance.ffill(dim='iterations')

        # performance = performance.where(
        #     ~performance.isnull(), other=INVALID_PERFORMANCE_VALUE)
        print(f"Performance calculation took: {perf_counter()-t_start:0.2f}")
        return performance

    @ lru_cache(maxsize=1)
    def y_binary(self, rel_tolerance: float = 0.1) -> xr.DataArray:
        _performance_at_last_t = self.performance.isel(
            iterations=-1, output_dim=0, drop=True)

        if rel_tolerance == 0.0:
            return (_performance_at_last_t.idxmin(
                'optimizer') == self.optimizers).T

        threshold = _performance_at_last_t.min(
            dim='optimizer') + abs(
                _performance_at_last_t.min(dim='optimizer') * rel_tolerance)

        return (_performance_at_last_t <= threshold).T

    @ lru_cache(maxsize=1)
    def X_trajectory(self) -> xr.DataArray:
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
        xarray.DataArray
            The selected trajectory.
        """

        def scale(x: np.ndarray) -> np.ndarray:
            # 3. Make sure the quality indicator value is non-negative
            p = (x - x.min())
            # 4. Add epsilon to make quality indicator value non-zero
            eps = np.where(p > 0.0, p, p.max()+1e-16).min()
            x_log10 = np.log10(p + eps)

            # 5. Min-max scaling of the data
            x_min = x_log10.min()
            x_max = x_log10.max()
            return (x_log10 - x_min) / (x_max - x_min + 1e-16)

        trajectory: xr.Dataset = xr.apply_ufunc(
            scale,
            self.data,
            input_core_dims=[['iterations', 'output_dim']],
            output_core_dims=[['iterations', 'output_dim']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={'allow_rechunk': True}  # False
        )

        return trajectory.output.isel(
            output_dim=0, drop=True).rename(
                'trajectory').transpose(
                    'optimizer', 'iterations', 'problem')  # .fillna(
        # INVALID_PERFORMANCE_VALUE)

#                                                            Selecting problems
# =============================================================================

    def select_problems(self, problem_indices: xr.DataArray) -> DataLoader:
        """
        Select a subset of problems of the data loader.

        Parameters
        ----------
        problem_indices : xarray.DataArray
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

        return DataLoader(
            task_labels=self.task_labels.sel(problem=_problem_indices),
            data=self.data.sel(problem=_problem_indices),
        )

    def isel(self, problems: Iterable[int]) -> DataLoader:
        """
        Select a subset of problems of the data loader.

        Parameters
        ----------
        problems : Iterable[int]
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
        return DataLoader(
            task_labels=self.task_labels.isel(problem=problems),
            data=self.data.isel(problem=problems),
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
            range(len(self.problems)), test_size=test_size,
            random_state=random_state)

        return (self.isel(_X_train_indices),
                self.isel(_X_test_indices))

#                                                          Selecting iterations
# =============================================================================

    def select_t(self, t: int, t_start: Optional[int] = 0) -> DataLoader:
        """
        Select a subset of the data loader at a specific iteration.

        Parameters
        ----------
        t : int
            The iteration to select.

        Returns
        -------
        DataLoader
            The selected data loader.
        """
        return DataLoader(
            task_labels=self.task_labels,
            data=self.data.isel(iterations=range(t_start, t)),
        )


#                                                          Selecting optimizers
# =============================================================================

    def select_single_optimizers(
            self, squash: bool = False) -> DataLoader:
        """
        Select only the single optimizers.

        Parameters
        ----------
        include_random_search : bool, optional
            Whether to include the random search, by default False

        Returns
        -------
        DataLoader
            The data loader with only the single optimizers.
        """
        return self.select_optimizer(
            optimizers=[o for o in self.optimizers.values if '+' not in o],
            squash=squash
        )

    def select_optimizer(self, optimizers: Iterable[str] | str,
                         squash: bool = False) -> DataLoader:
        """
        Select a specific optimizer.

        Parameters
        ----------
        optimizer : str
            The optimizer to select.

        Returns
        -------
        DataLoader
            The data loader with only the selected optimizer.
        """
        if isinstance(optimizers, str):
            optimizers = [optimizers]

        if squash:
            data = xr.concat(
                [squash_optimizer(data=self.data, target_optimizer=opt)
                 for opt in optimizers],
                dim='optimizer')
        else:
            data = self.data.sel(optimizer=optimizers)

        return DataLoader(
            task_labels=self.task_labels,
            data=data,
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
            [o for o in self.optimizers.values if o.startswith(optimizer)],
            dtype=object
        )
        data_selected = self.data.sel(
            optimizer=filtered_optimizer).assign_coords(
            optimizer=np.array(
                [o.split('+')[-1] if '+' in o else o
                 for o in filtered_optimizer], dtype=object))

        return DataLoader(
            task_labels=self.task_labels,
            data=data_selected,
        )


#                                                           Performance metrics
# =============================================================================


    def performance_profile(
            self, strategies: Optional[Iterable[xr.DataArray]] = None,
            include_best: bool = False
    ) -> xr.Dataset:

        if strategies is None:
            strategies = []

        if include_best:
            best_performance = self.performance.isel(
                iterations=-1, drop=True).idxmin('optimizer').sortby('problem')
            strategies.append(best_performance)

        _strategies = [*strategies]

        performance_profile = create_performance_profile(
            ds_raw=self.data,
            iterations=self.n_iterations,
            strategies=_strategies)

        return performance_profile

    def rtd_profile(
            self, strategies: Optional[Iterable[xr.DataArray]] = None,
            targets: Optional[np.array] = None,
            include_best: bool = False) -> xr.Dataset:

        if strategies is None:
            strategies = []

        if targets is None:
            rtd = self.performance
        else:
            # TODO: This cannot be computed due to memory errors
            _targets = xr.DataArray(targets, dims=['targets'])
            rtd = (self.performance.expand_dims(
                {'targets': len(_targets)}) > _targets).mean('targets')

        rtd = rtd.sortby('problem').to_dataset('optimizer')

        for strategy in strategies:
            rtd[strategy.name] = self.performance.sortby(
                'problem').sel(optimizer=strategy.sortby('problem')).T

        if include_best:
            best_performance = self.performance.isel(
                iterations=-1, drop=True).idxmin('optimizer').sortby('problem')
            rtd['best'] = self.performance.sortby(
                'problem').sel(optimizer=best_performance)

        return 1.0-rtd

    def load(self):
        self.data.load()

#                                                          OptimizationDynamics
# =============================================================================


class OptimizationDynamics:
    def __init__(
            self,
            X_features: xr.DataArray,
            X_trajectory: xr.DataArray,
            Y: xr.DataArray,
            rel_tolerance: float = 0.0):
        """
        Learning dataset inherited from pytorch Dataset. Used for learning the
        machine learning models

        Parameters
        ----------
        X_features : np.ndarray
            The features of the optimization problems.
        X_trajectory : np.ndarray
            The trajectories of the optimization problems.
        Y : np.ndarray
            The performance value (labels) of the optimization problems.
        """
        # Create mask that only selects problems where we have trajectories
        mask = ~X_trajectory.isnull().all('iterations').isel(
            optimizer=0, drop=True)
        self._X_features = X_features.sel(problem=mask)
        self._X_trajectory = X_trajectory.sel(problem=mask)
        self._Y = Y.sel(problem=mask)
        self.rel_tolerance = rel_tolerance

    def __len__(self):
        """
        Return the length of the dataset. Needed for inheriting from pytorch
        Dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self._X_features)

    def train_test_split(
        self, test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[OptimizationDynamics, OptimizationDynamics]:
        """
        Split the dataset into train and test sets.

        Parameters
        ----------
        test_size : float, optional
            The size of the test set, by default 0.2
        random_state : int, optional
            The random state, by default None

        Returns
        -------
        Tuple[OptimizationDynamics, OptimizationDynamics]
            The train and test datasets.
        """
        train_indices, test_indices = train_test_split(
            range(len(self._X_features)), test_size=test_size,
            random_state=random_state)

        return (self.isel(train_indices),
                self.isel(test_indices))

    def select_problems(self, problems: xr.DataArray) -> OptimizationDynamics:
        return OptimizationDynamics(
            X_features=self._X_features.sel(problem=problems),
            X_trajectory=self._X_trajectory.sel(problem=problems),
            Y=self._Y.sel(problem=problems),
            rel_tolerance=self.rel_tolerance)

    def isel(self, problems: Iterable[int]) -> OptimizationDynamics:
        return OptimizationDynamics(
            X_features=self._X_features.isel(problem=problems),
            X_trajectory=self._X_trajectory.isel(problem=problems),
            Y=self._Y.isel(problem=problems),
            rel_tolerance=self.rel_tolerance)

    def to_feature_data(self):
        return FeaturesDataSet(
            X_features=self._X_features,
            Y_features=self._Y.isel(output_dim=0, drop=True).T,
            rel_tolerance=self.rel_tolerance)

    @ property
    def X_features(self) -> np.ndarray:
        # shape = (n_problems, n_features)
        # return np.expand_dims(self._X_features.values, axis=-1)
        return self._X_features.values

    @ property
    def X_trajectory(self) -> np.ndarray:
        # shape = (n_problems, t, 1)
        # return np.expand_dims(
        #     self._X_trajectory.values.transpose(2, 1, 0), axis=-1)
        return self._X_trajectory.values.transpose(2, 1, 0).squeeze()

    @ property
    def Y(self):
        return (self._Y - self._Y.mean('optimizer')).values.T.squeeze()

    @ property
    def problem(self):
        return self._X_features.problem

    def __getitem__(self, idx):
        """
        Get an item from the dataset. Needed for inheriting from pytorch
        Dataset.

        Parameters
        ----------
        idx : int
            The index of the item to get.
        """
        return (self.X_features[idx], self.X_trajectory[idx]), self.Y[idx]

    def save(self, path: str):
        """
        Save the dataset to disk.

        Parameters
        ----------
        path : str
            The path to save the dataset.
        """
        _path = Path(path)

        # make sure _path exists
        _path.mkdir(parents=True, exist_ok=True)

        self._X_features.to_netcdf(_path / 'X_features.nc')
        self._X_trajectory.to_netcdf(_path / 'X_trajectory.nc')
        self._Y.to_netcdf(_path / 'Y_features.nc')

    @ classmethod
    def load(self, path: str, rel_tolerance: float) -> OptimizationDynamics:
        """
        Load the dataset from disk.

        Parameters
        ----------
        path : str
            The path to load the dataset.

        Returns
        -------
        FeaturesDataSet
            The loaded dataset.
        """
        _path = Path(path)
        X_features = xr.open_dataarray(_path / 'X_features.nc')
        X_trajectory = xr.open_dataarray(_path / 'X_trajectory.nc')
        Y = xr.open_dataarray(_path / 'Y_features.nc')

        return OptimizationDynamics(X_features=X_features,
                                    X_trajectory=X_trajectory,
                                    Y=Y,
                                    rel_tolerance=rel_tolerance)

    def boxplot(self, ax: Optional[plt.Axes] = None,
                subtract_mean: bool = True) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if subtract_mean:
            data = (self._Y - self._Y.mean('optimizer')).squeeze().T
        else:
            data = self._Y.squeeze().T

        return data.to_pandas().boxplot(ax=ax)

# =============================================================================


def save_all_trajectories(data: Dict[str, OptimizationDynamics], path: str):
    """
    Save all trajectories to disk.

    Parameters
    ----------
    data : Dict[str, OptimizationDynamics]
        The optimization dynamics data.
    path : str
        The path to save the data.
    """
    for optimizer, trajectory in data.items():
        trajectory.save(path / optimizer)
        print(f"Saved {optimizer} to {path / optimizer}")


def load_all_trajectories(path: str, rel_tolerance: float
                          ) -> Dict[str, OptimizationDynamics]:
    """
    Load all trajectories from disk.

    Parameters
    ----------
    path : str
        The path to load the data.

    Returns
    -------
    Dict[str, OptimizationDynamics]
        The loaded optimization dynamics data.
    """
    _path = Path(path)
    return {optimizer.stem: OptimizationDynamics.load(_path / optimizer,
                                                      rel_tolerance)
            for optimizer in _path.iterdir()}

# =============================================================================


class FeaturesDataSet:
    def __init__(self, X_features: xr.DataArray, Y_features: xr.DataArray,
                 rel_tolerance: float = 0.0):
        """
        A dataset containing the features and labels of the optimization
        problems.

        Parameters
        ----------
        X_features : pd.DataFrame
            The features of the optimization problems.
        Y_features : pd.DataFrame
            The labels of the optimization problems.
        rel_tolerance : float, optional
            The relative tolerance to use, by default 0.0
        """

        mask = ~Y_features.isnull().all('optimizer')
        self.X_features = X_features.sel(problem=mask)
        self.Y_features = Y_features.sel(problem=mask)
        self.rel_tolerance = rel_tolerance

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X_features)

    def isel(self, problems: Iterable[int]) -> FeaturesDataSet:
        return FeaturesDataSet(
            X_features=self.X_features.isel(problem=problems),
            Y_features=self.Y_features.isel(problem=problems),
            rel_tolerance=self.rel_tolerance)

    @property
    def Y(self) -> xr.DataArray:
        if self.rel_tolerance == 0.0:
            return (self.Y_features.idxmin(
                'optimizer') == self.Y_features.optimizer).T

        threshold = self.Y_features.min(
            dim='optimizer') + abs(
                self.Y_features.min(dim='optimizer') * self.rel_tolerance)

        return (self.Y_features <= threshold).T

    def select_problems(self, problems: xr.DataArray) -> FeaturesDataSet:
        return FeaturesDataSet(
            X_features=self.X_features.sel(problem=problems),
            Y_features=self.Y_features.sel(problem=problems),
            rel_tolerance=self.rel_tolerance)

    def save(self, path: str):
        """
        Save the dataset to disk.

        Parameters
        ----------
        path : str
            The path to save the dataset.
        """
        _path = Path(path)
        self.X_features.to_netcdf(_path / 'X_features.nc')
        self.Y_features.to_netcdf(_path / 'Y_features.nc')

    @ classmethod
    def load(self, path: str, rel_tolerance: float) -> FeaturesDataSet:
        """
        Load the dataset from disk.

        Parameters
        ----------
        path : str
            The path to load the dataset.

        Returns
        -------
        FeaturesDataSet
            The loaded dataset.
        """
        _path = Path(path)
        X_features = xr.open_dataarray(_path / 'X_features.nc')
        Y_features = xr.open_dataarray(_path / 'Y_features.nc')

        return FeaturesDataSet(X_features=X_features,
                               Y_features=Y_features,
                               rel_tolerance=rel_tolerance)

    def train_test_split(
        self, test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[FeaturesDataSet, FeaturesDataSet]:
        """
        Split the dataset into train and test sets.

        Parameters
        ----------
        test_size : float, optional
            The size of the test set, by default 0.2
        random_state : int, optional
            The random state, by default None

        Returns
        -------
        Tuple[OptimizationDynamics, OptimizationDynamics]
            The train and test datasets.
        """
        train_indices, test_indices = train_test_split(
            range(len(self.X_features)), test_size=test_size,
            random_state=random_state)

        return (self.isel(train_indices),
                self.isel(test_indices))


def load_features(path: Path | str, rel_tolerance: float) -> FeaturesDataSet:
    return FeaturesDataSet.load(path=path, rel_tolerance=rel_tolerance)

# =============================================================================


def squash_optimizer(data: xr.Dataset, target_optimizer: str) -> xr.Dataset:
    _data = data.sel(
        optimizer=[o for o in data.optimizer.values
                   if o.startswith(target_optimizer)])
    mask = _data.notnull().any(dim='iterations').idxmax(
        dim='optimizer').isel(output_dim=0, drop=True).output

    return _data.sel(optimizer=mask).reset_coords(
        'optimizer', drop=True).expand_dims(
            optimizer=np.array([target_optimizer], dtype=object))
