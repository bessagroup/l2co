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

from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local
from ._logging import logger as l2co_logger
from .content_based import get_labels
from .performance import (add_random_samples_combinations,
                          create_performance_profile, scaling2)

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
        X: xr.DataArray,
        n_iterations: int,
        n_realizations: int,
        data: xr.Dataset,
        config: DictConfig,
        experimentdata: ExperimentData,
        data_samples: xr.Dataset
    ):
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
        data : xarray.Dataset
            The raw optimization trajectories.
        config : DictConfig
            The configuration file in hydra format.
        experimentdata : ExperimentData
            The experiment data.
        """
        self.X = X
        self.n_iterations = n_iterations
        self.n_realizations = n_realizations
        self.data = data
        self.config = config
        self.experimentdata = experimentdata
        self.data_samples = data_samples

#                                                      Alternative Constructors
# =============================================================================

    @classmethod
    def from_saved_files(
            cls, path: str, config: Optional[str] = None, **kwargs):
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
        path = Path(path)

        # Retrieve the ExperimentData
        experimentdata = ExperimentData.from_file(
            path).select_with_status('finished')

        # Retrieve the optimization dynamics data
        _data = xr.open_dataset(
            path / 'entire_dataset.nc',
            chunks='auto')

        # try:
        #     data = data.set_xindex(
        #         ['realization', 'itemID'])  # .stack('problem')
        # except ValueError:
        #     data = data.stack(problem=('realization', 'itemID'))

        # _data = data.unstack('problem')

        if (path / 'entire_dataset_samples.nc').exists():
            data_samples = xr.open_dataset(
                path / 'entire_dataset_samples.nc', chunks='auto'
            )
            try:
                data_samples = data_samples.set_xindex(
                    ['realization', 'itemID']
                ).unstack('problem')
            except ValueError:
                pass
        else:
            data_samples = _data
        _dataloader = cls.from_preprocessed_files(
            config=config,
            experimentdata=experimentdata,
            data=_data,
            data_samples=data_samples,
            **kwargs
        )

        return _dataloader.select_problems(_data.problem)

    @classmethod
    def from_preprocessed_files(cls, config: DictConfig,
                                experimentdata: ExperimentData,
                                data: xr.Dataset, data_samples: xr.Dataset,
                                add_random_search: bool = False,
                                arbitrary_labels: bool = False):
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
        add_random_search : bool, optional
            Whether to add the random search roll-outs to the data loader,
            by default False

        Returns
        -------
        DataLoader
            The resulting data loader.
        """

        n_realizations = data.realization.size
        n_iterations = data.iterations.size

        _, _df_output = experimentdata.to_pandas()
        # remove all values that say 'ERROR' and 'invalid'
        _df_output = _df_output.replace(
            ['ERROR', 'invalid', 'not working'], np.nan)

        # remove all experiments where none of the optimizers have finished
        if 'random_samples' in _df_output.columns:
            _valid_indices = _df_output.drop(
                columns=['random_samples']).dropna(how='all').index

        else:
            _valid_indices = _df_output.dropna(how='all').index

        experimentdata = experimentdata.select(_valid_indices)

        # get column names where all rows are nan
        _columns_to_drop = _df_output.columns[_df_output.isna().all()]
        experimentdata = experimentdata.drop_output(_columns_to_drop)

        if add_random_search:
            data = add_random_samples_combinations(
                data=data,
                data_samples=data_samples,
                iterations=config.iterations,
                t_switch=config.t_switch)

        # Features
        if arbitrary_labels:
            pass

        else:
            X = get_labels(experimentdata=experimentdata,
                           data=data)

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   data=data,
                   config=config,
                   experimentdata=experimentdata,
                   data_samples=data_samples
                   )

#                                                         Storing and exporting
# =============================================================================

    def store(self, path: str, store_samples: bool = False):
        """
        Store the data loader to disk.

        Parameters
        ----------
        path : str
            The path to store the data loader.
        """
        _path = Path(path)
        _path.mkdir(parents=True, exist_ok=True)

        self.experimentdata.store(_path)

        if store_samples:
            self.data_samples.reset_index('problem').to_netcdf(
                _path / 'entire_dataset_samples.nc')

        # Writing dask arrays to disk is poorly optimized, so we compute
        # the data before writing it to disk
        # Source: https://github.com/pydata/xarray/issues/2912
        self.data.reset_index('problem').compute(
        ).to_netcdf(_path / 'entire_dataset.nc')

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
        X_trajectory = self.select_optimizer(optimizer).select_t(
            t=t, t_start=t_start).X_trajectory()
        X_features = self.X
        Y = self.query_optimizer(optimizer).performance.isel(
            iterations=-1)

        return OptimizationDynamics(
            X_features=X_features,
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

    def to_feature_data(self, rel_tolerance: float = 0.0,
                        optimizer: Optional[str] = None) -> FeaturesDataSet:
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

        X_features = self.X

        if optimizer is None:
            Y_features = self.select_single_optimizers().performance.isel(
                iterations=-1, output_dim=0, drop=True).T
        else:
            Y_features = self.query_optimizer(optimizer).performance.isel(
                iterations=-1, output_dim=0, drop=True).T

        return FeaturesDataSet(X_features=X_features,
                               Y_features=Y_features,
                               rel_tolerance=rel_tolerance)
#                                                                    Properties
# =============================================================================

    @ property
    def features(self) -> xr.DataArray:
        """
        Property method to return the features coordinate

        Returns
        -------
        xarray.DataArray
            The features coordinate.
        """
        return self.X.features

    @ property
    def problem(self) -> xr.DataArray:
        """
        Property method to return the problem coordinate

        Returns
        -------
        xarray.DataArray
            The problem coordinate.
        """
        return self.X.problem

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

    @property
    def performance(self) -> xr.DataArray:
        t_start = perf_counter()
        print("Triggered performance")
        performance = scaling2(self.data)

        performance = performance.where(
            ~performance.isnull(), other=INVALID_PERFORMANCE_VALUE)
        print(f"Performance calculation took: {perf_counter()-t_start:0.2f}")
        return performance

    @ lru_cache(maxsize=1)
    def y_binary(self, rel_tolerance: float = 0.1) -> xr.DataArray:
        _performance_at_last_t = self.performance.isel(
            iterations=-1, output_dim=0, drop=True)

        if rel_tolerance == 0.0:
            return (_performance_at_last_t.idxmin(
                'optimizer') == self.data.optimizer).T

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
                    'optimizer', 'iterations', 'problem').fillna(
                        INVALID_PERFORMANCE_VALUE)

        # X_d = self.data.isel(iterations=range(
        #     t_start, iterations), drop=True)  # needed?

        # # 2. Make sure the loss is non-negative
        # y_min = X_d.output.min(dim=['iterations']).where(
        #     X_d.output.min(dim=['iterations']) < 0.0, 0.0)
        # ds_positive = X_d.output - y_min

        # # 3. Make sure the loss is non-zero
        # ds_add_to_become_non_zero = (
        #     ds_positive.min(dim='iterations') == 0.0).where(
        #     (~(ds_positive.min(dim='iterations') == 0.0)
        #      ), other=ds_positive.where(
        #         ds_positive > 0.0).min(
        #         dim='iterations')
        # )

        # ds_positive += ds_add_to_become_non_zero

        # # 4. Take the base 10 logarithm of the data
        # ds_log10_positive = np.log10(ds_positive)

        # # 5. Scale the data
        # def scale(arr: np.ndarray) -> np.ndarray:
        #     return np.reshape(MinMaxScaler().fit_transform(
        #         arr.flatten().reshape(-1, 1)), arr.shape)

        # performance: xr.DataArray = xr.apply_ufunc(
        #     scale, ds_log10_positive,
        #     input_core_dims=[
        #         ['iterations', 'output_dim']],
        #     output_core_dims=[['iterations', 'output_dim']],
        #     vectorize=True,
        #     dask='parallelized',
        #     dask_gufunc_kwargs={'allow_rechunk': True}
        # )

        # return performance.isel(
        #     output_dim=0, drop=True).rename(
        #         'trajectory').transpose(
        #             'optimizer', 'iterations', 'problem').fillna(
        #                 INVALID_PERFORMANCE_VALUE)


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

        # Select unique itemIDs to select the experiment data
        _unique_itemID = np.unique(problem_indices.itemID.values)

        return DataLoader(
            X=self.X.sel(problem=_problem_indices),
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            data=self.data.sel(problem=_problem_indices),
            config=self.config,
            experimentdata=self.experimentdata.select(_unique_itemID),
            data_samples=self.data_samples.sel(problem=_problem_indices)
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

        _problem_indices = self.problem.isel(problem=problems).sortby('itemID')

        # Select unique itemIDs to select the experiment data
        _unique_itemID = np.unique(_problem_indices.itemID.values)

        return DataLoader(
            X=self.X.isel(problem=problems),
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            data=self.data.isel(problem=problems),
            config=self.config,
            experimentdata=self.experimentdata.select(_unique_itemID),
            data_samples=self.data_samples.isel(problem=problems)
        )

        # return self.select_problems(self.problem.isel(problem=problems))

    # def balance(self, remove_highest_std: Optional[int] = None):
    #     rtd = self.rtd_profile().to_array(
    #         dim='optimizer').T.isel(iterations=-1).compute()

    #     if remove_highest_std is not None:
    #         X = rtd[rtd.std('optimizer').argsort().values][
    #             remove_highest_std:]
    #     else:
    #         X = rtd[rtd.std('optimizer').argsort().values]

    #     initial_centers = np.array([
    #         [0., 0., 0., 1.],
    #         [0., 0., 1., 0.],
    #         [0., 1., 0., 0.],
    #         [1., 0., 0., 0.],
    #     ])

    #     kmeans = KMeans(n_clusters=len(initial_centers), random_state=0,
    #                     init=initial_centers, n_init=1, max_iter=1).fit(X)
    #     kmeans.cluster_centers_ = initial_centers
    #     new_x, new_y = RandomUnderSampler(replacement=True).fit_resample(
    #         X.problem.values.reshape(-1, 1), kmeans.predict(X))

    #     return self.select_problems(X.sel(problem=new_x.flatten()).problem)

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
            range(len(self.X.problem)), test_size=test_size,
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
        data = self.data.isel(iterations=range(t_start, t))
        data_samples = self.data_samples.isel(iterations=range(t_start, t))

        # Recalculate the performance since the budget has changed
        # performance_t = scaling2(data)

        # # Making sure that we mask NaN values
        # performance_stacked_t = performance_t.where(
        #     ~performance_t.isnull(), other=INVALID_PERFORMANCE_VALUE)

        return DataLoader(
            X=self.X,
            n_iterations=t-t_start,
            n_realizations=self.n_realizations,
            # performance=performance_stacked_t,
            data=data,
            config=self.config,
            experimentdata=self.experimentdata,
            data_samples=data_samples
        )


#                                                          Selecting optimizers
# =============================================================================

    def select_single_optimizers(
            self, include_random_search: bool = False) -> DataLoader:
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
        single_optimizers = [o.values for o in self.data.optimizer
                             if '+' not in str(o.values)]

        if include_random_search:
            single_optimizers += [f'random_samples+{opt}' for opt in
                                  single_optimizers]

        return self.select_optimizer(single_optimizers)

    def select_optimizer(self, optimizers: Iterable[str] | str) -> DataLoader:
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

        data_selected = self.data.sel(optimizer=optimizers)
        # performance = self.performance.sel(optimizer=optimizers)

        return DataLoader(
            X=self.X,
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            # performance=performance,
            data=data_selected,
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
        # data_selected = self.data.sel(optimizer=[optimizer])
        data_selected = self.data.sel(
            optimizer=filtered_optimizer).assign_coords(
            optimizer=np.array(
                [o.split('+')[-1] if '+' in o else o
                 for o in filtered_optimizer], dtype=object))

        n_iterations = self.n_iterations
        n_realizations = self.n_realizations

        # Config
        config = self.config

        # # Performance
        # performance = self.performance.sel(
        #     optimizer=filtered_optimizer).assign_coords(
        #     optimizer=np.array(
        #         [o.split('+')[-1] if '+' in o else o
        #          for o in filtered_optimizer], dtype=object))

        return DataLoader(
            X=self.X,
            n_iterations=n_iterations,
            n_realizations=n_realizations,
            # performance=performance,
            data=data_selected,
            config=config,
            experimentdata=self.experimentdata,
            data_samples=self.data_samples
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
        self._X_features = X_features
        self._X_trajectory = X_trajectory
        self._Y = Y
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

    @property
    def X_features(self) -> np.ndarray:
        # shape = (n_problems, n_features)
        # return np.expand_dims(self._X_features.values, axis=-1)
        return self._X_features.values

    @property
    def X_trajectory(self) -> np.ndarray:
        # shape = (n_problems, t, 1)
        # return np.expand_dims(
        #     self._X_trajectory.values.transpose(2, 1, 0), axis=-1)
        return self._X_trajectory.values.transpose(2, 1, 0).squeeze()

    @property
    def Y(self):
        return (self._Y - self._Y.mean('optimizer')).values.T.squeeze()

    @property
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

    @classmethod
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
        self.X_features = X_features
        self.Y_features = Y_features
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

    @classmethod
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


def create_entire_dataset(experiment_data: ExperimentData):
    """
    Create an entire dataset from a given experiment data object.

    Parameters
    ----------
    experiment_data : f3dasm.ExperimentData
        The experiment data object.

    Returns
    -------
    xarray.Dataset
        The resulting entire dataset.

    Notes
    -----
    The entire dataset is created by concatenating all the optimization
    trajectories (NetCDF files) of the experiment data object.

    All of the files are checked on:
    - if they are flagged as 'invalid' or 'ERROR'
    - if they contain np.nan values
    """

    experiment_data = experiment_data.select_with_status('finished')

    _, df_output = experiment_data.to_pandas(keep_references=True)

    # remove all values that say 'ERROR' and 'invalid'
    df_output = df_output.replace('ERROR', np.nan)
    df_output = df_output.replace('invalid', np.nan)
    df_output = df_output.replace('not working', np.nan)

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.applymap(
        lambda x: experiment_data.project_dir / 'experiment_data' / x,
        na_action='ignore')
    # df_output = df_output.apply(lambda x: experiment_data.project_dir / x)

    # don't include the column 'random_samples'
    if 'random_samples' in df_output.columns:
        df_output = df_output.drop(columns=['random_samples'])

    # remove all experiments where none of the optimizers have finished
    df_output = df_output.dropna(how='all')

    # Open the whole thing:
    l2co_logger.info("Opening the whole thing")

    combined_rows = []
    faulty_experiments = []

    for row, values in tqdm(df_output.iterrows(), total=len(df_output)):
        try:
            a = xr.open_mfdataset(values.dropna(), combine='by_coords')
        except OSError as e:
            l2co_logger.error(f"Error: {e}")
            faulty_experiments.append(row)
            continue

        if 'input' in a.data_vars:
            combined_rows.append(a.assign_coords(
                {'itemID': [row]}).drop('input'))
        else:
            combined_rows.append(a.assign_coords(
                {'itemID': [row]}))

        l2co_logger.info(f"Entire dataset: {row}/{len(df_output)}")

    experiment_data.mark(indices=faulty_experiments, status='error')

    xarr_full: xr.Dataset = xr.concat(combined_rows, dim='itemID')

    xarr_full = xarr_full.stack(
        problem=('realization', 'itemID'), create_index=False
    )

    xarr_full = xarr_full.assign_coords(
        {'problem': np.arange(xarr_full.problem.size)})

    if 'input_dim' in xarr_full.coords:
        return xarr_full.drop('input_dim'), experiment_data

    else:
        return xarr_full, experiment_data
