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
from typing import Iterable, Optional, Tuple

# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset as torchDataset
from tqdm import tqdm

# Local
from ._logging import logger as l2co_logger
from .content_based import get_labels, get_labels_arbitrary
from .performance import (add_random_samples_combinations,
                          create_performance_profile, scaling)

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
        performance: xr.DataArray,
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
        """
        path = Path(path)
        experimentdata = ExperimentData.from_file(
            path).select_with_status('finished')
        data = xr.open_dataset(
            path / 'entire_dataset.nc',
            chunks='auto')

        try:
            data = data.set_xindex(
                ['realization', 'itemID']).stack('problem')
        except ValueError:
            data = data.stack(problem=('realization', 'itemID'))

        _data = data.unstack('problem')

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

        return _dataloader.select_problems(data.problem)

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

        data = data.sel(itemID=experimentdata.index)
        data_samples = data_samples.sel(itemID=experimentdata.index)

        if add_random_search:
            data = add_random_samples_combinations(
                data=data,
                data_samples=data_samples,
                iterations=config.iterations,
                t_switch=config.t_switch)

        # Features
        if arbitrary_labels:
            X = get_labels_arbitrary(experimentdata=experimentdata,
                                     n_realizations=n_realizations)
        else:
            X = get_labels(experimentdata=experimentdata,
                           n_realizations=n_realizations)

        # # Performance
        # Stack dimensions realization and itemID
        data_stacked = data.stack(problem=('realization', 'itemID'))
        data_samples_stacked = data_samples.stack(
            problem=('realization', 'itemID'))

        # New performance idea
        # performance = scaling(data).stack(
        #     problem=('realization', 'itemID')).sel(
        #     output_dim='y', drop=True)

        performance = scaling(data_stacked)

        performance_stacked = performance.where(
            ~performance.isnull(), other=INVALID_PERFORMANCE_VALUE)

        return cls(X=X,
                   n_iterations=n_iterations,
                   n_realizations=n_realizations,
                   data=data_stacked,
                   performance=performance_stacked.rename('performance'),
                   config=config,
                   experimentdata=experimentdata,
                   data_samples=data_samples_stacked
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

        self.data.reset_index('problem').to_netcdf(_path / 'entire_dataset.nc')

    def to_learning_data(
            self, optimizer: str,
            t: int, t_start: int) -> OptimizationDynamics:
        X_traj = self.select_optimizer(optimizer).select_t(
            t=t, t_start=t_start).X_trajectory(t, t_start=t_start)
        X_features = self.X
        y = self.query_optimizer(optimizer).performance.isel(
            iterations=-1)

        X_features = np.expand_dims(X_features.values, axis=-1)
        X_trajectory = np.expand_dims(
            X_traj.values.transpose(2, 1, 0), axis=-1)
        Y = y.values.T

        return OptimizationDynamics(
            X_features=X_features,
            X_trajectory=X_trajectory,
            Y=Y)

#                                                                    Properties
# =============================================================================

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
        return self.performance.idxmin('optimizer')

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
        _performance_at_last_t = self.performance.isel(
            iterations=-1, drop=True)

        if rel_tolerance == 0.0:
            # return (
            #     self.performance.idxmin(
            #         'optimizer') == self.performance.optimizer).T
            return (_performance_at_last_t.idxmin(
                'optimizer') == self.performance.optimizer).T

        threshold = _performance_at_last_t.min(
            dim='optimizer') + abs(
                _performance_at_last_t.min(dim='optimizer') * rel_tolerance)

        # return (self.performance <= threshold).sel(
        #     problem=self.problem, drop=True).T

        return (_performance_at_last_t <= threshold).T

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
        X_d = self.data.isel(iterations=range(t_start, iterations), drop=True)

        # 2. Make sure the loss is non-negative
        y_min = X_d.output.min(dim=['iterations']).where(
            X_d.output.min(dim=['iterations']) < 0.0, 0.0)
        ds_positive = X_d.output - y_min

        # 3. Make sure the loss is non-zero
        ds_add_to_become_non_zero = (
            ds_positive.min(dim='iterations') == 0.0).where(
            (~(ds_positive.min(dim='iterations') == 0.0)
             ), other=ds_positive.where(
                ds_positive > 0.0).min(
                dim='iterations')
        )

        ds_positive += ds_add_to_become_non_zero

        # 4. Take the base 10 logarithm of the data
        ds_log10_positive = np.log10(ds_positive)

        # 5. Scale the data
        def scale(arr: np.ndarray) -> np.ndarray:
            return np.reshape(MinMaxScaler().fit_transform(
                arr.flatten().reshape(-1, 1)), arr.shape)

        performance: xr.DataArray = xr.apply_ufunc(
            scale, ds_log10_positive,
            input_core_dims=[
                ['iterations', 'output_dim']],
            output_core_dims=[['iterations', 'output_dim']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={'allow_rechunk': True}
        )

        return performance.isel(
            output_dim=0, drop=True).rename(
                'trajectory').transpose(
                    'optimizer', 'iterations', 'problem').fillna(
                        INVALID_PERFORMANCE_VALUE)


#                                                            Selecting problems
# =============================================================================


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

    def balance(self, remove_highest_std: Optional[int] = None) -> DataLoader:
        rtd = self.rtd_profile().to_array(
            dim='optimizer').T.isel(iterations=-1).compute()

        if remove_highest_std is not None:
            X = rtd[rtd.std('optimizer').argsort().values][
                remove_highest_std:]
        else:
            X = rtd[rtd.std('optimizer').argsort().values]

        initial_centers = np.array([
            [0., 0., 0., 1.],  # 1
            [0., 0., 1., 0.],  # 2
            [0., 1., 0., 0.],  # 4
            [1., 0., 0., 0.],  # 8
        ])

        kmeans = KMeans(n_clusters=len(initial_centers), random_state=0,
                        init=initial_centers, n_init=1, max_iter=1).fit(X)
        kmeans.cluster_centers_ = initial_centers
        new_x, new_y = RandomUnderSampler(replacement=True).fit_resample(
            X.problem.values.reshape(-1, 1), kmeans.predict(X))

        return self.select_problems(X.sel(problem=new_x.flatten()).problem)

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
        performance_t = scaling(data)

        performance_stacked_t = performance_t.where(
            ~performance_t.isnull(), other=INVALID_PERFORMANCE_VALUE)

        return DataLoader(
            X=self.X,
            n_iterations=t-t_start,
            n_realizations=self.n_realizations,
            performance=performance_stacked_t,
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
        single_optimizers = [o.values for o in self.performance.optimizer
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
        performance = self.performance.sel(optimizer=optimizers)

        return DataLoader(
            X=self.X,
            n_iterations=self.n_iterations,
            n_realizations=self.n_realizations,
            performance=performance,
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


#                                                           Performance metrics
# =============================================================================

    def performance_profile(
            self, strategies: Optional[Iterable[xr.DataArray]] = None
    ) -> xr.Dataset:

        if strategies is None:
            strategies = []

        _strategies = [*strategies]

        performance_profile = create_performance_profile(
            ds_raw=self.data,
            iterations=self.n_iterations,
            strategies=_strategies)

        return performance_profile

    def rtd_profile(
            self, strategies: Optional[Iterable[xr.DataArray]] = None,
            targets: Optional[np.array] = None) -> xr.Dataset:

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
                'problem').sel(optimizer=strategy).T

        return rtd

#                                                          OptimizationDynamics
# =============================================================================


class OptimizationDynamics(torchDataset):
    def __init__(
            self,
            X_features: np.ndarray,
            X_trajectory: np.ndarray,
            Y: np.ndarray):
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
        self.X_features = X_features
        self.X_trajectory = X_trajectory
        self.Y = Y

    def __len__(self):
        """
        Return the length of the dataset. Needed for inheriting from pytorch
        Dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X_features)

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

    _, df_output = experiment_data.to_pandas()

    # remove all values that say 'ERROR' and 'invalid'
    df_output = df_output.replace('ERROR', np.nan)
    df_output = df_output.replace('invalid', np.nan)
    df_output = df_output.replace('not working', np.nan)

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.applymap(
        lambda x: experiment_data.project_dir / x, na_action='ignore')
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

    xarr_full = xr.concat(combined_rows, dim='itemID')

    if 'input_dim' in xarr_full.coords:
        return xarr_full.drop('input_dim'), experiment_data

    else:
        return xarr_full, experiment_data
