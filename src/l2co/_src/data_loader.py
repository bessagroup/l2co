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

import json
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import svd
from sklearn.model_selection import train_test_split

# Local
from .performance import create_performance_profile, scaling2
from .tasks.benchmarkfunctions.optimizerschedule import OptimizationSchedule

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
        opt_steps: Dict[str, OptimizationSchedule]
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
        self.opt_steps = opt_steps

#                                                      Alternative Constructors
# =============================================================================

    @classmethod
    def from_file(
        cls, path: str, labels_filename: str = 'labels',
            data_filename: str = 'entire_dataset',
            opt_steps_filename: str = 'steps'):
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

        # Open opt steps
        with open(Path(_path / opt_steps_filename
                       ).with_suffix('.json'), 'r') as f:
            opt_steps = {k: OptimizationSchedule.from_dict(
                v) for k, v in json.load(f).items()}

        return cls(data=data,
                   task_labels=task_labels,
                   opt_steps=dict(sorted(opt_steps.items())),
                   )

#                                                         Storing and exporting
# =============================================================================

    def to_learning_data(
            self, t: int, t_start: int = 0, t_end: int = 0,
            optimizer: Optional[str] = None) -> OptimizationDynamics:
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

        Returns
        -------
        OptimizationDynamics
            The resulting optimization dynamics.

        Notes
        -----
        OptimizationDynamics is a pytorch Dataset object.
        """
        if optimizer is None:
            X_trajectory = None
            Y_data = self.select_single_optimizers(squash=True).select_t(
                t_start=t, t=t_end)

            Y = Y_data.performance.isel(
                iterations=t_end-1, output_dim=0, drop=True).sortby(
                    'optimizer')
            previous_step = None

        else:
            X_trajectory = compute_X_trajectory(self.select_optimizer(
                optimizer, squash=True).select_t(
                t_start=t_start, t=t).data)
            Y_data = self.query_optimizer(optimizer)

            Y = Y_data.performance.isel(
                iterations=t_end-1, output_dim=0, drop=True).sortby(
                    'optimizer')
            previous_step = self.opt_steps[optimizer]

        return OptimizationDynamics.from_data(
            X_features=self.task_labels,
            X_trajectory=X_trajectory,
            Y=Y,
            opt_steps=Y_data.opt_steps,
            previous_step=previous_step,
        )

    def export_all_and_store(
            self, path: str, t: int, t_start: int, t_end: int):

        _path = Path(path)
        _path.mkdir(parents=True, exist_ok=True)

        optimizers = self.select_single_optimizers().optimizers.values

        for optimizer in optimizers:
            dynamics = self.to_learning_data(
                optimizer=optimizer,
                t=t, t_start=t_start, t_end=t_end)

            dynamics.save(_path / str(optimizer))
            del dynamics

        dynamics = self.to_learning_data(
            t_start=t_start, t=t_start, t_end=t_end)
        dynamics.save(_path / 'first')

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

        print(f"Performance calculation took: {perf_counter()-t_start:0.2f}")
        return performance

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
            opt_steps=self.opt_steps,
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
            opt_steps=self.opt_steps,
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
        if t == 0:
            t = self.n_iterations

        return DataLoader(
            task_labels=self.task_labels,
            data=self.data.isel(iterations=range(t_start, t)),
            opt_steps=self.opt_steps
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

        opt_steps = dict(filter(lambda item: item[0] in
                                optimizers, self.opt_steps.items()))

        return DataLoader(
            task_labels=self.task_labels,
            data=data,
            opt_steps=dict(sorted(opt_steps.items()))
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

        # Opt steps
        filtered_schedules = dict(
            filter(lambda item: item[0].startswith(optimizer),
                   self.opt_steps.items()))

        new_schedules: Dict[str, OptimizationSchedule] = {}
        for k, v in filtered_schedules.items():

            name = k.split('+')[-1] if '+' in k else k
            v_copy = v.copy()
            v_copy.name = name
            v_copy.steps = [v_copy.steps[-1]]
            new_schedules[name] = v_copy

        new_schedules[optimizer].steps[0].x0_strategy = 'last'

        return DataLoader(
            task_labels=self.task_labels,
            data=data_selected,
            opt_steps=dict(sorted(new_schedules.items()))
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
    def __init__(self,
                 X_features: pd.DataFrame,
                 X_trajectory: Optional[np.ndarray],
                 Y: np.ndarray,
                 opt_steps: Dict[str, OptimizationSchedule],
                 previous_step: Optional[OptimizationSchedule]):

        self.X_features = X_features
        self.X_trajectory = X_trajectory
        self.Y = Y
        self.opt_steps = opt_steps
        self.previous_step = previous_step

    @property
    def has_trajectory(self) -> bool:
        """
        Property that returns whether the model has a trajectory

        Returns
        -------
        bool
            Whether the model has a trajectory
        """
        return self.X_trajectory is not None

    @classmethod
    def from_data(
        cls,
        X_features: xr.DataArray,
        X_trajectory: Optional[xr.DataArray],
        Y: xr.DataArray,
        opt_steps: Dict[str, OptimizationSchedule],
        previous_step: Optional[OptimizationSchedule]
    ):
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

        mask = ~Y.isnull().all('optimizer')

        if X_trajectory is not None:
            mask_X_trajectory = ~X_trajectory.isnull().all('iterations').isel(
                optimizer=0, drop=True)

            mask = mask & mask_X_trajectory
            X_trajectory = X_trajectory.sel(problem=mask)
            X_trajectory = X_trajectory.values.transpose(
                2, 1, 0).squeeze()

        Y = Y.sel(problem=mask).values

        Y = impute_missing_values_svd(Y).T

        X_features = X_features.sel(
            problem=mask).to_pandas().infer_objects().convert_dtypes()

        return cls(X_features=X_features,
                   X_trajectory=X_trajectory,
                   Y=Y,
                   opt_steps=opt_steps,
                   previous_step=previous_step
                   )

    @classmethod
    def load(cls, path: Path | str) -> OptimizationDynamics:
        """
        Load the model from a file

        Parameters
        ----------
        path : Path
            The path to the data

        Returns
        -------
        OptimizationDynamics
            The loaded optimization dynamics data.
        """
        _path = Path(path)
        data = np.load(_path.with_suffix('.npz'), allow_pickle=True)
        if (data['X_trajectory'].ndim == 0
                and data['X_trajectory'].item() is None):
            X_trajectory = None
        else:
            X_trajectory = data['X_trajectory']

        features = pd.read_csv(_path.with_suffix('.csv'), index_col=0,
                               header=0).convert_dtypes()

        # Open opt steps
        with open(_path.with_suffix('.json'), 'r') as f:
            opt_steps_dict = json.load(f)

        previous_step = opt_steps_dict.pop('previous_step', None)

        if previous_step is not None:
            previous_step = OptimizationSchedule.from_dict(previous_step)

        opt_steps = {k: OptimizationSchedule.from_dict(
            v) for k, v in opt_steps_dict.items()}

        return cls(X_features=features,
                   X_trajectory=X_trajectory,
                   Y=data['Y'],
                   opt_steps=opt_steps,
                   previous_step=previous_step)

    def save(self, path: Path | str) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        path : Path
            The path to the model
        """
        _path = Path(path)
        # Save trajectory and y
        np.savez(_path.with_suffix('.npz'), **self.to_dict())

        # Save features
        self.X_features.to_csv(_path.with_suffix('.csv'))

        opt_steps_dict = {k: v.to_dict()
                          for k, v in self.opt_steps.items()}

        if self.previous_step is not None:
            opt_steps_dict['previous_step'] = self.previous_step.to_dict()
        else:
            opt_steps_dict['previous_step'] = None

        with open(_path.with_suffix('.json'), 'w') as f:
            json.dump(opt_steps_dict, f, indent=4)

    def Y_binary(self, rel_tolerance: float = 0.0):
        if rel_tolerance == 0.0:
            return self.Y == self.Y.min(axis=1, keepdims=True)
        else:
            min_values = self.Y.min(axis=1, keepdims=True)
            threshold = min_values + abs(min_values * rel_tolerance)
            return self.Y <= threshold

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {'X_trajectory': self.X_trajectory,
                'Y': self.Y}


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


def squash_optimizer(data: xr.Dataset, target_optimizer: str) -> xr.Dataset:
    _data = data.sel(
        optimizer=[o for o in data.optimizer.values
                   if o.startswith(target_optimizer)])
    mask = _data.notnull().any(dim='iterations').idxmax(
        dim='optimizer').isel(output_dim=0, drop=True).output

    return _data.sel(optimizer=mask).reset_coords(
        'optimizer', drop=True).expand_dims(
            optimizer=np.array([target_optimizer], dtype=object))


def impute_missing_values_svd(y: np.ndarray, k=2):
    mask = ~np.isnan(y)
    mean_values = np.nanmean(y, axis=0)

    # Center the data (subtract mean from non-nan values)
    centered_matrix = np.where(mask, y - mean_values, 0)

    # Perform SVD
    U, s, Vt = svd(centered_matrix, full_matrices=False)

    # Keep only k components
    S_k = np.diag(s[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]

    # Reconstruct the matrix
    reconstructed = U_k @ S_k @ Vt_k + mean_values

    # Fill in missing values
    filled_matrix = np.where(mask, y, reconstructed)

    return filled_matrix


# @ lru_cache(maxsize=1)
def compute_X_trajectory(data: xr.Dataset) -> xr.DataArray:
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
        data,
        input_core_dims=[['iterations', 'output_dim']],
        output_core_dims=[['iterations', 'output_dim']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}  # False
    )

    return trajectory.output.isel(
        output_dim=0, drop=True).rename(
            'trajectory').transpose(
                'optimizer', 'iterations', 'problem')


def preprocess_task_labels(task_labels: xr.DataArray, ) -> xr.DataArray:
    ...
