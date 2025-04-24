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
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Literal, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from scipy.linalg import svd
from sklearn.model_selection import train_test_split

# Local
from .optimizerschedule import OptimizationSchedule, OptimizationStep
from .performance import scaling2
from .task import Task

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
        opt_steps: Dict[str, OptimizationSchedule],
        l2co_schedules: Optional[xr.Dataset] = None,
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
        self.l2co_schedules = l2co_schedules

    def __add__(self, __o: DataLoader) -> DataLoader:
        if __o is None:
            return self

        data = xr.concat([self.data, __o.data], dim='problem')
        task_labels = xr.concat([self.task_labels, __o.task_labels],
                                dim='problem')
        opt_steps = {**self.opt_steps, **__o.opt_steps}

        if self.l2co_schedules is not None:
            l2co_schedules = xr.concat(
                [self.l2co_schedules, __o.l2co_schedules], dim='problem')
        else:
            l2co_schedules = __o.l2co_schedules

        return DataLoader(data=data, task_labels=task_labels,
                          opt_steps=opt_steps, l2co_schedules=l2co_schedules)


#                                                      Alternative Constructors
# =============================================================================

    @classmethod
    def from_file(
        cls, path: str, data_filename: str = 'entire_dataset'
    ) -> DataLoader:
        # Create path object
        _path = Path(path)

        if (_path / 'steps.json').exists():
            return cls.from_file_legacy(path=path, data_filename=data_filename)

        # Retrieve the optimization dynamics data
        try:
            data = xr.open_dataset(
                (_path / data_filename).with_suffix('.nc'))
        except FileNotFoundError:
            data = xr.open_dataset(
                (_path / data_filename).with_suffix('.odata'))

        if (_path / data_filename).with_suffix(
                '.osched').exists():
            l2co_schedules = xr.open_dataset(
                (_path / data_filename).with_suffix('.osched'))

            # Assuming 'schedules' is the variable with JSON strings
            def parse_schedule(json_str):
                return OptimizationSchedule.from_json_string(json_str)

            l2co_schedules = xr.apply_ufunc(
                parse_schedule,
                l2co_schedules["schedules"],
                vectorize=True,
                output_dtypes=[object]
            )
        else:
            l2co_schedules = None

        def parse_mixed_value(val: str):
            """Try to convert a string into int, float, or leave as string."""
            if val.lower() == "true":
                return True
            elif val.lower() == "false":
                return False
            try:
                if "." in val or "e" in val.lower():
                    return float(val)
                return int(val)
            except ValueError:
                return val

        def my_f(x):
            return np.vectorize(parse_mixed_value)(x)

        task_labels = xr.apply_ufunc(
            my_f,
            data['task_labels'],
            input_core_dims=[["problem"]],
            output_core_dims=[["problem"]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[object],
        ).T

        # Opt steps
        opt_steps = {name: OptimizationSchedule.from_json_string(
            opt) for name, opt in data['opt_steps'].to_series().items()}

        # delete opt_step and task_labels from data
        data = data.drop_vars(
            ['opt_steps', 'task_labels']).drop_dims('features')

        return cls(
            data=data,
            task_labels=task_labels,
            opt_steps=dict(sorted(opt_steps.items())),
            l2co_schedules=l2co_schedules
        )

    @classmethod
    def from_file_legacy(
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
            (_path / data_filename).with_suffix('.nc'))

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
            self, t: float, t_start: float = 0.0, t_end: float = 1.0,
            optimizer: Optional[str] = None) -> OptimizationDynamics:
        """
        Convert the data loader to learning data.

        Parameters
        ----------
        optimizer : str
            The optimizer to select.
        t : float
            The budget to select.
        t_start : int, optional
            The starting budget number of the requested trajectory,
            by default 0.0
        t_end : int, optional
            The ending budget number of the requested trajectory,
            by default 0.0

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

            Y = Y_data.performance.sel(
                budget=t_end, method='nearest', drop=True).isel(
                output_dim=0, drop=True).sortby(
                    'optimizer')
            previous_step = None

        else:
            X_trajectory = compute_X_trajectory(self.select_optimizer(
                optimizer, squash=True).select_t(
                t_start=t_start, t=t).data)
            Y_data = self.query_optimizer(optimizer)

            Y = Y_data.performance.sel(
                budget=t_end, method='nearest', drop=True).isel(
                output_dim=0, drop=True).sortby(
                    'optimizer')
            previous_step = self.opt_steps[optimizer].steps[0]

        return OptimizationDynamics.from_data(
            X_features=self.task_labels,
            X_trajectory=X_trajectory,
            Y=Y,
            opt_steps={k: v.steps[0] for k, v in Y_data.opt_steps.items()},
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

    def to_spreadsheet(self, criteria: Literal['min', 'mean'] = 'min'):
        ds = xr.Dataset({'task_labels': self.task_labels,
                         'output_min': self.data.output_min})

        if criteria == 'min':
            df = ds.groupby('itemID').min().min('budget').isel(
                output_dim=0).output_min.to_pandas()
        elif criteria == 'mean':
            df = ds.groupby('itemID').mean().min('budget').isel(
                output_dim=0).output_min.to_pandas()

        df.index = ds.groupby('itemID').min().min(
            'budget').isel(output_dim=0).task_labels

        # Function to highlight the smallest value per row
        def highlight_min_row(row):
            min_val = row.min()
            return ['font-weight: bold' if
                    val == min_val else '' for val in row]

        # Apply styling and scientific notation formatting
        styled_df = (
            df.style
            # Highlight smallest value per row
            .apply(highlight_min_row, axis=1)
            .format("{:.2e}")  # Display in scientific notation
        )

        return styled_df

    def display_sbs_table(
            self, exclude_optimizer: str | Iterable[str]) -> pd.DataFrame:
        sbs = self._calculate_sbs(performance=self.performance,
                                  exclude_optimizer=exclude_optimizer)
        ds = xr.Dataset({'sbs': sbs,
                         'task_labels': self.task_labels, })

        df_task_labels = ds.groupby('itemID').min().task_labels.to_pandas()
        df_sbs = ds.groupby('itemID').min().sbs.to_pandas().rename('sbs')

        return pd.concat([df_task_labels, df_sbs], axis=1)

    def display_l2co_schedules_table(self) -> pd.DataFrame:

        task_labels_l2co = self.task_labels.groupby('itemID').min().sel(
            itemID=self.l2co_schedules.itemID)

        ds = xr.Dataset({'task_labels': task_labels_l2co,
                        'schedules': self.l2co_schedules, }
                        ).sortby('itemID')

        df_schedules = ds.schedules.isel(schedule=0, optimizer=0).to_pandas()

        df_schedules = df_schedules.apply(
            lambda row: [s.name for s in row.steps])
        df_schedules = pd.DataFrame(
            df_schedules.tolist(), index=df_schedules.index)

        df_itemID = ds.task_labels.itemID.to_pandas()
        df_itemID.name = 'itemID'

        return pd.concat([
            df_itemID,
            ds.task_labels.to_pandas(),
            df_schedules
        ], axis=1)

#                                                                    Properties
# =============================================================================

    @property
    def n_iterations(self) -> int:
        return self.data.budget.size

    @property
    def n_realization(self) -> int:
        return np.unique(self.data.realization).size

    @property
    def features(self) -> xr.DataArray:
        """
        Property method to return the features coordinate

        Returns
        -------
        xarray.DataArray
            The features coordinate.
        """
        return self.task_labels.features

    @property
    def problems(self) -> xr.DataArray:
        """
        Property method to return the problem coordinate

        Returns
        -------
        xarray.DataArray
            The problem coordinate.
        """
        return self.task_labels.problem

    @property
    def optimizers(self) -> xr.DataArray:
        """
        Property method to return the optimizers coordinate

        Returns
        -------
        xarray.DataArray
            The optimizers coordinate.
        """
        return self.data.optimizer

    def _performance(self, include_best: bool,
                     exclude_optimizer: Optional[str] = None) -> xr.DataArray:
        t_start = perf_counter()
        print("Triggered performance")
        performance = scaling2(self.data)

        performance = performance.ffill(dim='budget')

        if include_best:
            sbs = self._calculate_sbs(performance=performance,
                                      exclude_optimizer=exclude_optimizer)
            perf_sbs = performance.sel(optimizer=sbs, drop=True).reset_coords(
                'optimizer', drop=True).expand_dims(optimizer=['sbs'])
            performance = xr.merge([performance, perf_sbs]).output_min

        print(f"Performance calculation took: {perf_counter()-t_start:0.2f}")
        return performance

    def _calculate_sbs(self, performance: xr.DataArray,
                       exclude_optimizer: str | Iterable[str]) -> xr.DataArray:
        if exclude_optimizer is not None:
            if isinstance(exclude_optimizer, str):
                exclude_optimizer = [exclude_optimizer]

            check_sbs = performance.where(
                ~performance.optimizer.isin(exclude_optimizer), drop=True)
        else:
            check_sbs = performance

        sbs = check_sbs.groupby('itemID').map(
            lambda x: x.mean('problem').isel(
                budget=-1, drop=True).idxmin(
                'optimizer').expand_dims(problem=x.problem)).isel(
            output_dim=0, drop=True).assign_coords(
                    {'problem': performance.problem})
        sbs = sbs.fillna(performance.optimizer.values[0])
        return sbs

    @property
    def performance(self) -> xr.DataArray:
        return self._performance(include_best=False,
                                 exclude_optimizer=None)


#                                                            Selecting problems
# =============================================================================

    def select_itemIDs(self, itemIDs: str | Iterable[str]) -> DataLoader:

        if len(itemIDs) == 0:
            return deepcopy(self)

        if isinstance(itemIDs, str):
            itemIDs = [itemIDs]

        task_labels = xr.concat(
            [group for i, group in self.task_labels.groupby(
                'itemID') if i in itemIDs], dim="problem")
        data = xr.concat(
            [group for i, group in self.data.groupby(
                'itemID') if i in itemIDs], dim="problem")

        if self.l2co_schedules is not None:
            l2co_schedules = xr.concat(
                [group for i, group in self.l2co_schedules.groupby(
                    'itemID') if i in itemIDs], dim="problem")
        else:
            l2co_schedules = None

        return DataLoader(
            task_labels=task_labels,
            data=data,
            opt_steps=self.opt_steps,
            l2co_schedules=l2co_schedules
        )

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

        if self.l2co_schedules is not None:
            l2co_schedules = self.l2co_schedules.sel(problem=_problem_indices)
        else:
            l2co_schedules = None

        return DataLoader(
            task_labels=self.task_labels.sel(problem=_problem_indices),
            data=self.data.sel(problem=_problem_indices),
            opt_steps=self.opt_steps,
            l2co_schedules=l2co_schedules
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

        if self.l2co_schedules is not None:
            l2co_schedules = self.l2co_schedules.isel(problem=problems)
        else:
            l2co_schedules = None

        return DataLoader(
            task_labels=self.task_labels.isel(problem=problems),
            data=self.data.isel(problem=problems),
            opt_steps=self.opt_steps,
            l2co_schedules=l2co_schedules
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

#                                                              Selecting budget
# =============================================================================

    def select_t(self, t: int, t_start: Optional[float] = 0.0) -> DataLoader:
        """
        Select a subset of the data loader at a specific budget.

        Parameters
        ----------
        t : int
            The budget to select.
        t_start : float, optional
            The starting budget number of the requested trajectory,

        Returns
        -------
        DataLoader
            The selected data loader.
        """
        return DataLoader(
            task_labels=self.task_labels,
            data=self.data.sel(budget=slice(t_start, t)),
            opt_steps=self.opt_steps,
            l2co_schedules=self.l2co_schedules
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
            opt_steps=dict(sorted(opt_steps.items())),
            l2co_schedules=self.l2co_schedules
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
            opt_steps=dict(sorted(new_schedules.items())),
            l2co_schedules=self.l2co_schedules,
        )


#                                                           Performance metrics
# =============================================================================


    def performance_profile(
            self, max_f_value: int = 2.0, include_best: bool = False,
            exclude_optimizer: Optional[str] = None,
    ) -> xr.DataArray:

        # Trigger performance
        p = self._performance(
            include_best=include_best,
            exclude_optimizer=exclude_optimizer)

        return performance_profile(performance=p, max_f_value=max_f_value)

    def rtd_profile(self, include_best: bool = False,
                    exclude_optimizer: Optional[str] = None,) -> xr.DataArray:
        p = self._performance(
            include_best=include_best,
            exclude_optimizer=exclude_optimizer)

        return rtd(performance=p)

    def load(self):
        self.data.load()

    def save(self, path: str):
        _path = Path(path)

        _path.mkdir(parents=True, exist_ok=True)

        # Save optimization dynamics data
        self.data.to_netcdf((_path / 'entire_dataset').with_suffix('.nc'))

        # Convert task_labels to a DataFrame and save as CSV
        df_task_labels = pd.DataFrame(
            self.task_labels.values,
            index=self.task_labels.coords['itemID'].values,
            columns=self.task_labels.coords['features'].values
        )
        df_task_labels = df_task_labels[~df_task_labels.index.duplicated(
            keep='first')]
        df_task_labels.to_csv((_path / 'labels').with_suffix('.csv'))

        # Save optimization steps as JSON
        opt_steps_serialized = {k: v.to_dict()
                                for k, v in self.opt_steps.items()}
        with open((_path / 'steps').with_suffix('.json'), 'w') as f:
            json.dump(opt_steps_serialized, f, indent=4)

#                                                          OptimizationDynamics
# =============================================================================


def performance_profile(performance: xr.DataArray, max_f_value: int = 2.0
                        ) -> xr.DataArray:
    p = performance.isel(budget=-1)
    rij = (p + 1.0) / 1.0

    ds_f = xr.Dataset(
        {'performance_profile': rij, 'relative_error': xr.DataArray(
            np.logspace(np.log10(1.0), np.log10(max_f_value), 1000),
            dims='relative_error')})

    performance_profile = np.less_equal(
        ds_f, ds_f.relative_error+1e-2).where(
        ~ds_f.isnull(), np.nan).mean(dim='problem').performance_profile

    rel_error = (performance_profile.relative_error * 100) - 100

    return performance_profile.assign_coords(relative_error=rel_error)


def rtd(performance: xr.DataArray) -> xr.DataArray:
    return (1.0-performance).mean('problem')

# =============================================================================


class OptimizationDynamics:
    def __init__(self,
                 X_features: pd.DataFrame,
                 X_trajectory: Optional[np.ndarray],
                 Y: np.ndarray,
                 opt_steps: Dict[str, OptimizationStep],
                 previous_step: Optional[OptimizationStep]):

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
        opt_steps: Dict[str, OptimizationStep],
        previous_step: Optional[OptimizationStep]
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
            mask_X_trajectory = ~X_trajectory.isnull().any('budget').isel(
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
    def from_task_and_run(
            cls,
            task: Task,
            previous_step: Optional[OptimizationStep] = None,
            data: Optional[ExperimentData] = None,
    ) -> OptimizationDynamics:
        X_features = pd.DataFrame([task.tag])
        previous_step = previous_step

        if data is None:
            X_trajectory = None

        else:
            X_trajectory = np.vstack([es.output_data['y']
                                      for _, es in data]).T

        return cls(X_features=X_features,
                   X_trajectory=X_trajectory,
                   Y=None,
                   opt_steps=None,
                   previous_step=previous_step)

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
        data = np.load(str(_path) + '.npz', allow_pickle=True)
        if (data['X_trajectory'].ndim == 0
                and data['X_trajectory'].item() is None):
            X_trajectory = None
        else:
            X_trajectory = data['X_trajectory']

        features = pd.read_csv(str(_path) + '.csv', index_col=0,
                               header=0).convert_dtypes()

        # Open opt steps
        with open(str(_path) + '.json', 'r') as f:
            opt_steps_dict = json.load(f)

        previous_step = opt_steps_dict.pop('previous_step', None)

        if previous_step is not None:
            previous_step = OptimizationStep.from_dict(previous_step)

        opt_steps = {k: OptimizationStep.from_dict(
            v) for k, v in opt_steps_dict.items()}

        return cls(X_features=features,
                   X_trajectory=X_trajectory,
                   Y=data['Y'],
                   opt_steps=opt_steps,
                   previous_step=previous_step)

    def save(self, path: str) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        path : Path
            The path to the model
        """
        _path = Path(path)
        # Save trajectory and y
        np.savez(str(_path) + '.npz', **self.to_dict())

        # Save features
        self.X_features.to_csv(str(_path) + '.csv')

        opt_steps_dict = {k: v.to_dict()
                          for k, v in self.opt_steps.items()}

        if self.previous_step is not None:
            opt_steps_dict['previous_step'] = self.previous_step.to_dict()
        else:
            opt_steps_dict['previous_step'] = None

        with open(str(_path) + '.json', 'w') as f:
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


def squash_optimizer(data: xr.Dataset, target_optimizer: str) -> xr.Dataset:
    _data = data.sel(
        optimizer=[o for o in data.optimizer.values
                   if o.startswith(target_optimizer)])
    mask = _data.notnull().any(dim='budget').idxmax(
        dim='optimizer').isel(output_dim=0, drop=True).output_mean

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
    data : xarray.Dataset
        The optimization trajectory data.
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
        input_core_dims=[['budget', 'output_dim']],
        output_core_dims=[['budget', 'output_dim']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}  # False
    )

    return trajectory.output_mean.isel(
        output_dim=0, drop=True).rename(
            'trajectory').transpose(
                'optimizer', 'budget', 'problem')
