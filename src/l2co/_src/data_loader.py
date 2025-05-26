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
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Type

import jax
import jax.numpy as jnp
# Third-party
import numpy as np
import optax
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from scipy.linalg import svd
from sklearn.model_selection import train_test_split

# Local
from .latent_space import fit_basis_function, fit_gpr
from .optimizerschedule import OptimizationSchedule, OptimizationStep
from .performance import output_cummin, scaling2, scaling3
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
    """
    Data loader for optimization dynamics in the L2CO framework.

    Parameters
    ----------
    data : xarray.Dataset
        The raw optimization trajectories.
    task_labels : xarray.DataArray
        Labels of the optimization problems.
    opt_steps : Iterable[OptimizationSchedule]
        Optimization schedules.
    l2co_schedules : xarray.Dataset, optional
        L2CO schedules, by default None.

    Attributes
    ----------
    data : xarray.Dataset
        The raw optimization trajectories.
    task_labels : xarray.DataArray
        Labels of the optimization problems.
    opt_steps : Iterable[OptimizationSchedule]
        Optimization schedules.
    l2co_schedules : xarray.Dataset
        L2CO schedules.
    """

    def __init__(
        self,
        data: xr.Dataset,
        task_labels: xr.DataArray,
        opt_steps: Iterable[OptimizationSchedule],
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

        # Retrieve the optimization dynamics data
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

        task_labels = xr.apply_ufunc(
            np.vectorize(parse_mixed_value),
            data['task_labels'],
            input_core_dims=[["problem"]],
            output_core_dims=[["problem"]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[object],
        ).T

        # Opt steps
        opt_steps = [OptimizationSchedule.from_json_string(
            opt) for _, opt in data['opt_steps'].to_series().items()]

        # delete opt_step and task_labels from data
        data = data.drop_vars(
            ['opt_steps', 'task_labels']).drop_dims('features')

        return cls(
            data=data.sortby('optimizer'),
            task_labels=task_labels,
            opt_steps=sorted(opt_steps),
            l2co_schedules=l2co_schedules
        )

#                                                         Storing and exporting
# =============================================================================

    def to_all_params(self,
                      t: float, t_start: float = 0.0, t_end: float = 1.0
                      ) -> Tuple[xr.DataTree, Iterable[OptimizationStep]]:

        optimizers = self.select_single_optimizers().optimizers.values

        _dict = {optimizer: self.query_optimizer(optimizer).to_params(
            optimizer=optimizer,
            t=t, t_start=t_start, t_end=t_end) for optimizer in optimizers}
        _dict['first'] = self.select_single_optimizers(squash=True).to_params(
            t=t_start, t_start=t_start, t_end=t_end)

        opt_steps = [schedule.steps[0]
                     for schedule in self.select_single_optimizers().opt_steps]

        return xr.DataTree.from_dict(_dict), opt_steps

        # ds_switch = xr.concat(
        #     [
        #         self.query_optimizer(optimizer).to_params(
        #             optimizer=optimizer,
        #             t=t, t_start=t_start, t_end=t_end)
        #         for optimizer in optimizers
        #     ],
        #     dim='problem')

        # ds_first = self.select_single_optimizers(squash=True).to_params(
        #     t=t_start, t_start=t_start, t_end=t_end)
        # return (
        #     OptimizationDynamics2.from_data(ds_first),
        #     OptimizationDynamics2.from_data(ds_switch),
        # )

    def to_params(
        self,
        t: float,
        t_start: float = 0.0,
        t_end: float = 1.0,
        optimizer: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Converts the data into a parameterized dataset.

        Parameters
        ----------
        t : float
            Current time step.
        t_start : float, optional
            Start time for the parameterization. Default is 0.0.
        t_end : float, optional
            End time for the parameterization. Default is 1.0.
        optimizer : str, optional
            Name of the optimizer to include in the dataset.

        Returns
        -------
        xr.Dataset
            Parameterized dataset.
        """

        vars = {}

        vars['task_labels'] = self.task_labels.astype(str)
        # Calculate the performance labels
        vars['performance'] = self.select_t(
            t_start=t, t=t_end).performance_per_realization.sel(
            budget=t_end, method='nearest', drop=True)

        if optimizer is not None:
            # Fit the shape of the trajectory
            # params = compute_parameters(
            #     self.select_t(t_start=t_start, t=t).trajectory)
            # vars['shape'] = _squash(params, reduce_dim='optimizer')

            # staircase_params = compute_staircase_parameters(
            #     self.select_t(t_start=t_start, t=t).staircase)
            trajectory = self.select_t(t_start=t_start, t=t).trajectory
            vars['trajectory'] = _squash(trajectory, reduce_dim='optimizer')

            staircase = self.select_t(t_start=t_start, t=t).staircase
            vars['staircase'] = _squash(
                staircase, reduce_dim='optimizer')

            # Calculate the relative improvement of the window
            # lri = log_relative_improvement(self.data)
            # vars['improvement'] = _squash(lri, reduce_dim='optimizer')

            # # Create a tag of the optimizer used for the trajectory
            # vars['opt'] = xr.DataArray(
            #     np.full(len(self.data.itemID), optimizer, dtype=object),
            #     dims='problem',
            #     coords={"itemID": self.data.itemID}
            # )

        # Put everything together in a dataset
        ds = xr.Dataset(vars)

        ds['optimizer'] = ds['optimizer'].astype(str)

        # Transpose dims
        desired_order = ('problem', 'output_dim', 'optimizer')

        return ds.map(lambda var: var.transpose(
            *[d for d in desired_order if d in var.dims],
            *[d for d in var.dims if d not in desired_order])
        )

    def to_spreadsheet(self, criteria: Literal['min', 'mean'] = 'min'):
        ds = xr.Dataset({'task_labels': self.task_labels,
                         'output_min': self.data.output_min})

        if criteria == 'min':
            df = ds.groupby('itemID').min().min('budget').isel(
                output_dim=0)['output_min'].to_pandas()
        elif criteria == 'mean':
            df = ds.groupby('itemID').mean().min('budget').isel(
                output_dim=0)['output_min'].to_pandas()

        df.index = ds.groupby('itemID').min().min(
            'budget').isel(output_dim=0)['task_labels']

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

        df_task_labels = ds.groupby('itemID').min()['task_labels'].to_pandas()
        df_sbs = ds.groupby('itemID').min()['sbs'].to_pandas().rename('sbs')

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
        return self.data['budget'].size

    @property
    def n_realization(self) -> int:
        return np.unique(self.data['realization']).size

    @property
    def features(self) -> xr.DataArray:
        """
        Property method to return the features coordinate

        Returns
        -------
        xarray.DataArray
            The features coordinate.
        """
        return self.task_labels['features']

    @property
    def problems(self) -> xr.DataArray:
        """
        Property method to return the problem coordinate

        Returns
        -------
        xarray.DataArray
            The problem coordinate.
        """
        return self.task_labels['problem']

    @property
    def optimizers(self) -> xr.DataArray:
        """
        Property method to return the optimizers coordinate

        Returns
        -------
        xarray.DataArray
            The optimizers coordinate.
        """
        return self.data['optimizer']

    @property
    def staircase(self) -> xr.DataArray:
        return scaling3(output_cummin(
            self.data['output_min'])).ffill('budget')

    @property
    def trajectory(self) -> xr.DataArray:
        """
        Returns the trajectory of the optimization process.

        Returns
        -------
        xr.DataArray
            Trajectory data.
        """
        return scaling3(self.data['output_mean'])

    def _performance(self, include_best: bool,
                     exclude_optimizer: Optional[str] = None) -> xr.DataArray:
        """
        Computes the performance of the optimizers.

        Parameters
        ----------
        include_best : bool
            Whether to include the best optimizer in the performance.
        exclude_optimizer : str, optional
            Name of the optimizer to exclude from the performance.

        Returns
        -------
        xr.DataArray
            Performance data.
        """
        t_start = perf_counter()
        print("Triggered performance")

        performance = scaling2(output_cummin(
            self.data['output_min'])).ffill('budget')

        if include_best:
            sbs = self._calculate_sbs(performance=performance,
                                      exclude_optimizer=exclude_optimizer)
            perf_sbs = performance.sel(optimizer=sbs, drop=True).reset_coords(
                'optimizer', drop=True).expand_dims(optimizer=['sbs'])
            performance = xr.merge([performance, perf_sbs])['output_min']

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

    @property
    def performance_per_realization(self) -> xr.DataArray:

        t_start = perf_counter()
        print("Triggered performance")
        performance = scaling3(output_cummin(
            self.data['output_min'])).ffill('budget')
        # performance = scaling3(self.data).ffill('budget')
        print(f"Performance calculation took: {perf_counter()-t_start:0.2f}")
        return performance

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

        # Check if there are budget coordinate values below t_start
        prior = self.data['budget'].where(
            self.data['budget'] < t_start, drop=True)

        # If so, take the value before t_start
        if prior.sizes['budget'] > 0:
            t_prior = prior[-1].item()

        # If not, do nothing
        else:
            t_prior = t_start

        return DataLoader(
            task_labels=self.task_labels,
            data=self.data.sel(budget=slice(t_prior, t)),
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

        opt_steps = list(filter(lambda item: item.name in
                                optimizers, self.opt_steps))

        return DataLoader(
            task_labels=self.task_labels,
            data=data.sortby('optimizer'),
            opt_steps=sorted(opt_steps),
            l2co_schedules=self.l2co_schedules
        )

    def query_optimizer(self, optimizer: str) -> DataLoader:
        """
        Filters the data for a specific optimizer.

        Parameters
        ----------
        optimizer : str
            Name of the optimizer to filter.

        Returns
        -------
        DataLoader
            A new DataLoader instance with filtered data.
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
        filtered_schedules = list(
            filter(lambda item: item.steps[0].name in optimizer,
                   self.opt_steps))

        new_schedules: Iterable[OptimizationSchedule] = []
        for v in filtered_schedules:
            name = v.name.split('+')[-1] if '+' in v.name else v.name
            v_copy = v.copy()
            v_copy.name = name
            v_copy.steps = [v_copy.steps[-1]]
            if name == optimizer:
                v_copy.steps[0].x0_strategy = 'last'
            new_schedules.append(v_copy)

        return DataLoader(
            task_labels=self.task_labels,
            data=data_selected.sortby('optimizer'),
            opt_steps=sorted(new_schedules),
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
    """
    Create a performance profile for the given performance data.

    Parameters
    ----------
    performance : xarray.DataArray
        Performance data for optimization problems.
    max_f_value : int, optional
        Maximum performance ratio, by default 2.0.

    Returns
    -------
    xarray.DataArray
        Performance profile.
    """
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
    """
    Compute the runtime distribution (RTD) profile.

    Parameters
    ----------
    performance : xarray.DataArray
        Performance data for optimization problems.

    Returns
    -------
    xarray.DataArray
        RTD profile.
    """
    return (1.0-performance).mean('problem')

# =============================================================================


class OptimizationDynamics:
    def __init__(self,
                 data: xr.Dataset,
                 previous_step: Optional[str]):
        self.data = data
        self.previous_step = previous_step

    @classmethod
    def from_data(cls: Type[OptimizationDynamics],
                  data: xr.Dataset,
                  previous_step: Optional[str] = None) -> OptimizationDynamics:
        # Fill in missing performance values
        data['performance'] = xr.apply_ufunc(
            impute_missing_values_svd,
            data['performance'],
            input_core_dims=[['problem', 'optimizer']],
            output_core_dims=[['problem', 'optimizer']],
            vectorize=True,
        )

        # Check for NaNs in each variable and reduce over all dims
        # except 'problem'
        nan_flags = [
            var.isnull().any(dim=set(var.dims) - {'problem'})
            for var in data.data_vars.values()
        ]

        # Combine all flags: True where any variable has NaNs in that problem
        has_nan_anywhere = xr.concat(nan_flags, dim='_vars').any(dim='_vars')

        # Filter out nans
        data = data.sel(problem=~has_nan_anywhere)

        # Select first output_dim
        data = data.isel(output_dim=0, drop=True)

        if previous_step == 'first':
            previous_step = None

        return cls(data=data, previous_step=previous_step)

    @property
    def X_features(self) -> pd.DataFrame:
        return self.data['task_labels'].to_pandas(
        ).infer_objects().convert_dtypes()

    @property
    def X_trajectory(self) -> jax.Array:
        return jnp.stack(
            (self.data['trajectory'].values, self.data['staircase'].values),
            axis=-1)

    @property
    def Y(self) -> jax.Array:
        return jnp.array(self.data['performance'])

    @classmethod
    def from_task_and_run(
            cls: Type[OptimizationDynamics],
            task: Task,
            data: Optional[ExperimentData] = None,
            previous_step: Optional[str] = None,
    ) -> OptimizationDynamics:
        ...


def load_dynamics(path: str | Path | xr.DataTree
                  ) -> Dict[str, OptimizationDynamics]:
    if not isinstance(path, xr.DataTree):
        _path = Path(path)
        datatree = xr.open_datatree(_path.with_suffix('.nc'))
    else:  # Assume path is already a DataTree
        datatree = path

    return {name: OptimizationDynamics.from_data(
        data=dynamics.to_dataset(),
        previous_step=name) for name, dynamics in datatree.items()}


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
def compute_X_trajectory(data: xr.DataArray) -> xr.DataArray:
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
        # input_core_dims=[['budget', 'output_dim']],
        input_core_dims=[['budget']],
        # output_core_dims=[['budget', 'output_dim']],
        output_core_dims=[['budget']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}  # False
    )

    # return trajectory.output_mean.isel(
    #     output_dim=0, drop=True).rename(
    #         'trajectory').transpose(
    #             'optimizer', 'budget', 'problem')

    return trajectory


def compute_parameters(
        trajectory: xr.DataArray,
        optimizer: optax.GradientTransformation = optax.adam(1e-01),
        inducing_points: int = 10, num_iters: int = 300) -> xr.DataArray:
    """
    Compute the parameters of the optimization trajectory.

    Parameters
    ----------
    data : xarray.Dataset
        The optimization trajectory data.
    Returns
    -------
    xarray.DataArray
        The selected trajectory.
    """
    x = jnp.array(trajectory.values.reshape(-1, trajectory.budget.size))

    params = jax.vmap(
        partial(fit_gpr, optimizer=optimizer,
                inducing_points=inducing_points, num_iters=num_iters)
    )(x)

    params = params.reshape(
        trajectory.optimizer.size, trajectory.problem.size,
        trajectory.output_dim.size, -1
    )

    return xr.DataArray(
        params,
        dims=['optimizer', 'problem', 'output_dim', 'fit_params'],
        coords={
            'optimizer': trajectory.optimizer,
            'itemID': trajectory.itemID,
            'output_dim': trajectory.output_dim,
        }
    )


def compute_staircase_parameters(
        staircase: xr.DataArray,
        num_iters: int = 50) -> xr.DataArray:
    x = jnp.array(staircase.values.reshape(-1, staircase.budget.size))

    params = jax.vmap(
        partial(fit_basis_function, max_iter=num_iters)
    )(x)

    params = params.reshape(
        staircase.optimizer.size, staircase.problem.size,
        staircase.output_dim.size, -1
    )

    return xr.DataArray(
        params,
        dims=['optimizer', 'problem', 'output_dim', 'staircase_fit_params'],
        coords={
            'optimizer': staircase.optimizer,
            'itemID': staircase.itemID,
            'output_dim': staircase.output_dim,
        }
    )


def log_relative_improvement(data: xr.Dataset) -> xr.DataArray:
    def compute_log_relative_improvement(x, b):
        impv = -(x[0] - x[-1]) / (b[0] - b[-1]) / (abs(x[0]) + 1e-8)
        t = np.log(impv + 1e-8)
        return np.atleast_1d(t)

    return xr.apply_ufunc(
        compute_log_relative_improvement,
        output_cummin(data.output_min),
        data.budget,
        input_core_dims=[["budget"], ["budget"]],
        # output_core_dims=[["log_relative_improvement"]],
        vectorize=True,
    )


def _squash(da: xr.DataArray, reduce_dim: str,
            vector_dim: str = 'problem'):
    def f(x):
        for val in x:
            if not np.isnan(val).any():
                return val
        return val

    # Define dimension names
    all_dims = list(da.dims)
    keep_dims = [d for d in all_dims if d not in [reduce_dim, vector_dim]]

    return xr.apply_ufunc(
        f,
        da,
        input_core_dims=[[reduce_dim] + keep_dims],
        output_core_dims=[keep_dims],
        vectorize=True,
        dask="allowed"  # Optional: if using dask
    )


def compute_fitting(trajectory: xr.DataArray, fit_params: xr.DataArray,
                    basis_function: Callable):

    x_input = np.linspace(0., 1., trajectory.coords['budget'].size)

    fitted_sigmoid = xr.apply_ufunc(
        lambda x, x_input: basis_function(x_input, *x),
        fit_params,
        kwargs={'x_input': x_input},
        input_core_dims=[['fit_params']],
        output_core_dims=[['budget']],
        vectorize=True,
    )

    fitted_sigmoid = fitted_sigmoid.assign_coords(
        budget=trajectory.coords['budget'])
