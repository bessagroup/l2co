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

import hashlib
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

# Third-party
import jax
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import svd
from sklearn.model_selection import train_test_split

# Local
from .naming import (
    BUDGET,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OPT_STEPS,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    PROBLEM,
    REALIZATION,
    SCHEDULE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)
from .optimizationdynamics import OptimizationDynamics
from .performance import (
    ertd,
    output_cummin,
    performance_profile,
    rtd,
    scale_by_problem_and_schedule,
    scale_by_task,
)
from .task_features import FeatureProcessor
from .tree_utils import traj_and_stair

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                                     Constants
# =============================================================================

INVALID_PERFORMANCE_VALUE = 1.0
PADDING_VALUE = 1.0
DEFAULT_COST_VALUES = np.logspace(-1, 6, 100) ** 0.8

#                                                                   Data Loader
# =============================================================================


@dataclass
class DataLoader:
    """
    Data loader for optimization dynamics in the L2CO framework.

    Parameters
    ----------
    data : xarray.Dataset
        The raw optimization trajectories.
    task_labels : xarray.DataArray
        Labels of the optimization problems.

    Attributes
    ----------
    data : xarray.Dataset
        The raw optimization trajectories.
    task_labels : xarray.DataArray
        Labels of the optimization problems.
    """

    data: xr.Dataset
    task_labels: xr.DataArray

    def __post_init__(self):
        self.data = self.data.sortby(TASK_ID)
        self.task_labels = self.task_labels.sortby(TASK_ID)

    def __add__(self, __o: DataLoader) -> DataLoader:
        """
        Concatenate two DataLoader objects.

        Parameters
        ----------
        __o : DataLoader
            DataLoader to concatenate.

        Returns
        -------
        DataLoader
            Concatenated DataLoader.
        """
        if __o is None:
            return self

        return replace(
            self,
            data=xr.concat([self.data, __o.data], dim=PROBLEM),
            task_labels=xr.concat(
                [self.task_labels, __o.task_labels], dim=PROBLEM
            ),
        )

    #                                                  Alternative Constructors
    # =========================================================================

    @classmethod
    def from_xarray(
        cls,
        data: xr.Dataset,
    ):
        """
        Create a DataLoader from an xarray.Dataset.

        Parameters
        ----------
        data : xarray.Dataset
            Dataset containing optimization trajectories and labels.

        Returns
        -------
        DataLoader
            Instantiated DataLoader object.
        """

        def parse_mixed_value(val: str):
            """Try to convert a string into int, float, or leave as string."""
            if val.lower() == "true":
                return True
            elif val.lower() == "false":
                return False
            elif val == "None":
                return None
            try:
                if "." in val or "e" in val.lower():
                    return float(val)
                return int(val)
            except ValueError:
                return val

        task_labels = xr.apply_ufunc(
            np.vectorize(parse_mixed_value),
            data[TASK_LABELS],
            input_core_dims=[[PROBLEM]],
            output_core_dims=[[PROBLEM]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[object],
        ).T

        # Remove variable features from data
        data = data.drop_vars([TASK_LABELS, FEATURE])

        # Remove unnecessary coordinates from data

        return cls(
            data=data,
            task_labels=task_labels,
        )

    @classmethod
    def from_file(
        cls, path: str, data_filename: str = "entire_dataset"
    ) -> DataLoader:
        """
        Create a DataLoader from a file path.

        Parameters
        ----------
        path : str
            Path to the dataset file or directory.
        data_filename : str, optional
            Name of the dataset file, by default "entire_dataset".

        Returns
        -------
        DataLoader
            Instantiated DataLoader object.
        """
        # Create path object
        _path = Path(path)

        # Retrieve the optimization dynamics data
        if (_path / data_filename).with_suffix(".odata").exists():
            data = xr.open_dataset(
                (_path / data_filename).with_suffix(".odata")
            )
        else:
            data = xr.open_dataset(_path)
        return cls.from_xarray(data=data)

    #                                                                 Exporting
    # =========================================================================

    def to_params(
        self,
        t: float,
        t_end: int,
        schema: dict[str, str],
        cost_metric: Literal["fevals", "iterations"] = "fevals",
    ):
        """
        Return task parameters and labels for a given budget.

        Parameters
        ----------
        t : float
            Budget value.
        t_end : int
            End budget value.
        schema : dict[str, str]
            Schema for feature processing.
        cost_metric : {'fevals', 'iterations'}, optional
            Cost metric to use, by default 'fevals'.

        Returns
        -------
        tuple
            (task parameters, labels)
        """

        # TASK FEATURES
        dict_features = self.task_labels.transpose(PROBLEM, FEATURE)

        tags_list = [
            {
                str(k): v
                for k, v in zip(
                    self.task_labels[FEATURE].values, xi.values, strict=True
                )
            }
            for xi in self.task_labels
        ]

        feature_processor_task = FeatureProcessor.init(
            schema=schema, tags=tags_list
        )

        x_features = np.stack(
            [feature_processor_task.transform(xi) for xi in tags_list]
        )

        x_features = xr.DataArray(
            x_features,
            coords={
                PROBLEM: self.problems,
                FEATURE: list(schema.keys()),
            },
            dims=dict_features.dims,
        )

        x_features = x_features.assign_coords({PROBLEM: self.problems}).sortby(
            TASK_ID
        )

        x_features = x_features.drop_vars(PROBLEM)

        # OPTIMIZER FEATURES
        # tags = [{"optimizer": str(v)} for v in self.optimizers.values]
        # feature_processor_optimizer = FeatureProcessor.init(
        #     schema={"optimizer": "categorical"}, tags=tags
        # )

        # OPTIMIZATION TRAJECTORY

        x_history = (
            self.select_t(t=t)
            .to_traj_and_stair()
            .sortby(TASK_ID)
            .transpose(PROBLEM, BUDGET, SCHEDULE, "kind")
        )

        # Add one-hot encoded schedule to x_history
        n_schedules = x_history.sizes[SCHEDULE]
        one_hot = np.eye(n_schedules)
        new_kind_coords = [f"schedule_{i}" for i in range(n_schedules)]
        one_hot_da = xr.DataArray(
            one_hot,
            dims=[SCHEDULE, "kind"],
            coords={SCHEDULE: x_history[SCHEDULE], "kind": new_kind_coords},
        )
        # Broadcast to match x_history dimensions (except kind)
        one_hot_expanded = one_hot_da.broadcast_like(
            x_history.isel(kind=0, drop=True)
        )
        x_history = xr.concat([x_history, one_hot_expanded], dim="kind")

        y = (
            self.to_labels(t=t_end, cost_metric=cost_metric)
            .sortby(TASK_ID)
            .transpose(PROBLEM, SCHEDULE)
        )

        return OptimizationDynamics(
            x_features=x_features,
            x_history=x_history,
            y=y,
            feature_processor=feature_processor_task,
        )

    def to_traj_and_stair(
        self,
    ) -> xr.DataArray:
        x_min = self.data[OUTPUT_MIN].stack(
            {"batch": (SCHEDULE, OUTPUT_DIM, PROBLEM)}
        )
        x_mean = self.data[OUTPUT_MEAN].stack(
            {"batch": (SCHEDULE, OUTPUT_DIM, PROBLEM)}
        )

        out = jax.vmap(traj_and_stair)(x_mean.values.T, x_min.values.T)

        y = xr.DataArray(
            out,
            dims=["batch", BUDGET, "kind"],
            coords={
                BUDGET: self.data[BUDGET],
                "batch": x_min["batch"],
                "kind": ["trajectory", "staircase"],
            },
        )
        y = y.unstack("batch")
        return (
            y.assign_coords({PROBLEM: self.data[PROBLEM]})
            .reset_index(PROBLEM, drop=True)
            .isel({OUTPUT_DIM: 0}, drop=True)
        )

    def to_labels(
        self, t: int, cost_metric: Literal["fevals", "iterations"] = "fevals"
    ):
        """
        Return labels for a given budget and cost metric.

        Parameters
        ----------
        t : int
            Budget value.
        cost_metric : {'fevals', 'iterations'}, optional
            Cost metric to use, by default 'fevals'.

        Returns
        -------
        xarray.DataArray
            Labels array.
        """
        _ertd = self.ertd_profile(
            include_best=False,
            cost_values=np.array([t]),
        )

        return _ertd.sel({"metric": cost_metric}).mean("cost_per_dim")

    def to_spreadsheet(self, criteria: Literal["min", "mean"] = "min"):
        """
        Generate a spreadsheet summarizing optimization results.

        Parameters
        ----------
        criteria : {'min', 'mean'}, optional
            Criteria for summarizing results, by default 'min'.

        Returns
        -------
        pd.DataFrame
            Styled DataFrame for visualization.
        """

        ds = xr.Dataset(
            {TASK_LABELS: self.task_labels, OUTPUT_MIN: self.data[OUTPUT_MIN]}
        )

        if criteria == "min":
            df = (
                ds.groupby(TASK_ID)
                .min()
                .min(BUDGET)
                .isel({OUTPUT_DIM: 0})[OUTPUT_MIN]
                .to_pandas()
            )
        elif criteria == "mean":
            df = (
                ds.groupby(TASK_ID)
                .mean()
                .min(BUDGET)
                .isel({OUTPUT_DIM: 0})[OUTPUT_MIN]
                .to_pandas()
            )

        df.index = (
            ds.groupby(TASK_ID)
            .min()
            .min(BUDGET)
            .isel({OUTPUT_DIM: 0})[TASK_LABELS]
        )

        # Function to highlight the smallest value per row
        def highlight_min_row(row):
            min_val = row.min()
            return [
                "font-weight: bold" if val == min_val else "" for val in row
            ]

        # Apply styling and scientific notation formatting
        styled_df = (
            df.style
            # Highlight smallest value per row
            .apply(highlight_min_row, axis=1).format(
                "{:.2e}"
            )  # Display in scientific notation
        )

        return styled_df

    def display_sbs_table(
        self, exclude_optimizer: str | Iterable[str]
    ) -> pd.DataFrame:
        """
        Display a table of SBS (Single Best Solver) choices.

        Parameters
        ----------
        exclude_optimizer : str or Iterable[str]
            Optimizers to exclude from the SBS calculation.

        Returns
        -------
        pd.DataFrame
            DataFrame containing SBS choices.
        """

        sbs = calculate_sbs(
            performance=self.performance, exclude_optimizer=exclude_optimizer
        )
        ds = xr.Dataset(
            {
                "sbs": sbs,
                TASK_LABELS: self.task_labels,
            }
        )

        df_task_labels = ds.groupby(TASK_ID).min()[TASK_LABELS].to_pandas()
        df_sbs = ds.groupby(TASK_ID).min()["sbs"].to_pandas().rename("sbs")

        return pd.concat([df_task_labels, df_sbs], axis=1)

    #                                                                Properties
    # =========================================================================

    @property
    def hash_opt_step(self) -> xr.DataArray:
        """
        Returns a hash for each optimization step.

        Returns
        -------
        xarray.DataArray
            Hash values for optimization steps.
        """
        return xr.apply_ufunc(
            lambda x: int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)
            % (2**31 - 1),
            self.data[UPDATE_STEP],
            input_core_dims=[[BUDGET]],
            output_core_dims=[[]],
            vectorize=True,
        )

    @property
    def n_iterations(self) -> int:
        """
        Returns the number of iterations (budget size).

        Returns
        -------
        int
            Number of iterations.
        """
        return self.data[BUDGET].size

    @property
    def n_realization(self) -> int:
        """
        Returns the number of unique realizations.

        Returns
        -------
        int
            Number of realizations.
        """
        return np.unique(self.data[REALIZATION]).size

    @property
    def features(self) -> xr.DataArray:
        """
        Returns the features coordinate.

        Returns
        -------
        xarray.DataArray
            The features coordinate."""
        """
        Property method to return the features coordinate

        Returns
        -------
        xarray.DataArray
            The features coordinate.
        """
        return self.task_labels[FEATURE]

    @property
    def problems(self) -> xr.DataArray:
        """
        Returns the problem coordinate.

        Returns
        -------
        xarray.DataArray
            The problem coordinate.
        """
        return self.task_labels[PROBLEM]

    @property
    def optimizers(self) -> xr.DataArray:
        """
        Property method to return the optimizers coordinate

        Returns
        -------
        xarray.DataArray
            The optimizers coordinate.
        """
        return self.data[SCHEDULE]

    @property
    def valid_files(self) -> xr.DataArray:
        """
        Returns a boolean array indicating valid files. It has coordinates
        of PROBLEM, SCHEDULE and OUTPUT_DIM.

        Returns
        -------
        xarray.DataArray
            Boolean array indicating valid files.
        """
        return ~self.data[OUTPUT_MIN].isnull().all(BUDGET)

    @property
    def performance(self) -> xr.DataArray:
        """
        Returns the performance data for optimizers.

        Returns
        -------
        xarray.DataArray
            Performance data.
        """
        return self._performance(include_best=False, exclude_optimizer=None)

    @property
    def performance_per_realization(self) -> xr.DataArray:
        """
        Returns the performance data per realization.

        Returns
        -------
        xarray.DataArray
            Performance data per realization.
        """
        t_start = perf_counter()
        print("Triggered performance")
        performance = scale_by_problem_and_schedule(
            output_cummin(self.data[OUTPUT_MIN])
        ).ffill(BUDGET)
        # performance = scaling3(self.data).ffill(BUDGET)
        print(f"Performance calculation took: {perf_counter() - t_start:0.2f}")
        return performance

    #                                                        Selecting problems
    # =========================================================================

    def select_itemIDs(self, itemIDs: str | Iterable[str]) -> DataLoader:
        """
        Select tasks by their item IDs.

        Parameters
        ----------
        itemIDs : str or Iterable[str]
            Item IDs to select.

        Returns
        -------
        DataLoader
            DataLoader containing selected tasks.
        """

        if len(itemIDs) == 0:
            return deepcopy(self)

        if isinstance(itemIDs, str):
            itemIDs = [itemIDs]

        task_labels = xr.concat(
            [
                group
                for i, group in self.task_labels.groupby(TASK_ID)
                if i in itemIDs
            ],
            dim=PROBLEM,
        )
        data = xr.concat(
            [group for i, group in self.data.groupby(TASK_ID) if i in itemIDs],
            dim=PROBLEM,
        )

        return replace(self, data=data, task_labels=task_labels)

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
        _problem_indices = problem_indices.sortby(TASK_ID)

        return DataLoader(
            task_labels=self.task_labels.sel({PROBLEM: _problem_indices}),
            data=self.data.sel({PROBLEM: _problem_indices}),
        )

    def select_feature(self, feature: dict) -> DataLoader:
        mask = None
        for feature_name, feature_value in feature.items():
            # extract the 1D array for this feature
            feature_data = self.task_labels.sel(
                feature=feature_name, drop=True
            )
            if isinstance(feature_value, list | tuple | set):
                current_mask = False
                for val in feature_value:
                    current_mask |= feature_data == val
            else:
                current_mask = feature_data == feature_value
            mask = current_mask if mask is None else (mask & current_mask)

        problems = feature_data[PROBLEM].where(mask, drop=True).astype(int)
        return self.select_problems(problems)

    def create_strategy(self, choices: xr.DataArray, name: str) -> DataLoader:
        strategy = self.data.sel({SCHEDULE: choices}).drop_vars(SCHEDULE)
        strategy[OPT_STEPS] = strategy[OPT_STEPS].isel({PROBLEM: 0}, drop=True)
        strategy = strategy.expand_dims({SCHEDULE: [name]})

        data = xr.concat([self.data, strategy], dim=SCHEDULE)

        return replace(self, data=data)

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

        return replace(
            self,
            data=self.data.isel({PROBLEM: problems}),
            task_labels=self.task_labels.isel({PROBLEM: problems}),
        )

    def train_test_split(
        self, test_size: float = 0.2, random_state: int | None = None
    ) -> tuple[DataLoader, DataLoader]:
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
        (
            _X_train_indices,
            _X_test_indices,
        ) = train_test_split(
            range(len(self.problems)),
            test_size=test_size,
            random_state=random_state,
        )

        return (self.isel(_X_train_indices), self.isel(_X_test_indices))

    #                                                          Selecting budget
    # =========================================================================

    def select_t(
        self,
        t: int,
        t_start: float | None = 0.0,
        select_one_before: bool = True,
    ) -> DataLoader:
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
        if select_one_before:
            # Check if there are budget coordinate values below t_start
            prior = self.data[BUDGET].where(
                self.data[BUDGET] < t_start, drop=True
            )

            # If so, take the value before t_start
            if prior.sizes[BUDGET] > 0:
                t_prior = prior[-1].item()

            # If not, do nothing
            else:
                t_prior = t_start
        else:
            t_prior = t_start

        return replace(
            self,
            data=self.data.sel({BUDGET: slice(t_prior, t)}),
        )

    #                                                      Selecting optimizers
    # =========================================================================

    def select_optimizer(
        self, optimizers: Iterable[str] | str, squash: bool = False
    ) -> DataLoader:
        """
        Select specific optimizers.

        Parameters
        ----------
        optimizers : Iterable[str] or str
            Optimizers to select.
        squash : bool, optional
            Whether to squash multiple optimizers into one, by default False.

        Returns
        -------
        DataLoader
            DataLoader containing selected optimizers.
        """
        """
        Select specific optimizers.

        Parameters
        ----------
        optimizers : Iterable[str] or str
            Optimizers to select.
        squash : bool, optional
            Whether to squash multiple optimizers into one, by default False.

        Returns
        -------
        DataLoader
            DataLoader containing selected optimizers.
        """
        if isinstance(optimizers, str):
            optimizers = [optimizers]

        if squash:
            data = xr.concat(
                [
                    squash_optimizer(data=self.data, target_optimizer=opt)
                    for opt in optimizers
                ],
                dim=SCHEDULE,
            )
        else:
            data = self.data.sel({SCHEDULE: optimizers})

        return replace(
            self,
            data=data.sortby(SCHEDULE),
        )

    #                                                       Performance metrics
    # =========================================================================

    def _performance(
        self, include_best: bool, exclude_optimizer: str | None = None
    ) -> xr.Dataset:
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

        # performance = scale_by_task(
        #     output_cummin(self.data[OUTPUT_MIN])).ffill(
        #     BUDGET
        # )

        performance = xr.Dataset(
            {
                OUTPUT_MIN: scale_by_task(
                    data=output_cummin(self.data[OUTPUT_MIN]),
                    global_min=self.task_labels.sel(
                        {FEATURE: GLOBAL_MIN}
                    ).astype(float),
                ).ffill(BUDGET),
                FEVALS: self.data[FEVALS],
                ITERATIONS: self.data[ITERATIONS],
            }
        )

        if include_best:
            sbs = calculate_sbs(
                performance=performance, exclude_optimizer=exclude_optimizer
            )
            perf_sbs = (
                performance.sel({SCHEDULE: sbs}, drop=True)
                .reset_coords(SCHEDULE, drop=True)
                .expand_dims({SCHEDULE: ["sbs"]})
            )

            vbs = calculate_vbs(
                performance=performance, exclude_optimizer=exclude_optimizer
            )

            perf_vbs = (
                performance.sel({SCHEDULE: vbs}, drop=True)
                .reset_coords(SCHEDULE, drop=True)
                .expand_dims({SCHEDULE: ["vbs"]})
            )
            performance = xr.merge([performance, perf_sbs, perf_vbs])

        print(f"Performance calculation took: {perf_counter() - t_start:0.2f}")
        return performance

    def performance_profile(
        self,
        max_f_value: int = 2.0,
        include_best: bool = False,
        exclude_optimizer: str | None = None,
    ) -> xr.DataArray:
        """
        Create a performance profile for the given performance data.

        Parameters
        ----------
        max_f_value : int, optional
            Maximum performance ratio, by default 2.0.
        include_best : bool, optional
            Whether to include the best optimizer, by default False.
        exclude_optimizer : str, optional
            Optimizer to exclude, by default None.

        Returns
        -------
        xr.DataArray
            Performance profile.
        """

        # Trigger performance
        p = self._performance(
            include_best=include_best, exclude_optimizer=exclude_optimizer
        )

        return performance_profile(performance=p, max_f_value=max_f_value)

    def rtd_profile(
        self,
        include_best: bool = False,
        exclude_optimizer: str | None = None,
    ) -> xr.DataArray:
        """
        Compute the runtime distribution (RTD) profile.

        Parameters
        ----------
        include_best : bool, optional
            Whether to include the best optimizer, by default False.
        exclude_optimizer : str, optional
            Optimizer to exclude, by default None.

        Returns
        -------
        xr.DataArray
            RTD profile.
        """

        p = self._performance(
            include_best=include_best, exclude_optimizer=exclude_optimizer
        )

        return rtd(performance=p)

    def ertd_profile(
        self,
        include_best: bool = False,
        exclude_optimizer: str | None = None,
        cost_values: np.ndarray = DEFAULT_COST_VALUES,
    ) -> xr.DataArray:
        """
        Compute the expected runtime distribution (ERTD) profile.

        Parameters
        ----------
        include_best : bool, optional
            Whether to include the best optimizer, by default False.
        exclude_optimizer : str, optional
            Optimizer to exclude, by default None.

        Returns
        -------
        xr.DataArray
            ERTD profile.
        """
        # TODO : include best
        # p = self._performance(
        #     include_best=include_best, exclude_optimizer=exclude_optimizer
        # )
        dims = self.task_labels.sel(
            {FEATURE: "dimensionality"}, drop=True
        ).astype(int)

        _ertd = ertd(
            performance=self.data,
            dims=dims,
            cost_values=cost_values,
        )

        # 1. Get the base mask from valid_files
        mask = self.valid_files.isel({OUTPUT_DIM: 0}, drop=True)

        # 2. Reindex the mask to include 'vbs' and 'sbs' from _ertd
        # fill_value=True ensures that any schedule NOT in self.data
        # (like vbs/sbs) defaults to True.
        mask = mask.reindex_like(_ertd[SCHEDULE], method=None, fill_value=True)

        # 3. Apply the expanded mask
        _ertd = _ertd.where(mask)

        return _ertd

    def load(self):
        """
        Load the data into memory.
        """
        self.data.load()


def calculate_sbs(
    performance: xr.Dataset, exclude_optimizer: str | Iterable[str]
) -> xr.DataArray:
    """
    Calculate the Single Best Solver (SBS) for each task.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data.
    exclude_optimizer : str or Iterable[str]
        Optimizers to exclude.

    Returns
    -------
    xr.DataArray
        SBS choices for each task.
    """
    return calculate_best_solver(
        performance=performance,
        exclude_optimizer=exclude_optimizer,
        group=TASK_ID,
    )


def calculate_vbs(
    performance: xr.Dataset, exclude_optimizer: str | Iterable[str]
) -> xr.DataArray:
    """
    Calculate the Virtual Best Solver (VBS) for each problem.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data.
    exclude_optimizer : str or Iterable[str]
        Optimizers to exclude.

    Returns
    -------
    xr.DataArray
        VBS choices for each problem.
    """
    return calculate_best_solver(
        performance=performance,
        exclude_optimizer=exclude_optimizer,
        group=PROBLEM,
    )


def calculate_best_solver(
    performance: xr.Dataset,
    exclude_optimizer: str | Iterable[str],
    group: str,
) -> xr.DataArray:
    """
    Calculate the best solver for a given group.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data.
    exclude_optimizer : str or Iterable[str]
        Optimizers to exclude.
    group : str
        Grouping dimension (e.g., TASK_ID or PROBLEM).

    Returns
    -------
    xr.DataArray
        Best solver choices for the group.
    """
    if exclude_optimizer is not None:
        if isinstance(exclude_optimizer, str):
            exclude_optimizer = [exclude_optimizer]

        check_sbs = performance.where(
            ~performance[SCHEDULE].isin(exclude_optimizer), drop=True
        )
    else:
        check_sbs = performance

    best = (
        check_sbs[OUTPUT_MIN]
        .groupby(group)
        .map(
            lambda x: x.mean(PROBLEM)
            .isel({BUDGET: -1}, drop=True)
            .idxmin(SCHEDULE)
            .expand_dims({PROBLEM: x[PROBLEM]})
        )
        .isel({OUTPUT_DIM: 0}, drop=True)
        .assign_coords({PROBLEM: performance[PROBLEM]})
    )
    best = best.fillna(performance[SCHEDULE].values[0])
    return best


#                                                          OptimizationDynamics
# =============================================================================
def load_dynamics(
    path: str | Path | xr.DataTree,
) -> dict[str, OptimizationDynamics]:
    """
    Load optimization dynamics from a file or DataTree.

    Parameters
    ----------
    path : str, Path, or xr.DataTree
        Path to the file or DataTree object.

    Returns
    -------
    dict[str, OptimizationDynamics]
        Dictionary of OptimizationDynamics objects.
    """
    if not isinstance(path, xr.DataTree):
        _path = Path(path)
        datatree = xr.open_datatree(_path.with_suffix(".nc"))
    else:  # Assume path is already a DataTree
        datatree = path

    return {
        name: OptimizationDynamics.from_data(
            data=dynamics.to_dataset(), previous_step=name
        )
        for name, dynamics in datatree.items()
    }


# =============================================================================


def squash_optimizer(data: xr.Dataset, target_optimizer: str) -> xr.Dataset:
    """
    Squash multiple optimizer runs into a single run for the target optimizer.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing optimizer runs.
    target_optimizer : str
        Optimizer name to squash.

    Returns
    -------
    xr.Dataset
        Squashed dataset for the target optimizer.
    """
    _data = data.sel(
        {
            SCHEDULE: [
                o
                for o in data.optimizer.values
                if o.startswith(target_optimizer)
            ]
        }
    )
    mask = (
        _data.notnull()
        .any(dim=BUDGET)
        .idxmax(dim=SCHEDULE)
        .isel({OUTPUT_DIM: 0}, drop=True)
        .output_mean
    )

    return (
        _data.sel({SCHEDULE: mask})
        .reset_coords(SCHEDULE, drop=True)
        .expand_dims({SCHEDULE: np.array([target_optimizer], dtype=object)})
    )


def impute_missing_values_svd(y: np.ndarray, k=2):
    """
    Impute missing values in a matrix using SVD.

    Parameters
    ----------
    y : np.ndarray
        Input matrix with missing values (NaNs).
    k : int, optional
        Number of SVD components to use, by default 2.

    Returns
    -------
    np.ndarray
        Matrix with imputed values.
    """
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
        p = x - x.min()
        # 4. Add epsilon to make quality indicator value non-zero
        eps = np.where(p > 0.0, p, p.max() + 1e-16).min()
        x_log10 = np.log10(p + eps)

        # 5. Min-max scaling of the data
        x_min = x_log10.min()
        x_max = x_log10.max()
        return (x_log10 - x_min) / (x_max - x_min + 1e-16)

    trajectory: xr.Dataset = xr.apply_ufunc(
        scale,
        data,
        # input_core_dims=[[BUDGET, 'output_dim']],
        input_core_dims=[[BUDGET]],
        # output_core_dims=[[BUDGET, 'output_dim']],
        output_core_dims=[[BUDGET]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},  # False
    )

    # return trajectory.output_mean.isel(
    #     output_dim=0, drop=True).rename(
    #         'trajectory').transpose(
    #             'optimizer', BUDGET, 'problem')

    return trajectory


def log_relative_improvement(data: xr.Dataset) -> xr.DataArray:
    """
    Compute the log relative improvement of an optimization trajectory.

    Parameters
    ----------
    data : xr.Dataset
        Optimization trajectory data.

    Returns
    -------
    xr.DataArray
        Log relative improvement values.
    """

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


def _squash(da: xr.DataArray, reduce_dim: str, vector_dim: str = PROBLEM):
    """
    Squash a DataArray along a dimension, keeping the first non-NaN value.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to squash.
    reduce_dim : str
        Dimension to reduce.
    vector_dim : str, optional
        Vector dimension, by default PROBLEM.

    Returns
    -------
    xr.DataArray
        Squashed DataArray.
    """

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
        dask="allowed",  # Optional: if using dask
    )


def compute_fitting(
    trajectory: xr.DataArray,
    fit_params: xr.DataArray,
    basis_function: Callable,
):
    """
    Compute fitted values for a trajectory using a basis function.

    Parameters
    ----------
    trajectory : xr.DataArray
        Trajectory data.
    fit_params : xr.DataArray
        Parameters for fitting.
    basis_function : Callable
        Basis function for fitting.
    """
    x_input = np.linspace(0.0, 1.0, trajectory.coords[BUDGET].size)

    fitted_sigmoid = xr.apply_ufunc(
        lambda x, x_input: basis_function(x_input, *x),
        fit_params,
        kwargs={"x_input": x_input},
        input_core_dims=[["fit_params"]],
        output_core_dims=[[BUDGET]],
        vectorize=True,
    )

    fitted_sigmoid = fitted_sigmoid.assign_coords(
        {BUDGET: trajectory.coords[BUDGET]}
    )


def encode(xarr: xr.DataArray) -> xr.DataArray:
    """
    Encode a DataArray with unique integer values.

    Parameters
    ----------
    xarr : xr.DataArray
        DataArray to encode.

    Returns
    -------
    xr.DataArray
        Encoded DataArray with integer values.
    """
    _, enc = np.unique(xarr, return_inverse=True)
    return xarr.copy(data=enc.reshape(xarr.shape))
