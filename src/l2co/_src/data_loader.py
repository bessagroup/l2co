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

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

# Third-party
import jax
import numpy as np
import xarray as xr

# Local
from .naming import (
    BUDGET,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    KIND,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    PROBLEM,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
    TARGET,
    TASK_ID,
    TASK_LABELS,
)
from .optimizationdynamics import OptimizationDynamics
from .performance import (
    ertd,
    output_cummin,
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

# DEFAULT_COST_VALUES = np.logspace(-1, 6, 100) ** 0.8
DEFAULT_COST_VALUES = np.append(np.logspace(-1, 6, 1000) ** 0.8, 10**5)

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

    def __add__(self, other: DataLoader) -> DataLoader:
        """
        Concatenate two DataLoader objects.

        Parameters
        ----------
        other : DataLoader
            DataLoader to concatenate.

        Returns
        -------
        DataLoader
            Concatenated DataLoader.
        """
        if other is None:
            return self

        return replace(
            self,
            data=xr.concat([self.data, other.data], dim=PROBLEM),
            task_labels=xr.concat(
                [self.task_labels, other.task_labels], dim=PROBLEM
            ),
        )

    def join(self, other: DataLoader) -> DataLoader:
        """Join another DataLoader along the schedule dimension.

        Parameters
        ----------
        other : DataLoader
            DataLoader to join with.

        Returns
        -------
        DataLoader
            Combined DataLoader with concatenated schedules.
        """
        return replace(
            self,
            data=xr.concat([self.data, other.data], dim=SCHEDULE),
            task_labels=self.task_labels,
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
        data = data.stack({PROBLEM: (REALIZATION, TASK_ID)}, create_index=True)

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
        data = xr.open_dataset(_path.with_suffix(".odata"))

        return cls.from_xarray(data=data)

    #                                                                 Exporting
    # =========================================================================

    def task_features(
        self,
        schema: dict[str, str],
        feature_processor: FeatureProcessor | None = None,
    ) -> tuple[xr.DataArray, FeatureProcessor]:
        """Extract and transform task features from labels.

        Parameters
        ----------
        schema : dict[str, str]
            Feature name to type mapping for the processor.
        feature_processor : FeatureProcessor | None, optional
            Pre-fitted processor. If None, a new one is
            initialized from the data, by default None.

        Returns
        -------
        tuple[xr.DataArray, FeatureProcessor]
            Transformed feature array and the processor used.
        """
        tags_list = [
            {
                str(k): v
                for k, v in zip(
                    self.task_labels[FEATURE].values, xi.values, strict=True
                )
            }
            for xi in self.task_labels
        ]

        if feature_processor is None:
            feature_processor_task = FeatureProcessor.init(
                schema=schema, tags=tags_list
            )
        else:
            feature_processor_task = feature_processor

        x_features = np.stack(
            [feature_processor_task.transform(xi) for xi in tags_list]
        )

        return (
            xr.DataArray(
                x_features,
                coords={
                    PROBLEM: self.problems,
                    FEATURE: list(feature_processor_task.schema.keys()),
                },
                dims=[PROBLEM, FEATURE],
            ),
            feature_processor_task,
        )

    def optimizer_features(
        self, feature_processor: FeatureProcessor | None = None
    ) -> tuple[xr.DataArray, FeatureProcessor]:
        """Extract and transform optimizer features.

        Parameters
        ----------
        feature_processor : FeatureProcessor | None, optional
            Pre-fitted processor. If None, a new one is
            initialized from the optimizer labels,
            by default None.

        Returns
        -------
        tuple[xr.DataArray, FeatureProcessor]
            Transformed optimizer feature array and the
            processor used.
        """
        # OPTIMIZER FEATURES
        opt_labels = self.optimizers.broadcast_like(self.problems)

        tags_optimizer = [
            {"optimizer": str(xi)} for xi in opt_labels.values.flatten()
        ]

        # OPTIMIZER FEATURES
        if feature_processor is None:
            feature_processor_optimizer = FeatureProcessor.init(
                schema={"optimizer": "categorical"},
                tags=[{"optimizer": str(xi)} for xi in self.optimizers.values],
            )
        else:
            feature_processor_optimizer = feature_processor

        x_features_optimizer = np.stack(
            [
                feature_processor_optimizer.transform(xi)
                for xi in tags_optimizer
            ]
        ).reshape(*opt_labels.shape, -1)

        return (
            xr.DataArray(
                x_features_optimizer,
                coords={
                    PROBLEM: self.problems,
                    SCHEDULE: self.optimizers,
                    "optimizer_feature": ["optimizer"],
                },
                dims=[PROBLEM, SCHEDULE, "optimizer_feature"],
            ),
            feature_processor_optimizer,
        )

    def to_params(
        self,
        t: float,
        t_end: int,
        schema: dict[str, str],
        feature_processor_task: FeatureProcessor | None = None,
        feature_processor_optimizer: FeatureProcessor | None = None,
        cost_metric: Literal["fevals", "iterations"] = "fevals",
    ) -> OptimizationDynamics:
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
        feature_processor_task : FeatureProcessor | None, optional
            Feature processor for task features, by default None.
        feature_processor_optimizer : FeatureProcessor | None, optional
            Feature processor for optimizer features, by default None.

        Returns
        -------
        OptimizationDynamics
            Optimization dynamics data.
        """

        x_features_task, feature_processor_task = self.task_features(
            schema=schema,
            feature_processor=feature_processor_task,
        )
        x_features_optimizer, feature_processor_optimizer = (
            self.optimizer_features(
                feature_processor=feature_processor_optimizer
            )
        )
        x_history = self.select_t(t=t).to_traj_and_stair()

        y = self.to_labels(t=t_end, cost_metric=cost_metric)

        ds = xr.Dataset(
            {
                "x_features_task": x_features_task,
                "x_features_optimizer": x_features_optimizer,
                "x_history": x_history,
                "y": y,
            }
        )

        ds = (
            ds.unstack(PROBLEM)
            .stack({"sample": (TASK_ID, REALIZATION, SCHEDULE)})
            .transpose("sample", BUDGET, KIND, ...)
        )

        return OptimizationDynamics(
            x_features_task=ds["x_features_task"],
            x_features_optimizer=ds["x_features_optimizer"],
            x_history=ds["x_history"],
            y=ds["y"],
            feature_processor_task=feature_processor_task,
            feature_processor_optimizer=feature_processor_optimizer,
        )

    def to_traj_and_stair(self) -> xr.DataArray:
        """Compute trajectory and staircase arrays.

        Applies log-stabilized min-max scaling to the mean
        and cumulative-min output histories, producing a
        normalized (trajectory, staircase) representation.

        Returns
        -------
        xr.DataArray
            Scaled trajectory and staircase data with
            shape (batch, budget, kind).
        """
        # 1. Reset the index on PROBLEM to remove the existing MultiIndex
        # This turns taskID and realization back into normal coordinates
        data_reset = self.data.unstack(PROBLEM).isel(
            {OUTPUT_DIM: 0}, drop=True
        )

        # Get the global minimum per problem
        global_min = self.task_labels.sel({FEATURE: "global_min"}).astype(
            float
        )
        # Calculate emperical minimum per problem
        emperical_min = self.data.unstack(PROBLEM)[OUTPUT_MIN].min(
            [BUDGET, SCHEDULE, OUTPUT_DIM, REALIZATION]
        )
        emperical_min = emperical_min.expand_dims(
            {REALIZATION: np.arange(self.n_realization)}
        )

        # mask nans of global_min with values of emperical_min
        global_min = self.task_labels.sel({FEATURE: "global_min"}).astype(
            float
        )
        global_min = global_min.unstack(PROBLEM).fillna(emperical_min)

        # broadcast to another dim SCHEDULE
        global_min = global_min.expand_dims({SCHEDULE: self.data[SCHEDULE]})

        global_min = global_min.stack(
            {"batch": (SCHEDULE, TASK_ID, REALIZATION)}
        )

        x_min = data_reset[OUTPUT_MIN].stack(
            {"batch": (SCHEDULE, TASK_ID, REALIZATION)}
        )
        x_mean = data_reset[OUTPUT_MEAN].stack(
            {"batch": (SCHEDULE, TASK_ID, REALIZATION)}
        )

        out = jax.vmap(traj_and_stair)(
            x_mean.values.T, x_min.values.T, global_min.values.T
        )

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
        return y.stack({PROBLEM: (REALIZATION, TASK_ID)})

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

        y = _ertd.sel({"metric": cost_metric}).mean("cost_per_dim")
        y = y - y.mean(SCHEDULE)
        return y.rename({SCHEDULE: "label"})

    #                                                                Properties
    # =========================================================================

    # @property
    # def hash_opt_step(self) -> xr.DataArray:
    #     """
    #     Returns a hash for each optimization step.

    #     Returns
    #     -------
    #     xarray.DataArray
    #         Hash values for optimization steps.
    #     """
    #     return xr.apply_ufunc(
    #         lambda x: int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)
    #         % (2**31 - 1),
    #         self.data[UPDATE_STEP],
    #         input_core_dims=[[BUDGET]],
    #         output_core_dims=[[]],
    #         vectorize=True,
    #     )

    # @property
    # def n_iterations(self) -> int:
    #     """
    #     Returns the number of iterations (budget size).

    #     Returns
    #     -------
    #     int
    #         Number of iterations.
    #     """
    #     return self.data[BUDGET].size

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

    # @property
    # def features(self) -> xr.DataArray:
    #     """
    #     Returns the features coordinate.

    #     Returns
    #     -------
    #     xarray.DataArray
    #         The features coordinate."""
    #     """
    #     Property method to return the features coordinate

    #     Returns
    #     -------
    #     xarray.DataArray
    #         The features coordinate.
    #     """
    #     return self.task_labels[FEATURE]

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
        """Select problems matching given feature values.

        Parameters
        ----------
        feature : dict
            Mapping of feature names to target values.
            Values may be scalars or iterables.

        Returns
        -------
        DataLoader
            Filtered DataLoader containing only matching
            problems.
        """
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

        problems = feature_data[PROBLEM].where(mask, drop=True)
        return self.select_problems(problems)

    def create_strategy(self, choices: xr.DataArray, name: str) -> DataLoader:
        """Create a new optimizer schedule from per-problem choices.

        Parameters
        ----------
        choices : xr.DataArray
            Per-problem schedule selections.
        name : str
            Name for the new composite schedule.

        Returns
        -------
        DataLoader
            DataLoader with the new strategy appended along
            the schedule dimension.
        """
        # Align coordinates
        choices = choices.sortby(TASK_ID).assign_coords(
            {PROBLEM: self.problems.sortby(TASK_ID)}
        )
        strategy = self.data.sel({SCHEDULE: choices}).drop_vars(SCHEDULE)
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

    # def train_test_split(
    #     self, test_size: float = 0.2, random_state: int | None = None
    # ) -> tuple[DataLoader, DataLoader]:
    #     """
    #     Split the data into train and test sets.

    #     Parameters
    #     ----------
    #     test_size : float, optional
    #         The size of the test set, by default 0.2
    #     random_state : int, optional
    #         The random state, by default None

    #     Returns
    #     -------
    #     Tuple[DataLoader, DataLoader]
    #         The train and test data loaders.
    #     """
    #     (
    #         _X_train_indices,
    #         _X_test_indices,
    #     ) = train_test_split(
    #         range(len(self.problems)),
    #         test_size=test_size,
    #         random_state=random_state,
    #     )

    #     return (self.isel(_X_train_indices), self.isel(_X_test_indices))

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

        data = self.data.sel({SCHEDULE: optimizers})

        return replace(
            self,
            data=data,
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
        if include_best:
            sbs = self.data[SUCCESS].sum(TARGET).idxmax(SCHEDULE)
            d = self.create_strategy(choices=sbs, name="sbs")
        else:
            d = self
        # TODO : include best
        # p = self._performance(
        #     include_best=include_best, exclude_optimizer=exclude_optimizer
        # )
        dims = self.task_labels.sel(
            {FEATURE: "dimensionality"}, drop=True
        ).astype(int)

        _ertd = ertd(
            performance=d.data,
            dims=dims,
            cost_values=cost_values,
        )

        # Expand realization and stack dimensions to match valid_files
        _ertd = _ertd.expand_dims(
            {REALIZATION: np.arange(self.n_realization)}
        ).stack({PROBLEM: (REALIZATION, TASK_ID)})

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
        """Compute log relative improvement between endpoints."""
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
