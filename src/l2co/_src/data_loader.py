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
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Literal

# Third-party
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from scipy.linalg import svd
from sklearn.model_selection import train_test_split

# Local
from .naming import (
    BATCH,
    BUDGET,
    FEATURE,
    FEVALS,
    OPT_CHOICE,
    OPT_HISTORY,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    PERFORMANCE,
    PROBLEM,
    REALIZATION,
    SCHEDULE,
    SEED,
    STAIRCASE,
    SUCCESS,
    TARGET,
    TASK_ID,
    TASK_LABELS,
    TRAJECTORY,
    UPDATE_STEP,
)
from .performance import (
    output_cummin,
    scale_by_problem,
    scale_by_problem_and_schedule,
    scale_by_task,
)
from .task import Task

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

    Attributes
    ----------
    data : xarray.Dataset
        The raw optimization trajectories.
    task_labels : xarray.DataArray
        Labels of the optimization problems.
    """

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

    def __add__(self, __o: DataLoader) -> DataLoader:
        if __o is None:
            return self

        data = xr.concat([self.data, __o.data], dim=PROBLEM)
        task_labels = xr.concat(
            [self.task_labels, __o.task_labels], dim=PROBLEM
        )

        return DataLoader(
            data=data,
            task_labels=task_labels,
        )

    #                                                  Alternative Constructors
    # =========================================================================

    @classmethod
    def from_xarray(
        cls,
        data: xr.Dataset,
    ):
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
            data[TASK_LABELS],
            input_core_dims=[[PROBLEM]],
            output_core_dims=[[PROBLEM]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[object],
        ).T

        return cls(
            data=data.sortby(SCHEDULE),
            task_labels=task_labels,
        )

    @classmethod
    def from_file(
        cls, path: str, data_filename: str = "entire_dataset"
    ) -> DataLoader:
        # Create path object
        _path = Path(path)

        # Retrieve the optimization dynamics data
        if (_path / data_filename).with_suffix(".odata").exists():
            data = xr.open_dataset(
                (_path / data_filename).with_suffix(".odata")
            )
        else:
            data = xr.open_dataset(_path)
        # data = xr.open_dataset((_path / data_filename).with_suffix(".odata"))
        return cls.from_xarray(data=data)

    #                                                     Storing and exporting
    # =========================================================================

    def to_params(
        self, t: float, t_start: float = 0.0, t_end: float = 1.0
    ) -> xr.Dataset:
        """
        Generate parameters for optimization tasks within a specific
        budget range.

        Parameters
        ----------
        t : float
            The budget to select.
        t_start : float, optional
            The starting budget, by default 0.0.
        t_end : float, optional
            The ending budget, by default 1.0.

        Returns
        -------
        xr.Dataset
            Dataset containing task parameters.
        """

        d_input = self.select_t(t_start=t_start, t=t, select_one_before=True)
        d_output = self.select_t(t_start=t, t=t_end, select_one_before=False)

        perf = output_cummin(
            d_output.data.assign_coords(
                {
                    OPT_HISTORY: d_input.hash_opt_step,
                    OPT_CHOICE: d_output.hash_opt_step,
                }
            )[OUTPUT_MIN]
        ).ffill(BUDGET)

        total = []

        for fp in np.unique(perf[OPT_HISTORY]):
            _perf = perf.where(perf[OPT_HISTORY] == fp)
            total.append(scale_by_problem_and_schedule(_perf))

        performance = xr.merge(total)[OUTPUT_MIN].isel(budget=-1, drop=True)

        ds = xr.Dataset(
            {
                TASK_LABELS: self.task_labels,
                TRAJECTORY: d_input.trajectory,
                STAIRCASE: d_input.staircase,
                UPDATE_STEP: d_input.data[UPDATE_STEP],
                PERFORMANCE: performance,
            }
        )

        new_params = ds.groupby(
            (OPT_HISTORY, PROBLEM, OUTPUT_DIM), restore_coord_dims=True
        ).first()

        def f(x, first_part, second_part):
            # Create DataFrame
            df = pd.DataFrame(
                {
                    OPT_HISTORY: first_part,
                    OPT_CHOICE: second_part,
                    PERFORMANCE: x,
                }
            )

            # Pivot table: rows = y, columns = z, values = x
            matrix = df.pivot_table(
                index=OPT_HISTORY,
                columns=OPT_CHOICE,
                values=PERFORMANCE,
                dropna=False,
            )

            # Convert to NumPy array (optional)
            return matrix.to_numpy()

        new_params = new_params.assign_coords(
            {OPT_CHOICE: np.unique(ds[OPT_CHOICE])}
        )

        new_params[PERFORMANCE] = xr.apply_ufunc(
            f,
            ds[PERFORMANCE],
            ds[OPT_HISTORY],
            ds[OPT_CHOICE],
            input_core_dims=[[SCHEDULE], [SCHEDULE], [SCHEDULE]],
            output_core_dims=[[OPT_HISTORY, OPT_CHOICE]],
            vectorize=True,
        )

        # Transpose dims
        desired_order = (PROBLEM, OUTPUT_DIM, OPT_HISTORY)

        return new_params.map(
            lambda var: var.transpose(
                *[d for d in desired_order if d in var.dims],
                *[d for d in var.dims if d not in desired_order],
            )
        )

    def to_all_params(
        self,
        window_size: float,
        budget_size: int | None = None,
        include_last: bool = False,
    ) -> xr.DataTree:
        """
        Generate parameters for all tasks over multiple budget windows.

        Parameters
        ----------
        window_size : float
            Size of each budget window.
        budget_size : Optional[int], optional
            Total budget size, by default None.
        include_last : bool, optional
            Whether to include the last window, by default False.

        Returns
        -------
        xr.DataTree
            DataTree containing parameters for all tasks.
        """

        if budget_size is None:
            budget_size = self.data[BUDGET].size

        if include_last:
            window_end = 1.0 + window_size
        else:
            window_end = 1.0

        _dict = {}

        for t in np.arange(window_size, window_end, window_size):
            t_end = np.minimum(t + window_size, 1.0)
            params = self.to_params(t_start=0.0, t=t, t_end=t_end)
            params[BUDGET] = range(
                budget_size - params[BUDGET].size, budget_size
            )

            params = params.reindex(
                budget=range(budget_size), fill_value=PADDING_VALUE
            )

            params[UPDATE_STEP] = params[UPDATE_STEP].astype(str)

            _dict[f"{t:.2f}"] = params

        return xr.DataTree.from_dict(_dict)

    @property
    def hash_opt_step(self) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda x: int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)
            % (2**31 - 1),
            self.data[UPDATE_STEP],
            input_core_dims=[[BUDGET]],
            output_core_dims=[[]],
            vectorize=True,
        )

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
    def n_iterations(self) -> int:
        return self.data[BUDGET].size

    @property
    def n_realization(self) -> int:
        return np.unique(self.data[REALIZATION]).size

    @property
    def features(self) -> xr.DataArray:
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
        Property method to return the problem coordinate

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
    def staircase(self) -> xr.DataArray:
        return scale_by_problem(output_cummin(self.data[OUTPUT_MIN])).ffill(
            BUDGET
        )

    @property
    def trajectory(self) -> xr.DataArray:
        """
        Returns the trajectory of the optimization process.

        Returns
        -------
        xr.DataArray
            Trajectory data.
        """
        trajectory = scale_by_problem(self.data[OUTPUT_MEAN])
        return trajectory.transpose(SCHEDULE, OUTPUT_DIM, PROBLEM, BUDGET)

    def _performance(
        self, include_best: bool, exclude_optimizer: str | None = None
    ) -> xr.DataArray:
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

        performance = scale_by_problem_and_schedule(
            output_cummin(self.data[OUTPUT_MIN])
        ).ffill(BUDGET)

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
            performance = xr.merge([performance, perf_sbs, perf_vbs])[
                OUTPUT_MIN
            ]

        print(f"Performance calculation took: {perf_counter() - t_start:0.2f}")
        return performance

    @property
    def performance(self) -> xr.DataArray:
        return self._performance(include_best=False, exclude_optimizer=None)

    @property
    def performance_per_realization(self) -> xr.DataArray:
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

        return DataLoader(
            task_labels=task_labels,
            data=data,
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
        _problem_indices = problem_indices.sortby(TASK_ID)

        return DataLoader(
            task_labels=self.task_labels.sel({PROBLEM: _problem_indices}),
            data=self.data.sel({PROBLEM: _problem_indices}),
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
            task_labels=self.task_labels.isel({PROBLEM: problems}),
            data=self.data.isel({PROBLEM: problems}),
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

        return DataLoader(
            task_labels=self.task_labels,
            data=self.data.sel({BUDGET: slice(t_prior, t)}),
        )

    #                                                      Selecting optimizers
    # =========================================================================

    def select_single_optimizers(self, squash: bool = False) -> DataLoader:
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
            optimizers=[o for o in self.optimizers.values if "+" not in o],
            squash=squash,
        )

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

        return DataLoader(
            task_labels=self.task_labels,
            data=data.sortby(SCHEDULE),
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
            dtype=object,
        )
        data_selected = self.data.sel(
            {SCHEDULE: filtered_optimizer}
        ).assign_coords(
            {
                SCHEDULE: np.array(
                    [
                        o.split("+")[-1] if "+" in o else o
                        for o in filtered_optimizer
                    ],
                    dtype=object,
                )
            }
        )

        return DataLoader(
            task_labels=self.task_labels,
            data=data_selected.sortby(SCHEDULE),
        )

    #                                                       Performance metrics
    # =========================================================================

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
    ) -> xr.DataArray:
        p = scale_by_task(output_cummin(self.data[OUTPUT_MIN])).ffill(BUDGET)

        return ertd(performance=p, fevals=self.data[FEVALS])

    def load(self):
        self.data.load()

    def save(self, path: str):
        _path = Path(path)

        _path.mkdir(parents=True, exist_ok=True)

        # Save optimization dynamics data
        self.data.to_netcdf((_path / "entire_dataset").with_suffix(".nc"))

        # Convert task_labels to a DataFrame and save as CSV
        df_task_labels = pd.DataFrame(
            self.task_labels.values,
            index=self.task_labels.coords[TASK_ID].values,
            columns=self.task_labels.coords[FEATURE].values,
        )
        df_task_labels = df_task_labels[
            ~df_task_labels.index.duplicated(keep="first")
        ]
        df_task_labels.to_csv((_path / "labels").with_suffix(".csv"))


def calculate_sbs(
    performance: xr.DataArray, exclude_optimizer: str | Iterable[str]
) -> xr.DataArray:
    return calculate_best_solver(
        performance=performance,
        exclude_optimizer=exclude_optimizer,
        group=TASK_ID,
    )


def calculate_vbs(
    performance: xr.DataArray, exclude_optimizer: str | Iterable[str]
) -> xr.DataArray:
    return calculate_best_solver(
        performance=performance,
        exclude_optimizer=exclude_optimizer,
        group=PROBLEM,
    )


def calculate_best_solver(
    performance: xr.DataArray,
    exclude_optimizer: str | Iterable[str],
    group: str,
) -> xr.DataArray:
    if exclude_optimizer is not None:
        if isinstance(exclude_optimizer, str):
            exclude_optimizer = [exclude_optimizer]

        check_sbs = performance.where(
            ~performance[SCHEDULE].isin(exclude_optimizer), drop=True
        )
    else:
        check_sbs = performance

    best = (
        check_sbs.groupby(group)
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


def performance_profile(
    performance: xr.DataArray, max_f_value: int = 2.0
) -> xr.DataArray:
    """
    Create a performance profile for the given performance data.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data for optimization problems.
    max_f_value : int, optional
        Maximum performance ratio, by default 2.0.

    Returns
    -------
    xr.DataArray
        Performance profile.
    """

    p = performance.isel({BUDGET: -1})
    rij = (p + 1.0) / 1.0

    ds_f = xr.Dataset(
        {
            "performance_profile": rij,
            "relative_error": xr.DataArray(
                np.logspace(np.log10(1.0), np.log10(max_f_value), 1000),
                dims="relative_error",
            ),
        }
    )

    performance_profile = (
        np.less_equal(ds_f, ds_f.relative_error + 1e-2)
        .where(~ds_f.isnull(), np.nan)
        .mean(dim=PROBLEM)["performance_profile"]
    )

    rel_error = (performance_profile["relative_error"] * 100) - 100

    return performance_profile.assign_coords({"relative_error": rel_error})


def rtd(performance: xr.DataArray) -> xr.DataArray:
    """
    Compute the runtime distribution (RTD) profile.

    Parameters
    ----------
    performance : xr.DataArray
        Performance data for optimization problems.

    Returns
    -------
    xr.DataArray
        RTD profile.
    """

    return (1.0 - performance).mean(PROBLEM)


def ertd(performance: xr.DataArray, fevals: xr.DataArray) -> xr.DataArray:
    targets = xr.DataArray(np.linspace(0.0, 1.0, 51), dims=TARGET)
    solved = performance <= targets
    success = solved.any(BUDGET)
    first_solved = solved.argmax(BUDGET).where(
        success, other=solved[BUDGET].argmax()
    )

    ds = xr.Dataset(
        {
            FEVALS: fevals.isel({BUDGET: first_solved}),
            SUCCESS: success,
        }
    )

    ert_multi = compute_ert_multi_seed(ds, n_seeds=50, n_picks=50)

    budgets = xr.DataArray(
        np.logspace(0, 6, 20, dtype=int),
        dims="fevals",
        coords={"fevals": np.logspace(0, 6, 20, dtype=int)},
    )
    ertd = (ert_multi <= budgets).mean([SEED, TASK_ID, OUTPUT_DIM, TARGET])

    return ertd


def compute_ert_group(ds: xr.Dataset, seed: int, n_picks: int) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    n_realizations = int(ds[REALIZATION].max())
    samples = np.concat(
        [
            rng.permutation(n_realizations + 1),
            rng.integers(0, n_realizations + 1, size=n_picks - n_realizations),
        ]
    )

    ds = ds.isel({PROBLEM: samples})
    # Find index of first True along "problem"
    first_true = ds[SUCCESS].argmax(PROBLEM)

    # Handle case where there is no True at all
    first_true = first_true.where(ds[SUCCESS].any(PROBLEM), other=-1)

    # Coordinate index
    idx = xr.DataArray(np.arange(ds.sizes[PROBLEM]), dims=PROBLEM)

    # Mask up to first True
    mask = (idx <= first_true).fillna(False)

    # Sum fevals up to first True
    ert = ds[FEVALS].where(mask, 0).sum(PROBLEM)

    # If not any success, put NaN
    ert = ert.where(ds[SUCCESS].any(PROBLEM))
    return ert


def compute_ert_multi_seed(
    ds: xr.Dataset, n_seeds: int, n_picks: int
) -> xr.DataArray:
    """
    Compute ERT across multiple random seeds and return an xarray
    with dimensions (seed, taskID, ...).
    """
    # Create a version of compute_ert_group with n_picks fixed
    compute_partial = partial(compute_ert_group, n_picks=n_picks)

    # Map over seeds in a vectorized fashion
    ert_list = [
        ds.groupby(TASK_ID)
        .map(compute_partial, seed=seed)
        .expand_dims({SEED: [seed]})
        for seed in np.arange(n_seeds)
    ]

    # Concatenate along the seed dimension
    ert_all = xr.concat(ert_list, dim=SEED)

    return ert_all


# =============================================================================


class OptimizationDynamics:
    def __init__(self, data: xr.Dataset):
        self.data = data

    @classmethod
    def from_data(
        cls: type[OptimizationDynamics],
        data: xr.DataTree,
        previous_step: str | None = None,
    ) -> OptimizationDynamics:
        ds_list = []
        for k in data.keys():
            ds = data[k].to_dataset()

            # Impute missing values
            ds[PERFORMANCE] = xr.apply_ufunc(
                impute_missing_values_svd,
                ds[PERFORMANCE],
                input_core_dims=[[PROBLEM, OPT_CHOICE]],
                output_core_dims=[[PROBLEM, OPT_CHOICE]],
                vectorize=True,
            )

            # Stack the dataset along a new 'batch' dimension
            ds_list.append(
                ds.stack(
                    {BATCH: (PROBLEM, OPT_HISTORY, OUTPUT_DIM)},
                    create_index=False,
                )
            )

        # Concatenate all datasets along the 'batch' dimension
        new_ds: xr.Dataset = xr.concat(ds_list, dim=BATCH)

        nan_flags = [
            var.isnull().any(dim=set(var.dims) - {BATCH})
            for var in new_ds.data_vars.values()
        ]

        has_nan_anywhere = xr.concat(nan_flags, dim="_vars").any(dim="_vars")
        new_ds = new_ds.sel({BATCH: ~has_nan_anywhere}).transpose()

        new_ds[UPDATE_STEP] = encode(new_ds[UPDATE_STEP])

        return cls(data=new_ds)

    @property
    def X_features(self) -> pd.DataFrame:
        return (
            self.data[TASK_LABELS].to_pandas().infer_objects().convert_dtypes()
        )

    @property
    def X_trajectory(self) -> jax.Array:
        return jnp.stack(
            (
                self.data[TRAJECTORY].values,
                self.data[STAIRCASE].values,
                self.data[UPDATE_STEP].values,
            ),
            axis=-1,
        )

    @property
    def Y(self) -> jax.Array:
        return jnp.array(self.data[PERFORMANCE])

    @classmethod
    def from_task_and_run(
        cls: type[OptimizationDynamics],
        task: Task,
        data: ExperimentData | None = None,
    ) -> OptimizationDynamics: ...


def load_dynamics(
    path: str | Path | xr.DataTree,
) -> dict[str, OptimizationDynamics]:
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


# def compute_parameters(
#         trajectory: xr.DataArray,
#         optimizer: optax.GradientTransformation = optax.adam(1e-01),
#         inducing_points: int = 10, num_iters: int = 300) -> xr.DataArray:
#     """
#     Compute the parameters of the optimization trajectory.

#     Parameters
#     ----------
#     data : xarray.Dataset
#         The optimization trajectory data.
#     Returns
#     -------
#     xarray.DataArray
#         The selected trajectory.
#     """
#     x = jnp.array(trajectory.values.reshape(-1, trajectory.budget.size))

#     params = jax.vmap(
#         partial(fit_gpr, optimizer=optimizer,
#                 inducing_points=inducing_points, num_iters=num_iters)
#     )(x)

#     params = params.reshape(
#         trajectory.optimizer.size, trajectory.problem.size,
#         trajectory.output_dim.size, -1
#     )

#     return xr.DataArray(
#         params,
#         dims=[SCHEDULE, PROBLEM, OUTPUT_DIM, 'fit_params'],
#         coords={
#             SCHEDULE: trajectory.optimizer,
#             TASK_ID: trajectory.itemID,
#             OUTPUT_DIM: trajectory.output_dim,
#         }
#     )


# def compute_staircase_parameters(
#         staircase: xr.DataArray,
#         num_iters: int = 50) -> xr.DataArray:
#     x = jnp.array(staircase.values.reshape(-1, staircase.budget.size))

#     params = jax.vmap(
#         partial(fit_basis_function, max_iter=num_iters)
#     )(x)

#     params = params.reshape(
#         staircase.optimizer.size, staircase.problem.size,
#         staircase.output_dim.size, -1
#     )

#     return xr.DataArray(
#         params,
#         dims=[SCHEDULE, PROBLEM, OUTPUT_DIM, 'staircase_fit_params'],
#         coords={
#             SCHEDULE: staircase.optimizer,
#             TASK_ID: staircase.itemID,
#             OUTPUT_DIM: staircase.output_dim,
#         }
#     )


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


def _squash(da: xr.DataArray, reduce_dim: str, vector_dim: str = PROBLEM):
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
    _, enc = np.unique(xarr, return_inverse=True)
    return xarr.copy(data=enc.reshape(xarr.shape))
