"""
Module containing the data loader of the L2CO framework.

Classes
-------

DataLoader
    The single data loader for the L2CO framework. Holds the optimization
    dataset (raw loss trajectories *and* pre-computed ERT metrics) together
    with the parsed task labels, and provides problem/optimizer selection,
    concatenation, feature extraction, the (trajectory, staircase) learning
    representation, the expected-runtime distribution (ERTD) profile, the
    learning data (``to_params``), and the AI-friendly summary (``report``).

Notes
-----
The dataset is opened lazily (zarr-backed via xarray); operations stay lazy
with respect to the heavy ``loss_history`` variables. Only
``to_traj_and_stair`` (which feeds ``jax.vmap``) and explicit consumption of
``valid_files`` / ``_performance`` materialise data variables. Call
:meth:`DataLoader.load` to force everything into memory.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

# Third-party
import jax
import numpy as np
import xarray as xr

# Local
from ..naming import (
    BUDGET,
    COST,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    KIND,
    METRIC,
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
from ..optimization_dynamics import OptimizationDynamics
from ..performance import (
    ERTD_SEED_BATCH,
    ertd,
    output_cummin,
    scale_by_task,
)
from ..task_features import FeatureProcessor
from ..tree_utils import traj_and_stair
from .baselines import calculate_sbs, calculate_vbs

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                                     Constants
# =============================================================================

DEFAULT_COST_VALUES = np.concatenate(
    ([10**-1], np.logspace(-1, 6, 1000) ** 0.8, [10**5])
)


#                                                                   Data Loader
# =============================================================================


@dataclass
class DataLoader:
    """
    The data loader of the L2CO framework.

    Holds the full optimization dataset -- both the loss-history variables
    (``output_min``, ``output_mean``, ``fevals``, ``iterations``,
    ``update_step``) over the ``budget`` axis and the pre-computed ERT
    metrics (``cost``, ``success``, ``target_value``) over the ``target`` /
    ``metric`` axes -- together with the parsed task labels. Provides the
    problem/optimizer selection, concatenation, and feature-extraction
    logic as well as the loss-trajectory and performance operations.

    Parameters
    ----------
    data : xarray.Dataset
        The optimization dataset (loss trajectories and ERT metrics).
    task_labels : xarray.DataArray
        Labels of the optimization problems.

    Attributes
    ----------
    data : xarray.Dataset
        The optimization dataset.
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
        Concatenate two loaders along the problem dimension.

        Non-concat dimensions are outer-aligned (``join="outer"``,
        stated explicitly since xarray's default is changing to
        ``"exact"``): coordinates present on only one side are kept and
        NaN-padded on the other.

        Parameters
        ----------
        other : DataLoader
            Loader to concatenate.

        Returns
        -------
        DataLoader
            Concatenated loader.
        """
        if other is None:
            return self

        return replace(
            self,
            data=xr.concat([self.data, other.data], dim=PROBLEM, join="outer"),
            task_labels=xr.concat(
                [self.task_labels, other.task_labels],
                dim=PROBLEM,
                join="outer",
            ),
        )

    def join(self, other: DataLoader) -> DataLoader:
        """Join another loader along the schedule dimension.

        The ``problem`` axes are outer-aligned (``join="outer"``, stated
        explicitly since xarray's default is changing to ``"exact"``):
        problems present on only one side are kept and NaN-padded on the
        other, and data variables missing from one side are NaN-filled
        on that side's schedules.

        Parameters
        ----------
        other : DataLoader
            Loader to join with.

        Returns
        -------
        DataLoader
            Combined loader with concatenated schedules.
        """
        return replace(
            self,
            data=xr.concat(
                [self.data, other.data], dim=SCHEDULE, join="outer"
            ),
            task_labels=self.task_labels,
        )

    #                                                  Alternative Constructors
    # =========================================================================

    @classmethod
    def from_xarray(cls, data: xr.Dataset) -> DataLoader:
        """
        Create a loader from an xarray.Dataset.

        Stacks ``(realization, taskID)`` into the ``problem`` dimension and
        parses the string ``task_labels`` variable into typed values. Every
        data variable in ``data`` is kept (loss trajectories and ERT
        metrics alike).

        The label parse is forced to ``object`` dtype and runs on an
        eagerly-loaded copy of ``task_labels``: without ``otypes``,
        ``np.vectorize`` (and dask's sample-based dtype inference on a
        lazy zarr-backed input, e.g. from ``l2co.databank``) infers the
        output dtype from the first parsed element -- a boolean feature
        like ``separable`` would collapse the *entire* labels array to
        ``bool`` (``global_min`` ``"-116.56"`` -> ``True``).

        Parameters
        ----------
        data : xarray.Dataset
            Dataset containing optimization data and task labels.

        Returns
        -------
        DataLoader
            Instantiated loader.
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

        # Labels are tiny; load them so the parse below never goes
        # through dask (whose blockwise dtype inference ignores
        # ``otypes``-less vectorize results -- see docstring).
        task_labels = xr.apply_ufunc(
            np.vectorize(parse_mixed_value, otypes=[object]),
            data[TASK_LABELS].load(),
            input_core_dims=[[PROBLEM]],
            output_core_dims=[[PROBLEM]],
            vectorize=True,
        ).T

        # Remove variable features from data
        data = data.drop_vars([TASK_LABELS, FEATURE])

        return cls(
            data=data,
            task_labels=task_labels,
        )

    @classmethod
    def from_file(
        cls, path: str, data_filename: str = "entire_dataset"
    ) -> DataLoader:
        """
        Create a loader from a file path.

        Parameters
        ----------
        path : str
            Path to the dataset file or directory.
        data_filename : str, optional
            Name of the dataset file, by default "entire_dataset".

        Returns
        -------
        DataLoader
            Instantiated loader.
        """
        # Create path object
        _path = Path(path)

        # Retrieve the optimization dynamics data
        data = xr.open_dataset(_path.with_suffix(".odata"))

        return cls.from_xarray(data=data)

    #                                                                  Features
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
            Feature name to type mapping for the processor
            (see :meth:`FeatureProcessor.init`).
        feature_processor : FeatureProcessor | None, optional
            Pre-fitted processor. If None, a new one is
            initialized from the schema, by default None.

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
            feature_processor_task = FeatureProcessor.init(schema=schema)
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
                    FEATURE: list(feature_processor_task.vocabs.keys()),
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

        # OPTIMIZER FEATURES: one categorical feature whose vocabulary
        # is the data's schedule names, in schedule order.
        if feature_processor is None:
            feature_processor_optimizer = FeatureProcessor.init(
                schema={
                    "optimizer": [str(xi) for xi in self.optimizers.values]
                },
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

    #                                                                Properties
    # =========================================================================

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

        Derived from ``success`` nullity: a ``(problem, schedule)`` cell is
        valid iff it has at least one non-null target. This assumes the
        ``success`` variable is present, which holds for every dataset
        produced by the databank and by ``HistoryState.to_dataloader``
        (both always build the ERT metrics). ``join="outer"`` NaN-pads any
        missing ``(task, optimizer)`` combination identically across the
        loss and performance variables, so this mask matches the loss-side
        validity.

        Returns
        -------
        xarray.DataArray
            Boolean array indicating valid files.
        """
        return ~self.data[SUCCESS].isnull().all(TARGET)

    @property
    def performance(self) -> xr.Dataset:
        """
        Returns the (loss-trajectory) performance data for optimizers.

        Returns
        -------
        xarray.Dataset
            Performance data.
        """
        return self._performance(include_best=False, exclude_optimizer=None)

    #                                                                 Selecting
    # =========================================================================

    def select_problems(self, problem_indices: xr.DataArray) -> DataLoader:
        """
        Select a subset of problems of the loader.

        Parameters
        ----------
        problem_indices : xarray.DataArray
            The problem indices to select.

        Returns
        -------
        DataLoader
            The selected loader.

        Notes
        -----
        Selects on the problem dimension. Also sorts the problem indices on
        itemID.
        """
        _problem_indices = problem_indices.sortby(TASK_ID)

        return replace(
            self,
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
            Filtered loader containing only matching problems.
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
            Loader with the new strategy appended along the schedule
            dimension.
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
        Select a subset of problems of the loader.

        Parameters
        ----------
        problems : Iterable[int]
            The problem indices to select.

        Returns
        -------
        DataLoader
            The selected loader.

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
            Loader containing selected optimizers.
        """
        if isinstance(optimizers, str):
            optimizers = [optimizers]

        data = self.data.sel({SCHEDULE: optimizers})

        return replace(
            self,
            data=data,
        )

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
        select_one_before : bool, optional
            Whether to include the budget value just before ``t_start``,
            by default True.

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

    #                                                                 Exporting
    # =========================================================================

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

        Combines the loss-trajectory history (``x_history``) with the
        per-schedule performance labels (``y``).

        Parameters
        ----------
        t : float
            Budget value.
        t_end : int
            End budget value.
        schema : dict[str, str]
            Schema for feature processing.
        feature_processor_task : FeatureProcessor | None, optional
            Feature processor for task features, by default None.
        feature_processor_optimizer : FeatureProcessor | None, optional
            Feature processor for optimizer features, by default None.
        cost_metric : {'fevals', 'iterations'}, optional
            Cost metric to use, by default 'fevals'.

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

    #                                                       Performance metrics
    # =========================================================================

    def ertd_profile(
        self,
        include_best: bool = False,
        exclude_optimizer: str | None = None,
        cost_values: np.ndarray = DEFAULT_COST_VALUES,
        seed_batch: int = ERTD_SEED_BATCH,
    ) -> xr.DataArray:
        """
        Compute the expected runtime distribution (ERTD) profile.

        The profile is per-problem: it is never reduced over ``taskID``,
        and the bootstrap draws depend only on the *number* of
        realizations. A task subset therefore yields exactly the rows the
        full selection would, so a caller may shard the computation over
        tasks and concatenate the partials along ``problem`` (see
        ``tests/unit/test_ertd_separability.py``).

        Parameters
        ----------
        include_best : bool, optional
            Whether to include the per-problem virtual-best-solver
            (``vbs``) envelope, by default False.
        exclude_optimizer : str, optional
            Optimizer to exclude, by default None.
        cost_values : np.ndarray, optional
            Cost budget thresholds, by default ``DEFAULT_COST_VALUES``.
        seed_batch : int, optional
            Bootstrap replicates held in memory at once, forwarded to
            :func:`l2co._src.performance.ertd`. Affects peak memory only,
            not the result.

        Returns
        -------
        xr.DataArray
            ERTD profile.
        """
        if include_best:
            success = self.data[SUCCESS]
            counts = success.sum(TARGET)
            tied_for_best = counts == counts.max(SCHEDULE)

            # Tie-break: TARGET runs hardest -> easiest and success is a
            # monotone step function, so among schedules tied on target
            # count the first True entry is the hardest target each
            # reached. Prefer whichever got there in fewer fevals.
            # ``fillna`` guards a (schedule, task) combination absent
            # from the store, which argmax's NaN-slice handling can't.
            # ``compute`` realizes the (small) indexer eagerly -- the
            # store is dask-backed, and vectorized ``isel`` below
            # rejects a still-chunked indexer array (see the identical
            # ``.compute()`` on ``calculate_ert``'s ``first_solved``).
            hardest_reached = success.fillna(0.0).argmax(TARGET).compute()
            speed = (
                self.data[COST]
                .sel({METRIC: FEVALS}, drop=True)
                .isel({TARGET: hardest_reached})
            )

            vbs = speed.where(tied_for_best, other=np.inf).idxmin(SCHEDULE)
            d = self.create_strategy(choices=vbs, name="vbs")
        else:
            d = self
        dims = self.task_labels.sel(
            {FEATURE: "dimensionality"}, drop=True
        ).astype(int)

        _ertd = ertd(
            performance=d.data,
            dims=dims,
            cost_values=cost_values,
            seed_batch=seed_batch,
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

    def _performance(
        self, include_best: bool, exclude_optimizer: str | None = None
    ) -> xr.Dataset:
        """
        Computes the (loss-trajectory) performance of the optimizers.

        Parameters
        ----------
        include_best : bool
            Whether to include the best optimizer in the performance.
        exclude_optimizer : str, optional
            Name of the optimizer to exclude from the performance.

        Returns
        -------
        xr.Dataset
            Performance data.
        """
        t_start = perf_counter()
        print("Triggered performance")

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

    #                                                                    Report
    # =========================================================================

    def report(self) -> dict[str, Any]:
        """Compute an AI-friendly JSON-serialisable summary of this loader.

        Thin delegate around
        :func:`l2co._src.data.report_helpers.build_report`. The returned
        dict captures the dataset's shape, per-schedule performance
        scalars, cross-schedule rankings, task-feature breakdowns, and a
        list of programmatically-detected observations. Saving the JSON is
        the caller's job; provenance (git SHAs, config hashes, hostname,
        wall-clock) is *not* included so the method stays a pure function
        of the underlying loss and performance data.

        Returns
        -------
        dict[str, Any]
            Document conforming to ``schema_version: 1``.
        """
        # Imported lazily to avoid an import cycle with report_helpers,
        # which type-hints DataLoader.
        from .report_helpers import build_report

        return build_report(self)

    #                                                                   Loading
    # =========================================================================

    def load(self):
        """
        Load the data into memory.
        """
        self.data.load()


# Baseline metrics (`calculate_sbs`, `calculate_vbs`, `calculate_best_solver`,
# `log_relative_improvement`, `encode`) live in `l2co._src.data.baselines`
# and are re-exported from the `l2co._src.data` package __init__.
