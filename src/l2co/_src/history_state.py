"""
Module for creating the history state.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, Float, Int
from l2co_tasks import Task

from .data_loader import DataLoader
from .naming import (
    BUDGET,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_STD,
    REALIZATION,
    SCHEDULE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)
from .optimizerschedule import OptimizationStep
from .performance import calculate_ert
from .typing import OptHistoryType
from .validation import normalize_budget_grid

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class HistoryState(eqx.Module):
    """State container for optimization history tracking.

    This class stores the optimization history including input parameters,
    output statistics, and evaluation metrics across iterations. Entries
    are written front-to-back: ``cursor`` points at the next slot to fill,
    so valid data lives in ``[0, cursor)`` and uninitialized padding (NaN
    floats / zero ints) lives in ``[cursor, n_iterations)``.

    Attributes
    ----------
    output_min : Float[Array, " n_iterations"]
        Minimum output value at each iteration.
    output_mean : Float[Array, " n_iterations"]
        Mean output value at each iteration.
    output_std : Float[Array, " n_iterations"]
        Standard deviation of output values at each iteration.
    update_step : Int[Array, " n_iterations"]
        Update step number at each iteration.
    fevals : Int[Array, " n_iterations"]
        Number of function evaluations at each iteration.
    iterations : Int[Array, " n_iterations"]
        Iteration counter.
    cursor : Int[Array, ""]
        Index of the next write position; equals the number of filled
        slots. The most-recent valid entry lives at ``cursor - 1``.
    """

    output_min: Float[Array, " n_iterations"]
    output_mean: Float[Array, " n_iterations"]
    output_std: Float[Array, " n_iterations"]
    update_step: Int[Array, " n_iterations"]
    fevals: Int[Array, " n_iterations"]
    iterations: Int[Array, " n_iterations"]
    cursor: Int[Array, ""]

    @classmethod
    def init(cls, max_iterations: int) -> HistoryState:
        """Initialize a new HistoryState with NaN values.

        Parameters
        ----------
        max_iterations : int
            Number of iterations to track in history.

        Returns
        -------
        HistoryState
            Initialized history state with NaN floats, zero ints, and
            ``cursor`` set to 0 (empty buffer, next write at index 0).
        """
        nans_output = jnp.full((max_iterations,), jnp.nan, dtype=jnp.float32)
        zeros_int32 = jnp.zeros((max_iterations,), dtype=jnp.int32)

        return cls(
            output_min=nans_output,
            output_mean=nans_output,
            output_std=nans_output,
            update_step=zeros_int32,
            fevals=zeros_int32,
            iterations=zeros_int32,
            cursor=jnp.array(0, dtype=jnp.int32),
        )

    def reset(self) -> HistoryState:
        """Reset the history state to initial NaN values and cursor 0."""
        return self.init(max_iterations=self.output_min.shape[0])

    @classmethod
    def from_history(cls, history: OptHistoryType) -> HistoryState:
        """Create HistoryState from optimization history.

        Parameters
        ----------
        history : OptHistoryType
            Optimization history containing parameters and losses.

        Returns
        -------
        HistoryState
            History state populated with statistics from optimization history.
            ``cursor`` is set to ``history.loss.shape[0]`` since the buffer
            produced by the scan is fully filled.
        """
        # History has shape (n_iter, pop)
        # Compute statistics across P dimension for each time step
        return cls(
            output_min=jnp.min(history.loss, axis=1),
            output_mean=jnp.mean(history.loss, axis=1),
            output_std=jnp.std(history.loss, axis=1),
            update_step=history.update_step,
            fevals=history.fevals,
            iterations=history.iterations,
            cursor=jnp.array(history.loss.shape[0], dtype=jnp.int32),
        )

    def add(self, other: HistoryState) -> HistoryState:
        """
        Insert ``other``'s entries front-to-back at ``self.cursor``.

        Each field of ``other`` (length ``K``) is written into ``self`` at
        positions ``[cursor, cursor + K)`` via ``dynamic_update_slice``;
        the cursor is advanced by ``K``. Entries before ``cursor`` are
        preserved unchanged. Existing padding past ``cursor + K`` is left
        untouched (NaN floats, zero ints).

        Precondition: ``self.cursor + K <= n_iterations``. The buffer is
        sized to fit the full episode at the call site, so this is
        guaranteed by construction in current callers (rl2co allocates
        ``HistoryState.init(max_iterations + 1)``).

        Parameters
        ----------
        other : HistoryState
            HistoryState whose entries are appended at the cursor.

        Returns
        -------
        HistoryState
            New HistoryState with ``other``'s entries written at the cursor
            and ``cursor`` advanced by ``other.output_min.shape[0]``.
        """
        cursor = self.cursor

        def write(a, b):
            return jax.lax.dynamic_update_slice(a, b, (cursor,))

        return HistoryState(
            output_min=write(self.output_min, other.output_min),
            output_mean=write(self.output_mean, other.output_mean),
            output_std=write(self.output_std, other.output_std),
            update_step=write(self.update_step, other.update_step),
            fevals=write(self.fevals, other.fevals),
            iterations=write(self.iterations, other.iterations),
            cursor=cursor + other.output_min.shape[0],
        )

    def to_xarray(self, budget_steps: int | None = None) -> xr.Dataset:
        """Convert history state to an xarray Dataset.

        All ``n_iterations`` slots are written through, including the
        padding past ``cursor`` (NaN floats, zero ints). The Dataset
        does not carry ``cursor`` itself; callers that need the valid
        range must consult the source ``HistoryState`` directly.

        Parameters
        ----------
        budget_steps : int | None, optional
            Number of steps for budget interpolation. If None, no
            interpolation is performed, by default None. When set, the
            result is passed through :func:`normalize_budget_grid`,
            which requires ``budget_steps >= 3``.

        Returns
        -------
        xr.Dataset
            Dataset containing the history with appropriate
            coordinates.

        Notes
        -----
        ``fevals`` / ``iterations`` are stored as cumulative sums. This
        relies on the padding convention that slots past ``cursor`` are
        zero ints, so the cumulative curve plateaus past the valid
        range. If the padding sentinel ever changes, this needs to be
        revisited.
        """

        def _wrap(arr: jax.Array) -> xr.DataArray:
            return xr.DataArray(arr, dims=[BUDGET]).expand_dims(
                {OUTPUT_DIM: ["y"]}
            )

        fevals_cumsum = jnp.cumsum(self.fevals)
        iterations_cumsum = jnp.cumsum(self.iterations)

        ds = xr.Dataset(
            {
                OUTPUT_MIN: _wrap(self.output_min),
                OUTPUT_MEAN: _wrap(self.output_mean),
                OUTPUT_STD: _wrap(self.output_std),
                UPDATE_STEP: _wrap(self.update_step),
                FEVALS: _wrap(fevals_cumsum),
                ITERATIONS: _wrap(iterations_cumsum),
            }
        )

        if budget_steps is not None:
            return normalize_budget_grid(ds, budget_steps)

        return ds

    def to_xarray_realizations(
        self, budget_steps: int | None = None
    ) -> xr.Dataset:
        """Convert history state to xarray Dataset with realization dimension.

        Parameters
        ----------
        budget_steps : int | None, optional
            Number of steps for budget interpolation. If None, no
            interpolation is performed, by default None.

        Returns
        -------
        xr.Dataset
            Dataset containing histories for all realizations, with
            dims ``(REALIZATION, BUDGET, OUTPUT_DIM)``.

        Notes
        -----
        Expects ``self`` to have been produced by ``eqx.filter_vmap``:
        every field carries a leading ``n_realizations`` axis,
        including the originally-scalar ``cursor`` (now shape
        ``(n_realizations,)``). Calling this on a non-vmapped
        ``HistoryState`` will fail with a shape error.

        ``fevals`` / ``iterations`` are cumulatively summed along the
        ``BUDGET`` axis only (per realization), preserving the
        plateau-past-cursor behaviour of :meth:`to_xarray`. The
        per-realization ``cursor`` values themselves are not exposed
        on the resulting Dataset; consult the source ``HistoryState``
        when the valid range is needed.
        """
        n_realizations = self.output_min.shape[0]

        def _wrap(arr: jax.Array) -> xr.DataArray:
            return xr.DataArray(arr, dims=[REALIZATION, BUDGET]).expand_dims(
                {OUTPUT_DIM: ["y"]}
            )

        fevals_cumsum = jnp.cumsum(self.fevals, axis=1)
        iterations_cumsum = jnp.cumsum(self.iterations, axis=1)

        ds = xr.Dataset(
            {
                OUTPUT_MIN: _wrap(self.output_min),
                OUTPUT_MEAN: _wrap(self.output_mean),
                OUTPUT_STD: _wrap(self.output_std),
                UPDATE_STEP: _wrap(self.update_step),
                FEVALS: _wrap(fevals_cumsum),
                ITERATIONS: _wrap(iterations_cumsum),
            }
        ).assign_coords({REALIZATION: list(range(n_realizations))})

        if budget_steps is not None:
            return normalize_budget_grid(ds, budget_steps)

        return ds

    def to_dataloader(
        self,
        task: Task,
        optimizer: OptimizationStep,
        budget_steps: int | None = None,
    ) -> DataLoader:
        """Convert history to a DataLoader with task labels.

        Builds the xarray that
        :meth:`l2co.DataLoader.from_xarray` expects:

        - data variables (trajectory + ERT) carry dims
          ``(TASK_ID, SCHEDULE, REALIZATION, BUDGET, OUTPUT_DIM, ...)``
        - ``task_labels`` carries ``(TASK_ID, REALIZATION, FEATURE)``
          and is *not* duplicated across schedules, because the labels
          of a task are invariant under the optimizer used.

        Parameters
        ----------
        task : Task
            Task providing labels (via ``task.tag``) and global minimum.
        optimizer : OptimizationStep
            Optimizer used; its ``name`` becomes the singleton
            ``SCHEDULE`` coordinate on the data variables.
        budget_steps : int | None, optional
            Budget interpolation steps forwarded to
            :meth:`to_xarray_realizations`; ``None`` means no
            interpolation. Default is ``None``.

        Returns
        -------
        DataLoader
            DataLoader containing the optimization trajectories, ERT
            metrics, and per-task labels.
        """
        xarr = self.to_xarray_realizations(budget_steps=budget_steps)
        ert = calculate_ert(xarr, task.global_min)

        # Task labels are one row per task; broadcast across
        # realizations via expand_dims so xarray handles the tile.
        tag = {**task.tag, GLOBAL_MIN: task.global_min}
        task_labels = (
            xr.DataArray(
                list(tag.values()),
                coords={FEATURE: list(tag.keys())},
                dims=[FEATURE],
                name=TASK_LABELS,
            )
            .astype(str)
            .expand_dims({REALIZATION: xarr[REALIZATION].values})
        )

        # SCHEDULE is added before merging in the labels so it lands
        # only on data variables, not on the (schedule-invariant)
        # labels. TASK_ID is the outermost dim and is added last.
        data = xr.merge([xarr, ert], compat="override").expand_dims(
            {SCHEDULE: [optimizer.name]}
        )
        combined = xr.merge(
            [data, task_labels], compat="override"
        ).expand_dims({TASK_ID: [task.hash]})

        return DataLoader.from_xarray(combined)
