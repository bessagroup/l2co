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
import numpy as onp
import pandas as pd
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
    update_step : Float[Array, " n_iterations"]
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
    update_step: Float[Array, " n_iterations"]
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

        Parameters
        ----------
        budget_steps : int | None, optional
            Number of steps for budget interpolation. If None, no interpolation
            is performed, by default None.

        Returns
        -------
        xr.Dataset
            Dataset containing the history with appropriate coordinates.
        """
        fevals_cumsum = jnp.cumsum(self.fevals)
        iterations_cumsum = jnp.cumsum(self.iterations)

        output_min = xr.DataArray(self.output_min, dims=[BUDGET]).expand_dims(
            {OUTPUT_DIM: ["y"]}
        )

        output_mean = xr.DataArray(
            self.output_mean, dims=[BUDGET]
        ).expand_dims({OUTPUT_DIM: ["y"]})

        output_std = xr.DataArray(self.output_std, dims=[BUDGET]).expand_dims(
            {OUTPUT_DIM: ["y"]}
        )

        update_step = xr.DataArray(
            self.update_step, dims=[BUDGET]
        ).expand_dims({OUTPUT_DIM: ["y"]})

        fevals = xr.DataArray(fevals_cumsum, dims=[BUDGET]).expand_dims(
            {OUTPUT_DIM: ["y"]}
        )

        iterations = xr.DataArray(
            iterations_cumsum, dims=[BUDGET]
        ).expand_dims({OUTPUT_DIM: ["y"]})

        data_vars = {
            OUTPUT_MIN: output_min,
            OUTPUT_MEAN: output_mean,
            OUTPUT_STD: output_std,
            UPDATE_STEP: update_step,
            FEVALS: fevals,
            ITERATIONS: iterations,
        }

        ds = xr.Dataset(data_vars)

        if budget_steps is not None:
            ds = ds.assign_coords(
                {
                    BUDGET: onp.linspace(
                        0.0, 1.0, ds[BUDGET].size, dtype=onp.float32
                    )
                }
            )
            ds = ds.interp(
                {
                    BUDGET: onp.linspace(
                        0.0, 1.0, budget_steps, dtype=onp.float32
                    )
                },
                method="nearest",
            )
            n = budget_steps - 1
            step = 1.0 / (n - 1)
            arr_interp = onp.linspace(0.0, 1.0, n, dtype=onp.float32)
            new_budget_coord = onp.concatenate([[-step], arr_interp])

            return ds.assign_coords({BUDGET: new_budget_coord})

        return ds

    def to_xarray_realizations(
        self, budget_steps: int | None = None
    ) -> xr.Dataset:
        """Convert history state to xarray Dataset with realization dimension.

        Parameters
        ----------
        budget_steps : int | None, optional
            Number of steps for budget interpolation. If None, no interpolation
            is performed, by default None.

        Returns
        -------
        xr.Dataset
            Dataset containing histories for all realizations.
        """
        arrs = []
        n_realizations = self.output_min.shape[0]

        for realization in range(n_realizations):
            # Slice axis 0 uniformly: 2-D fields like ``output_min`` give
            # the row at ``realization``; the 0-D ``cursor`` (which becomes
            # 1-D under ``vmap``) gives the scalar for that realization.
            new_history: HistoryState = jax.tree.map(
                lambda x: x[realization],  # noqa
                self,
            )
            arrs.append(
                new_history.to_xarray(
                    budget_steps=budget_steps,
                )
            )

        return xr.concat(
            arrs,
            dim=xr.DataArray(
                range(n_realizations),
                dims=[REALIZATION],
            ),
        )

    def to_dataloader(
        self,
        task: Task,
        optimizer: OptimizationStep,
        budget_steps: int | None = None,
    ) -> DataLoader:
        """Convert history to a DataLoader with task labels.

        Parameters
        ----------
        task : Task
            Task providing labels and global minimum.
        optimizer : OptimizationStep
            Optimizer used, added as the schedule label.
        budget_steps : int | None, optional
            Budget interpolation steps, by default None.

        Returns
        -------
        DataLoader
            DataLoader containing the optimization
            trajectories and ERT metrics.
        """
        xarr = self.to_xarray_realizations(budget_steps=budget_steps)
        ert = calculate_ert(xarr, task.global_min)

        labels_data = pd.DataFrame(
            [
                {**task.tag, GLOBAL_MIN: task.global_min}
                for _ in xarr[REALIZATION]
            ],
        )
        xarr_task_labels = xr.DataArray(
            labels_data.values,
            coords={
                REALIZATION: xarr[REALIZATION],
                FEATURE: labels_data.columns,
            },
            dims=[REALIZATION, FEATURE],
            name=TASK_LABELS,
        ).astype(str)

        comb = xr.merge([xarr, ert], compat="override")
        comb = comb.expand_dims({SCHEDULE: [optimizer.name]})
        comb_labels = xr.merge([comb, xarr_task_labels], compat="override")
        comb_labels = comb_labels.expand_dims({TASK_ID: [task.hash]})
        return DataLoader.from_xarray(comb_labels)
