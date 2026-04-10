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
from .task import Task
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
    output statistics, and evaluation metrics across iterations.

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
    """

    output_min: Float[Array, " n_iterations"]
    output_mean: Float[Array, " n_iterations"]
    output_std: Float[Array, " n_iterations"]
    update_step: Float[Array, " n_iterations"]
    fevals: Int[Array, " n_iterations"]
    iterations: Int[Array, " n_iterations"]

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
            Initialized history state with NaN values.
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
        )

    def reset(self) -> HistoryState:
        """Reset the history state to initial NaN values."""
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
        )

    def add(self, other: HistoryState) -> HistoryState:
        """
        Append a multi-step HistoryState and drop the same number from the
        front.

        Parameters
        ----------
        other : HistoryState
            HistoryState to append.

        Returns
        -------
        HistoryState
            New HistoryState with appended values.
        """
        # Number of steps to append
        K = other.output_min.shape[0]

        def push(a, b):
            """Shift array a left by K and append b at the end."""
            N = a.shape[0]

            # 1. slice tail: a[K:]
            tail = jax.lax.dynamic_slice(a, (K,), (N - K,))

            # 2. create output buffer (length N)
            out = jnp.empty_like(a)

            # 3. write tail at the front
            out = jax.lax.dynamic_update_slice(out, tail, (0,))

            # 4. write new entries at the end
            out = jax.lax.dynamic_update_slice(out, b, (N - K,))

            return out

        return jax.tree.map(push, self, other)

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
            new_history: HistoryState = jax.tree.map(
                lambda x: x[realization, :],  # noqa
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
