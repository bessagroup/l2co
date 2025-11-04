"""
Module for creating the history state.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from abc import abstractmethod

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp
import xarray as xr
from jaxtyping import PyTree

from ._typing import OptHistoryType
from .naming import (
    BUDGET,
    FEVALS,
    INPUT_DIM,
    INPUT_MEAN,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_STD,
    REALIZATION,
    UPDATE_STEP,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class HistoryState(eqx.Module):
    input_mean: PyTree  # shape: (n_iterations, dim)
    output_min: jax.Array  # shape: (n_iterations,)
    output_mean: jax.Array  # shape: (n_iterations,)
    output_std: jax.Array  # shape: (n_iterations,)
    update_step: jax.Array  # shape: (n_iterations,)
    fevals: jax.Array  # shape: (n_iterations,)

    @classmethod
    def init(cls, n_iterations: int, params: PyTree) -> HistoryState:
        nans_input = jax.tree.map(
            lambda p: jnp.full(
                (n_iterations, *p.shape), jnp.nan, dtype=p.dtype
            ),
            params,
        )
        nans_output = jnp.full((n_iterations,), jnp.nan)
        default_step_value = jnp.zeros((n_iterations,), dtype=jnp.uint32)
        init_fevals = jnp.zeros((n_iterations,), dtype=jnp.uint32)
        return HistoryState(
            input_mean=nans_input,
            output_min=nans_output,
            output_mean=nans_output,
            output_std=nans_output,
            update_step=default_step_value,
            fevals=init_fevals,
        )

    def add(self, history: OptHistoryType) -> HistoryState:
        # Compute statistics across P dimension for each time step
        i_mean = jax.tree.map(lambda x: jnp.mean(x, axis=1), history.params)
        o_min = jnp.min(history.loss, axis=1)  # shape (T,)
        o_mean = jnp.mean(history.loss, axis=1)  # shape (T,)
        o_std = jnp.std(history.loss, axis=1)  # shape (T,)
        u_step = history.update_step[:, 0].astype(jnp.uint32)
        u_feval = history.fevals[:, 0].astype(jnp.uint32)

        def roll_concat(old: jax.Array, new: jax.Array) -> jax.Array:
            combined = jnp.concatenate([old, new], axis=0)
            return combined[-old.shape[0] :]  # keep last n_iterations

        return HistoryState(
            input_mean=jax.tree.map(roll_concat, self.input_mean, i_mean),
            output_min=roll_concat(self.output_min, o_min),
            output_mean=roll_concat(self.output_mean, o_mean),
            output_std=roll_concat(self.output_std, o_std),
            update_step=roll_concat(self.update_step, u_step),
            fevals=roll_concat(self.fevals, u_feval),
        )

    def to_xarray(
        self, include_input_mean: bool = True, budget_steps: int | None = None
    ) -> xr.Dataset:
        fevals_cumsum = jnp.cumsum(self.fevals)

        if include_input_mean:
            input_mean = xr.DataArray(
                self.input_mean, dims=[BUDGET, INPUT_DIM]
            )

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

        data_vars = {
            OUTPUT_MIN: output_min,
            OUTPUT_MEAN: output_mean,
            OUTPUT_STD: output_std,
            UPDATE_STEP: update_step,
            FEVALS: fevals,
        }
        if include_input_mean:
            data_vars[INPUT_MEAN] = input_mean

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
        self, include_input_mean: bool = True, budget_steps: int | None = None
    ) -> xr.Dataset:
        arrs = []
        n_realizations = self.output_min.shape[0]

        for realization in range(n_realizations):
            new_history: HistoryState = jax.tree.map(
                lambda x: x[realization, :],  # noqa
                self,
            )
            arrs.append(
                new_history.to_xarray(
                    include_input_mean=include_input_mean,
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
