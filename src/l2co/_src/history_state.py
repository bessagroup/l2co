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

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class HistoryState(eqx.Module):
    output_min: jax.Array     # shape: (n_iterations,)
    output_mean: jax.Array    # shape: (n_iterations,)
    output_std: jax.Array     # shape: (n_iterations,)
    update_step: jax.Array    # shape: (n_iterations,)

    @classmethod
    def init(cls, n_iterations: int) -> HistoryState:
        nans = jnp.full((n_iterations,), jnp.nan)
        zeros = jnp.zeros((n_iterations,), dtype=jnp.float32)
        return HistoryState(
            output_min=nans,
            output_mean=nans,
            output_std=nans,
            update_step=zeros,
        )

    def add(self, history: dict[str, jax.Array]) -> HistoryState:
        # Compute statistics across P dimension for each time step
        o_min = jnp.min(history['loss'], axis=1)    # shape (T,)
        o_mean = jnp.mean(history['loss'], axis=1)  # shape (T,)
        o_std = jnp.std(history['loss'], axis=1)    # shape (T,)
        u_step = jnp.mean(history['update_step'], axis=1)         # shape (T,)

        def roll_concat(old: jax.Array, new: jax.Array) -> jax.Array:
            combined = jnp.concatenate([old, new], axis=0)
            return combined[-old.shape[0]:]  # keep last n_iterations

        return HistoryState(
            output_min=roll_concat(self.output_min, o_min),
            output_mean=roll_concat(self.output_mean, o_mean),
            output_std=roll_concat(self.output_std, o_std),
            update_step=roll_concat(self.update_step, u_step),
        )

    def to_xarray(self) -> xr.Dataset:
        budget_coord = xr.DataArray(
            jnp.arange(self.output_min.shape[0]), dims=['budget'])

        output_min = xr.DataArray(self.output_min, dims=['budget'], coords={
                                  'budget': budget_coord})
        output_mean = xr.DataArray(self.output_mean, dims=[
                                   'budget'], coords={'budget': budget_coord})
        output_std = xr.DataArray(self.output_std, dims=['budget'], coords={
                                  'budget': budget_coord})
        update_step = xr.DataArray(self.update_step, dims=[
                                   'budget'], coords={'budget': budget_coord})

        return xr.Dataset({
            'output_min': output_min,
            'output_mean': output_mean,
            'output_std': output_std,
            'update_step': update_step
        })

    def to_xarray_realizations(self) -> xr.Dataset:
        arrs = []

        for realization in range(self.output_min.shape[0]):
            new_history = jax.tree.map(
                lambda x: x[realization, :],
                self,
            )
            arrs.append(new_history.to_xarray())

        return xr.concat(arrs, dim='realization')
