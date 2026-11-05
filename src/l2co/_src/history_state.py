"""
Exports of a :class:`~l2co_optimizers.HistoryState`.

The buffer itself lives in ``l2co_optimizers`` with the run loop that
fills it (ADR 0017). Turning it into an xarray ``Dataset`` or a
:class:`~l2co.DataLoader` needs xarray, the ERT and the task labels, so
those exports stay here as free functions.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from l2co_optimizers import HistoryState, OptimizationStep
from l2co_tasks import Task

# Local
from l2co._src.data import DataLoader
from l2co._src.data_validation import calculate_ert
from l2co._src.naming import (
    BUDGET,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    OUTPUT_STD,
    REALIZATION,
    SCHEDULE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)
from l2co._src.validation import normalize_budget_grid

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "history_to_dataloader",
    "history_to_xarray",
    "history_to_xarray_realizations",
]

# =============================================================================


def _noise_free_series(
    history_state: HistoryState, wrap: Callable[[jax.Array], xr.DataArray]
) -> dict[str, xr.DataArray]:
    """``{OUTPUT_MIN_TRUE: ...}`` for a run on a noisy task, else ``{}``.

    A noise-free run (``output_min_true is None``) gets no variable at
    all, not a copy of ``output_min``: its export, and every file and
    store made from it, stays what it was (ADR 0024).
    """
    if history_state.output_min_true is None:
        return {}
    return {OUTPUT_MIN_TRUE: wrap(history_state.output_min_true)}


def history_to_xarray(
    history_state: HistoryState, budget_steps: int | None = None
) -> xr.Dataset:
    """Convert history state to an xarray Dataset.

    All ``n_iterations`` slots are written through, including the
    padding past ``cursor`` (NaN floats, zero ints). The Dataset
    does not carry ``cursor`` itself; callers that need the valid
    range must consult the source ``HistoryState`` directly.

    Parameters
    ----------
    history_state : HistoryState
        One trajectory's buffer (unbatched: ``output_min`` has shape
        ``(n_iterations,)``).
    budget_steps : int | None, optional
        Number of steps for budget interpolation. If None, no
        interpolation is performed, by default None. When set, the
        result is passed through :func:`normalize_budget_grid`,
        which requires ``budget_steps >= 3``.

    Returns
    -------
    xr.Dataset
        Dataset containing the history with appropriate
        coordinates. A run on a noisy task (``output_min_true`` set)
        also carries ``output_min_true``, the lowest noise-free loss
        per step, next to the observed ``output_min``.

    Notes
    -----
    ``fevals`` / ``iterations`` are stored as cumulative sums. This
    relies on the padding convention that slots past ``cursor`` are
    zero ints, so the cumulative curve plateaus past the valid
    range. If the padding sentinel ever changes, this needs to be
    revisited.
    """

    def _wrap(arr: jax.Array) -> xr.DataArray:
        return xr.DataArray(np.asarray(arr), dims=[BUDGET]).expand_dims(
            {OUTPUT_DIM: ["y"]}
        )

    fevals_cumsum = jnp.cumsum(history_state.fevals)
    iterations_cumsum = jnp.cumsum(history_state.iterations)

    ds = xr.Dataset(
        {
            OUTPUT_MIN: _wrap(history_state.output_min),
            OUTPUT_MEAN: _wrap(history_state.output_mean),
            OUTPUT_STD: _wrap(history_state.output_std),
            UPDATE_STEP: _wrap(history_state.update_step),
            FEVALS: _wrap(fevals_cumsum),
            ITERATIONS: _wrap(iterations_cumsum),
            **_noise_free_series(history_state, _wrap),
        }
    )

    if budget_steps is not None:
        return normalize_budget_grid(ds, budget_steps)

    return ds


def history_to_xarray_realizations(
    history_state: HistoryState, budget_steps: int | None = None
) -> xr.Dataset:
    """Convert history state to xarray Dataset with realization dimension.

    Parameters
    ----------
    history_state : HistoryState
        A realization-batched buffer, as ``batch_run`` returns it; see
        Notes.
    budget_steps : int | None, optional
        Number of steps for budget interpolation. If None, no
        interpolation is performed, by default None.

    Returns
    -------
    xr.Dataset
        Dataset containing histories for all realizations, with
        dims ``(REALIZATION, BUDGET, OUTPUT_DIM)``. A run on a noisy
        task also carries ``output_min_true``, as
        :func:`history_to_xarray`.

    Notes
    -----
    Expects ``history_state`` to carry a realization axis, as a
    ``batch_run`` / ``eqx.filter_vmap`` produces it:
    every field carries a leading ``n_realizations`` axis,
    including the originally-scalar ``cursor`` (now shape
    ``(n_realizations,)``). Calling this on a non-vmapped
    ``HistoryState`` will fail with a shape error.

    ``fevals`` / ``iterations`` are cumulatively summed along the
    ``BUDGET`` axis only (per realization), preserving the
    plateau-past-cursor behaviour of :func:`history_to_xarray`. The
    per-realization ``cursor`` values themselves are not exposed
    on the resulting Dataset; consult the source ``HistoryState``
    when the valid range is needed.
    """
    n_realizations = history_state.output_min.shape[0]

    def _wrap(arr: jax.Array) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(arr), dims=[REALIZATION, BUDGET]
        ).expand_dims({OUTPUT_DIM: ["y"]})

    fevals_cumsum = jnp.cumsum(history_state.fevals, axis=1)
    iterations_cumsum = jnp.cumsum(history_state.iterations, axis=1)

    ds = xr.Dataset(
        {
            OUTPUT_MIN: _wrap(history_state.output_min),
            OUTPUT_MEAN: _wrap(history_state.output_mean),
            OUTPUT_STD: _wrap(history_state.output_std),
            UPDATE_STEP: _wrap(history_state.update_step),
            FEVALS: _wrap(fevals_cumsum),
            ITERATIONS: _wrap(iterations_cumsum),
            **_noise_free_series(history_state, _wrap),
        }
    ).assign_coords({REALIZATION: list(range(n_realizations))})

    if budget_steps is not None:
        return normalize_budget_grid(ds, budget_steps)

    return ds


def history_to_dataloader(
    history_state: HistoryState,
    task: Task,
    optimizer: OptimizationStep,
    targets: np.ndarray | None = None,
    budget_steps: int | None = None,
) -> DataLoader:
    """Convert history to a DataLoader with task labels.

    Builds the merged xarray (loss trajectories + ERT metrics + task
    labels) that :meth:`l2co.DataLoader.from_xarray` expects:

    - data variables (trajectory + ERT) carry dims
      ``(TASK_ID, SCHEDULE, REALIZATION, BUDGET, OUTPUT_DIM, ...)``
    - ``task_labels`` carries ``(TASK_ID, REALIZATION, FEATURE)``
      and is *not* duplicated across schedules, because the labels
      of a task are invariant under the optimizer used.

    Parameters
    ----------
    history_state : HistoryState
        A realization-batched buffer; see
        :func:`history_to_xarray_realizations`.
    task : Task
        Task providing labels (via ``task.tag``) and global minimum.
    optimizer : OptimizationStep
        Optimizer used; its ``name`` becomes the singleton
        ``SCHEDULE`` coordinate on the data variables.
    targets : np.ndarray | None, optional
        1-D array of target quality values (in quality-value space,
        i.e. relative to ``global_min``). When ``None`` (default),
        ``_fallback_targets(N_TARGETS) == np.logspace(-8, 2, 51)`` is
        used. Callers that want randomsearch-derived targets should
        compute them via :func:`compute_task_targets` and pass them
        in explicitly.
    budget_steps : int | None, optional
        Budget interpolation steps forwarded to
        :func:`history_to_xarray_realizations`; ``None`` means no
        interpolation. Default is ``None``.

    Returns
    -------
    DataLoader
        DataLoader containing the optimization trajectories, ERT
        metrics, and per-task labels. On a noisy task the ERT metrics
        are those of the noise-free series (see
        :func:`~l2co._src.performance.scoring_loss`).
    """
    xarr = history_to_xarray_realizations(
        history_state, budget_steps=budget_steps
    )
    ert = calculate_ert(xarr, targets=targets, global_min=task.global_min)

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
    combined = xr.merge([data, task_labels], compat="override").expand_dims(
        {TASK_ID: [task.hash]}
    )

    return DataLoader.from_xarray(combined)
