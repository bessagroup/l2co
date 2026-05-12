"""Baseline metrics over optimization-trajectory data.

These helpers operate on the xarray ``Dataset`` produced by the experiments
pipeline (or ``DataLoader``). They are the reference baselines used to score
the L2CO models: SBS (Single Best Solver), VBS (Virtual Best Solver), and
``log_relative_improvement``. ``encode`` is a small utility for converting a
DataArray of categorical labels into integer codes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import xarray as xr

from ..naming import (
    BUDGET,
    OUTPUT_DIM,
    OUTPUT_MIN,
    PROBLEM,
    SCHEDULE,
    TASK_ID,
)
from ..performance import output_cummin

__all__ = [
    "calculate_best_solver",
    "calculate_sbs",
    "calculate_vbs",
    "encode",
    "log_relative_improvement",
]


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
