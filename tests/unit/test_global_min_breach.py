"""A loss below ``global_min`` scores as reaching it, and is found.

An empirical ``global_min`` (e.g. CUTEst's) can sit above the true
minimum, so a run can beat it. Every place that measures a loss against
``global_min`` floors the gap at 0 instead of going negative (and NaN
under a log), and :func:`global_min_breaches` finds the schedules that
beat it so the dump can record them. The dump's record itself is tested
in ``test_databank.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from l2co._src.data.report_helpers import EPSILON, shifted_final_min
from l2co._src.performance import (
    calculate_ert,
    global_min_breaches,
    min_custom,
    shift_to_quality_value,
)
from l2co._src.tree_utils import log_stabilized_minmax_scale_with_global_min
from l2co.naming import (
    BUDGET,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OUTPUT_DIM,
    OUTPUT_MIN,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
)


def _task_data(output_min: np.ndarray) -> xr.Dataset:
    """One task: ``output_min`` over (schedule, realization, budget)."""
    n_sched, n_real, n_budget = output_min.shape
    counters = np.broadcast_to(
        np.arange(1, n_budget + 1), output_min.shape
    ).astype(int)
    dims = (SCHEDULE, REALIZATION, BUDGET)
    return xr.Dataset(
        {
            OUTPUT_MIN: (dims, output_min),
            FEVALS: (dims, counters),
            ITERATIONS: (dims, counters),
        },
        coords={
            SCHEDULE: [f"s{i}" for i in range(n_sched)],
            REALIZATION: np.arange(n_real),
            BUDGET: np.arange(n_budget),
        },
    )


#                                                               quality value
# =============================================================================


def test_shift_to_quality_value_floors_at_zero_and_keeps_nan():
    data = _task_data(np.array([[[3.0, 1.0, 0.5, np.nan]]]))
    qv = shift_to_quality_value(data, g_min=1.0)[OUTPUT_MIN].values
    np.testing.assert_array_equal(qv, [[[2.0, 0.0, 0.0, np.nan]]])


def test_calculate_ert_scores_a_breach_as_solved():
    # s0 beats global_min = 1.0; s1 stops short of it.
    data = _task_data(np.array([[[5.0, 2.0, 0.5]], [[5.0, 3.0, 2.0]]]))
    targets = np.array([1e-8, 1e-2, 1.0])
    perf = calculate_ert(data, targets=targets, global_min=1.0)
    success = perf[SUCCESS].sel({REALIZATION: 0})
    assert success.sel({SCHEDULE: "s0"}).values.all()
    assert success.sel({SCHEDULE: "s1"}).values.tolist() == [
        False,
        False,
        True,
    ]


def test_min_custom_stays_finite_on_a_breach():
    # 4 below global_min: deeper than min_custom's log offset (the
    # smallest positive gap, 1), so an unfloored gap would log a negative.
    data = _task_data(np.array([[[5.0, 2.0, -3.0]]]))
    ds = xr.Dataset({OUTPUT_MIN: data[OUTPUT_MIN], GLOBAL_MIN: 1.0})
    scaled = min_custom(ds).values
    assert np.isfinite(scaled).all()
    assert scaled.min() == pytest.approx(0.0)


def test_log_scale_with_global_min_maps_a_breach_to_zero():
    x = jnp.array([10.0, 1.0, 0.5, -3.0])
    y = log_stabilized_minmax_scale_with_global_min(x, jnp.array(1.0))
    assert jnp.isfinite(y).all()
    assert y[2] == 0.0
    assert y[3] == 0.0


def test_shifted_final_min_floors_a_breach_at_epsilon():
    problem = pd.MultiIndex.from_product(
        [[0], ["t_a"]], names=[REALIZATION, "taskID"]
    )
    # s0 ends at -12 (below global_min -10), s1 at -9.
    data = xr.Dataset(
        {
            OUTPUT_MIN: (
                (SCHEDULE, OUTPUT_DIM, BUDGET, "problem"),
                np.array([[[[-7.0], [-12.0]]], [[[-5.0], [-9.0]]]]),
            )
        },
        coords={
            SCHEDULE: ["s0", "s1"],
            OUTPUT_DIM: ["y"],
            BUDGET: [0.0, 1.0],
            "problem": problem,
        },
    )
    labels = np.full((1, 1), -10.0, dtype=object)
    task_labels = xr.DataArray(
        labels,
        dims=["problem", "feature"],
        coords={"problem": problem, "feature": [GLOBAL_MIN]},
    )
    shifted = shifted_final_min(data=data, task_labels=task_labels)
    assert shifted.sel({SCHEDULE: "s0"}).item() == pytest.approx(EPSILON)
    assert shifted.sel({SCHEDULE: "s1"}).item() == pytest.approx(1.0 + EPSILON)


#                                                         global_min_breaches
# =============================================================================


def test_global_min_breaches_names_schedule_best_and_run():
    output_min = np.array(
        [
            [[5.0, 2.0], [5.0, 0.25]],  # s0: realization 1 reaches 0.25
            [[5.0, 1.0], [5.0, 1.5]],  # s1: reaches global_min exactly
        ]
    )
    breaches = global_min_breaches(_task_data(output_min), g_min=1.0)
    assert breaches == {"s0": {"best": 0.25, "margin": 0.75, "realization": 1}}


def test_global_min_breaches_ignores_float_noise_at_the_optimum():
    # 1e-7 below a global_min of 1.0 is inside the 1e-6 margin; 1e-5 isn't.
    output_min = np.array([[[1.0 - 1e-7]], [[1.0 - 1e-5]]])
    breaches = global_min_breaches(_task_data(output_min), g_min=1.0)
    assert list(breaches) == ["s1"]


def test_global_min_breaches_margin_scales_with_global_min():
    # rtol * |global_min| = 1e-3 here, so 5e-4 below it is noise.
    output_min = np.array([[[-1000.0005]], [[-1000.01]]])
    breaches = global_min_breaches(_task_data(output_min), g_min=-1000.0)
    assert list(breaches) == ["s1"]


def test_global_min_breaches_needs_a_global_min_and_data():
    output_min = np.array([[[np.nan, np.nan]], [[0.5, np.nan]]])
    data = _task_data(output_min)
    assert global_min_breaches(data, g_min=np.nan) == {}
    assert list(global_min_breaches(data, g_min=1.0)) == ["s1"]
