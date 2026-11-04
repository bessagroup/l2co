"""Performance profiles: per-task expected runtime and its ratio curves.

What a performance profile computes is fixed by ``docs/adr/0021``. These
tests pin it at the three public seams: ``DataLoader.expected_runtime``
(the per-task cost table), ``l2co.data.performance_profile`` (the ratio
curves over any subset of that table) and
``l2co.plotting.plot_performance_profile``. Every expected value is
worked out by hand from the literal inputs.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from l2co import DataLoader  # noqa: E402
from l2co.data import performance_profile  # noqa: E402
from l2co.plotting import display_name, plot_performance_profile  # noqa: E402

#                                                                      Fixtures
# =============================================================================

_TASK_IDS = ["t0", "t1"]
_SCHEDULES = ["a", "b"]
_N_REALIZATIONS = 3

#: Stored target values, hardest first -- the order the databank keeps
#: (``compute_task_targets``): ladder position 1 is index 0.
_TARGETS = np.array([1e-8, 1e-3, 1.0])


def _loader(
    cells: dict[tuple[str, str], tuple[list[list[float]], list[list[float]]]],
) -> DataLoader:
    """Build a performance-only loader from literal per-run values.

    ``cells[(schedule, task)] = (success, fevals)``, each a list over
    runs of a list over targets (hardest first). A ``(schedule, task)``
    pair left out of ``cells`` has no data at all: NaN ``success`` and
    ``cost``, as a databank store leaves an optimizer that never ran on a
    task. ``iterations`` is ``fevals / 10`` throughout, so a test can tell
    which cost metric was read.
    """
    problem_index = pd.MultiIndex.from_product(
        [list(range(_N_REALIZATIONS)), _TASK_IDS],
        names=["realization", "taskID"],
    )
    problem = xr.Coordinates.from_pandas_multiindex(problem_index, "problem")
    n_s, n_t, n_p = len(_SCHEDULES), _TARGETS.size, len(problem_index)

    success = np.full((n_s, 1, n_t, n_p), np.nan)
    fevals = np.full((n_s, 1, n_t, n_p), np.nan)
    for (sched, tid), (s_runs, c_runs) in cells.items():
        j = _SCHEDULES.index(sched)
        for p, (real, ptid) in enumerate(problem_index):
            if ptid == tid:
                success[j, 0, :, p] = s_runs[real]
                fevals[j, 0, :, p] = c_runs[real]

    data = xr.Dataset(
        {
            "success": (
                ("schedule", "output_dim", "target", "problem"),
                success,
            ),
            "cost": (
                ("metric", "schedule", "output_dim", "target", "problem"),
                np.stack([fevals, fevals / 10.0]),
            ),
        },
        coords={
            "schedule": _SCHEDULES,
            "output_dim": ["y"],
            "target": np.arange(n_t),
            "metric": ["fevals", "iterations"],
        },
    ).assign_coords(problem)
    task_labels = xr.DataArray(
        np.full((n_p, 1), 2, dtype=object),
        dims=["problem", "feature"],
        coords={"feature": ["dimensionality"]},
    ).assign_coords(problem)
    return DataLoader(data=data, task_labels=task_labels)


def _ert(rows: dict[str, list[float]]) -> xr.DataArray:
    """A ``taskID`` x ``schedule`` runtime table from literal rows.

    ``rows[task] = [runtime of a, of b, of c]``; ``np.inf`` is unsolved,
    ``np.nan`` no data.
    """
    return xr.DataArray(
        np.array(list(rows.values()), dtype=float),
        dims=["taskID", "schedule"],
        coords={"taskID": list(rows), "schedule": ["a", "b", "c"]},
    )


#: Every run reaches every target, at the same cost on every target.
_ALL_SOLVED = ([[1, 1, 1]] * 3, [[5, 5, 5]] * 3)


#                                                             Expected runtime
# =============================================================================


def test_expected_runtime_counts_failed_runs_in_the_total_only() -> None:
    """Total evaluations over every run, divided by the successful runs.

    ``a`` on ``t0`` reaches the hardest target in runs 0 and 2, at 10 and
    30 evaluations; run 1 fails after 20. That is (10 + 20 + 30) / 2 = 30,
    not the mean of the successes (20) nor of all runs (20).
    """
    loader = _loader(
        {
            ("a", "t0"): (
                [[1, 1, 1], [0, 1, 1], [1, 1, 1]],
                [[10, 5, 1], [20, 5, 1], [30, 5, 1]],
            ),
            ("b", "t0"): _ALL_SOLVED,
            ("a", "t1"): _ALL_SOLVED,
            ("b", "t1"): _ALL_SOLVED,
        }
    )

    ert = loader.expected_runtime(positions=[1.0])

    assert ert.sel(taskID="t0", schedule="a", position=1.0) == 30.0


def test_expected_runtime_marks_no_data_nan_and_never_solved_infinite() -> (
    None
):
    """No data is NaN; data with no run reaching the target is infinite.

    ``a`` never ran on ``t1``. ``b`` ran on ``t0`` but no run reached the
    hardest target. The two must stay distinguishable: a performance
    profile drops the first and counts the second as unsolved. Neither
    may raise a division warning -- a full store has many of both.
    """
    loader = _loader(
        {
            ("a", "t0"): _ALL_SOLVED,
            ("b", "t0"): ([[0, 1, 1]] * 3, [[50, 5, 1]] * 3),
            ("b", "t1"): _ALL_SOLVED,
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ert = loader.expected_runtime(positions=[1.0])

    assert np.isnan(ert.sel(taskID="t1", schedule="a", position=1.0))
    assert np.isposinf(ert.sel(taskID="t0", schedule="b", position=1.0))


def test_positions_snap_to_the_nearest_stored_target_ties_to_harder() -> None:
    """Position p reads stored target (n - 1)(1 - p), hardest first.

    With three stored targets, positions 1, 0.5 and 0 land exactly on
    indices 0, 1 and 2. Positions 0.75 and 0.25 fall halfway (0.5 and
    1.5) and take the harder neighbour: indices 0 and 1. Each target has
    its own cost (100, 10, 1), so the runtime shows which one was read.
    """
    loader = _loader(
        {
            ("a", "t0"): ([[1, 1, 1]] * 3, [[100, 10, 1]] * 3),
            ("b", "t0"): _ALL_SOLVED,
            ("a", "t1"): _ALL_SOLVED,
            ("b", "t1"): _ALL_SOLVED,
        }
    )

    ert = loader.expected_runtime(positions=[1.0, 0.75, 0.5, 0.25, 0.0])
    cell = ert.sel(taskID="t0", schedule="a")

    np.testing.assert_array_equal(cell.values, [100, 100, 10, 10, 1])
    np.testing.assert_array_equal(cell["target"].values, [0, 0, 1, 1, 2])


def test_expected_runtime_reads_the_requested_cost_metric() -> None:
    """``cost_metric="iterations"`` reads iterations (fevals / 10 here)."""
    loader = _loader(
        {
            ("a", "t0"): ([[1, 1, 1]] * 3, [[100, 10, 1]] * 3),
            ("b", "t0"): _ALL_SOLVED,
            ("a", "t1"): _ALL_SOLVED,
            ("b", "t1"): _ALL_SOLVED,
        }
    )

    ert = loader.expected_runtime(positions=[1.0], cost_metric="iterations")

    assert ert.sel(taskID="t0", schedule="a", position=1.0) == 10.0


#                                                          Performance profile
# =============================================================================


def test_a_tie_for_cheapest_is_a_win_for_every_tied_optimizer() -> None:
    """``a`` ties ``b`` on t0 and ``c`` on t1, so wins both tasks.

    ``b`` and ``c`` each win one: 1, 1/2, 1/2.
    """
    profile = performance_profile(_ert({"t0": [10, 10, 20], "t1": [5, 50, 5]}))

    np.testing.assert_array_equal(
        profile["win_rate"].sel(schedule=["a", "b", "c"]).values,
        [1.0, 0.5, 0.5],
    )


def test_a_task_nobody_solves_stays_in_the_denominator() -> None:
    """t1 is unsolved by everyone: counted, and a failure for all.

    Over both tasks ``a`` and ``b`` solve one of two (1/2), ``c`` none;
    only ``a`` wins anything (t0). Dropping t1 would have given 1, 1, 0.
    """
    profile = performance_profile(
        _ert({"t0": [10, 20, np.inf], "t1": [np.inf, np.inf, np.inf]})
    )

    abc = {"schedule": ["a", "b", "c"]}
    np.testing.assert_array_equal(
        profile["solve_rate"].sel(abc).values, [0.5, 0.5, 0.0]
    )
    np.testing.assert_array_equal(
        profile["win_rate"].sel(abc).values, [0.5, 0.0, 0.0]
    )
    assert profile["n_tasks"] == 2
    assert profile["n_unsolved"] == 1


def test_a_task_missing_for_one_optimizer_is_dropped_for_all() -> None:
    """Only t0 has data for every optimizer, so the profile is over t0.

    ``b`` has no data on t1 and ``c`` none on t2. Both tasks leave the
    profile for everyone -- on t1 ``a`` and ``c`` would otherwise tie at
    5 -- and the missing counts name who caused it: a 0, b 1, c 1.
    """
    profile = performance_profile(
        _ert(
            {
                "t0": [10, 20, 40],
                "t1": [5, np.nan, 5],
                "t2": [8, 9, np.nan],
            }
        )
    )

    abc = {"schedule": ["a", "b", "c"]}
    assert profile["n_tasks"] == 1
    assert profile["n_dropped"] == 2
    np.testing.assert_array_equal(
        profile["n_missing"].sel(abc).values, [0, 1, 1]
    )
    np.testing.assert_array_equal(
        profile["win_rate"].sel(abc).values, [1.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        profile["solve_rate"].sel(abc).values, [1.0, 1.0, 1.0]
    )


def test_a_ratio_beyond_the_cap_is_solved_but_off_the_curve() -> None:
    """``c`` is 20 000 times slower than ``a`` on t0, past the 1e4 cap.

    It still solves both tasks (solve rate 1), but its curve stops at
    1/2 -- only t1, a three-way tie, is within the axis. ``b`` (3 times
    slower on t0) starts at 1/2 and reaches 1 by the axis end. Raising
    the cap to 1e5 brings ``c``'s solve back onto the curve.
    """
    ert = _ert({"t0": [10, 30, 200_000], "t1": [10, 10, 10]})
    abc = {"schedule": ["a", "b", "c"]}

    profile = performance_profile(ert)
    rho = profile["rho"].sel(abc)

    assert rho["ratio"].values[0] == 1.0
    assert rho["ratio"].values[-1] == 1e4
    np.testing.assert_array_equal(rho.isel(ratio=0).values, [1, 0.5, 0.5])
    np.testing.assert_array_equal(rho.isel(ratio=-1).values, [1, 1, 0.5])
    np.testing.assert_array_equal(
        profile["solve_rate"].sel(abc).values, [1, 1, 1]
    )

    wider = performance_profile(ert, max_ratio=1e5)
    np.testing.assert_array_equal(
        wider["rho"].sel(abc).isel(ratio=-1).values, [1, 1, 1]
    )


#                                                                     Plotting
# =============================================================================


@pytest.mark.parametrize("right_labels", [False, True])
@pytest.mark.parametrize("style_by", ["optimizer", "category"])
def test_plot_draws_one_curve_per_schedule_and_an_infinity_point(
    style_by: str, right_labels: bool
) -> None:
    """Smoke test: a log ratio axis, every schedule drawn, an ∞ tick."""
    schedules = ["adam", "cmaes", "lbfgs"]
    ert = xr.DataArray(
        [[10.0, 30.0, 200_000.0], [10.0, 10.0, np.inf]],
        dims=["taskID", "schedule"],
        coords={"taskID": ["t0", "t1"], "schedule": schedules},
    )
    fig, ax = plt.subplots()

    plot_performance_profile(
        performance_profile(ert),
        ax=ax,
        style_by=style_by,
        right_labels=right_labels,
    )
    fig.canvas.draw()

    assert ax.get_xscale() == "log"
    labels = {line.get_label() for line in ax.get_lines()}
    assert {display_name(s) for s in schedules} <= labels
    assert "∞" in [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)
