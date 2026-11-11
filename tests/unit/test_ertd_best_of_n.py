"""The best-of-N random-schedule reference, ``random_best_of_N``.

A tractable, explicitly optimistic stand-in for the (intractable) best
dynamic schedule: per task, the one run of a source schedule -- the
uniform-random schedule, in practice -- whose own ERTD curve has the
largest area, scored as if it were a schedule of its own.

The properties callers rely on:

one picked run per task
    The picked run is copied into every realization slot of its task,
    so the restart bootstrap only ever sees that run again: its expected
    runtime is the run's own cost.

at least as good as every run
    Per task, the area under the reference's curve is no smaller than
    that of any single run of the source schedule.

N = 1 is the source schedule
    With a single run there is nothing to pick.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from .test_ertd_oracles import _explicit_loader
from .test_ertd_separability import _N_REALIZATIONS, _TASK_IDS, _loader

_NAME = "random_best_of_N"


def _runs(d, schedule: str, task: str) -> xr.Dataset:
    """One schedule's runs on one task, indexed by realization."""
    data = d.data.sel({"schedule": schedule}, drop=True)
    return data.unstack("problem").sel({"taskID": task}, drop=True)


def _task_area(prof: xr.DataArray, schedule: str) -> xr.DataArray:
    """Per task, the area under one schedule's fevals curve."""
    curve = prof.sel({"metric": "fevals", "schedule": schedule}, drop=True)
    return curve.mean("cost_per_dim").groupby("taskID").mean()


#                                                     Building the reference
# =============================================================================


def test_the_best_run_is_copied_into_every_slot() -> None:
    """Per task, the run reaching the most targets fills every slot.

    Every cost is equal, so the run with the largest curve area is the
    one with the most targets: run 1 on ``t0``, run 0 on ``t1``.
    """
    d = _explicit_loader(
        {
            "random": {"t0": [1, 4, 2], "t1": [3, 0, 1]},
            "other": {"t0": [0, 0, 0], "t1": [4, 4, 4]},
        }
    )
    out = d.create_best_run_strategy("random")

    for task, best in [("t0", 1), ("t1", 0)]:
        picked = _runs(out, _NAME, task)
        source = _runs(d, "random", task).sel({"realization": best})
        for r in picked["realization"].values:
            xr.testing.assert_equal(
                picked.sel({"realization": r}).drop_vars("realization"),
                source.drop_vars("realization"),
            )


def test_a_tied_area_goes_to_the_run_with_more_targets() -> None:
    """Equal areas: the run that reached more targets is picked.

    Every run costs more per dimension than the whole cost grid, so every
    area is zero; run 1 reached the most targets.
    """
    d = _explicit_loader({"random": {"t0": [1, 3, 2]}}, {"random": 1e9})
    picked = _runs(d.create_best_run_strategy("random"), _NAME, "t0")
    source = _runs(d, "random", "t0").sel({"realization": 1})
    xr.testing.assert_equal(
        picked.isel({"realization": 0}).drop_vars("realization"),
        source.drop_vars("realization"),
    )


#                                                        In the ERTD profile
# =============================================================================


@pytest.mark.parametrize("source", ["randomsearch", "good_opt"])
def test_the_reference_is_no_worse_than_any_single_run(source: str) -> None:
    """Per task, its area is at least that of each run on its own.

    A run on its own is the source schedule's curve with a single-run
    restart pool.
    """
    d = _loader(_TASK_IDS)
    best = _task_area(d.ertd_profile(include_random_best=source), _NAME)
    for run in range(_N_REALIZATIONS):
        alone = _task_area(d.ertd_profile(realizations=[run]), source)
        assert (best.fillna(0.0) >= alone.fillna(0.0)).all(), f"run {run}"


@pytest.mark.parametrize("run", range(_N_REALIZATIONS))
def test_a_single_run_pool_picks_that_run(run: int) -> None:
    """Only the runs in ``realizations`` are candidates.

    So disjoint halves stay independent estimates. Would fail if the
    pick ranged over every run: the full-pool pick differs from ``run``
    on some task for every ``run``.
    """
    prof = _loader(_TASK_IDS).ertd_profile(
        include_random_best="good_opt", realizations=[run]
    )
    xr.testing.assert_equal(
        _task_area(prof, _NAME), _task_area(prof, "good_opt")
    )


def test_with_one_run_the_reference_is_the_source_schedule() -> None:
    """N = 1: the lone run is the pick, curve for curve."""
    d = _explicit_loader(
        {"random": {"t0": [2], "t1": [0]}, "other": {"t0": [4], "t1": [1]}},
        {"random": 30.0, "other": 300.0},
    )
    prof = d.ertd_profile(include_random_best="random")
    np.testing.assert_array_equal(
        prof.sel({"schedule": _NAME}).values,
        prof.sel({"schedule": "random"}).values,
    )


def test_the_reference_leaves_the_portfolio_untouched() -> None:
    """Appended last; every other row as without it, on the same grid."""
    d = _loader(_TASK_IDS)
    base = d.ertd_profile(include_task_best=True)
    with_ref = d.ertd_profile(
        include_task_best=True, include_random_best="randomsearch"
    )
    assert list(with_ref["schedule"].values) == [
        *base["schedule"].values,
        _NAME,
    ]
    xr.testing.assert_identical(
        with_ref.sel({"schedule": list(base["schedule"].values)}), base
    )
    assert with_ref["problem"].size == d.valid_files["problem"].size


#                                                      Runs that are holes
# =============================================================================


def _with_holes(d, schedule: str, holes: list[tuple[int, str]]):
    """``d`` with the given ``(realization, task)`` runs made NaN holes.

    The shape a timed-out run cell takes after the databank dump.
    """
    hole = xr.DataArray(
        [(r, t) in holes for r, t in d.data.indexes["problem"]],
        dims=["problem"],
        coords={"problem": d.data["problem"]},
    ) & (d.data["schedule"] == schedule)
    return replace(d, data=d.data.where(~hole))


def test_a_task_of_holes_is_masked_like_its_source() -> None:
    """Every run of a task a hole: the reference is masked there too.

    The source schedule's curve is NaN on such a task, so the reference
    must not count it as a solved-nothing task the source leaves out.
    """
    d = _with_holes(
        _explicit_loader({"random": {"t0": [1, 4], "t1": [3, 0]}}),
        "random",
        [(0, "t0"), (1, "t0")],
    )
    prof = d.ertd_profile(include_random_best="random")
    ref = prof.sel({"schedule": _NAME}, drop=True)
    src = prof.sel({"schedule": "random"}, drop=True)
    xr.testing.assert_equal(ref.isnull(), src.isnull())
    assert ref.sel({"taskID": "t0"}).isnull().all()
    assert ref.sel({"taskID": "t1"}).notnull().all()


def test_a_tie_never_goes_to_a_hole() -> None:
    """A hole ties a run that reached nothing; the real run is picked.

    Run 0 is a hole and run 1 reached no target: both have zero area and
    zero targets, and the lower realization would pick the hole.
    """
    d = _with_holes(
        _explicit_loader({"random": {"t0": [2, 0]}}), "random", [(0, "t0")]
    )
    prof = d.ertd_profile(include_random_best="random")
    assert prof.sel({"schedule": _NAME}).notnull().all()
