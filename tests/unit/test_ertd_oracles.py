"""The oracle references of :class:`DataLoader`, one per granularity.

All three share one rule -- most targets reached at the full budget, ties
to the fewer fevals -- and differ only in what one pick covers:

per-run virtual best solver (``vbs``)
    one realization of one task (``ertd_profile(include_best=True)``);
per-task virtual best solver (``vbs_task``)
    every run of one task, targets summed over the runs
    (``ertd_profile(include_task_best=True)``);
single best solver
    every run of every task (:meth:`DataLoader.single_best_schedule`).

The properties callers rely on:

one pick per task
    Each task's ``vbs_task`` curve is bit-identical to one portfolio
    schedule's curve (the bootstrap draws are shared across schedules).

pooled, not per-run
    A schedule that wins big on one run loses to one that wins steadily
    on all of them.

the pool is the bootstrap's pool
    With a ``realizations`` subset only those runs are pooled, so a
    single-run pool reduces ``vbs_task`` to ``vbs``.

the single best solver is not per-task
    It can name a schedule that is the per-task best on no task at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from l2co import DataLoader
from l2co._src.visualize.colors import display_name, get_schedule_styles

from .test_ertd_separability import _TASK_IDS, _loader

#                                                                      Fixtures
# =============================================================================

_TARGETS = np.array([1e-8, 1e-4, 1.0, 10.0])


def _explicit_loader(
    solved: dict[str, dict[str, list[int]]],
    fevals: dict[str, float] | None = None,
) -> DataLoader:
    """Loader where ``solved[s][t][r]`` = targets ``s`` hit on run r of t.

    Targets run hardest -> easiest, so hitting ``k`` targets means the
    easiest ``k``. Every cost of schedule ``s`` is ``fevals[s]`` (100 by
    default), so a tie-break is in play only where ``fevals`` says so.
    Every task is 2-D.
    """
    schedules = list(solved)
    tasks = list(solved[schedules[0]])
    n_real = len(solved[schedules[0]][tasks[0]])
    problem_index = pd.MultiIndex.from_product(
        [list(range(n_real)), tasks], names=["realization", "taskID"]
    )
    n_t, n_p = _TARGETS.size, len(problem_index)
    success = np.zeros((len(schedules), 1, n_t, n_p))
    cost = np.zeros((2, len(schedules), 1, n_t, n_p))
    for j, s in enumerate(schedules):
        cost[:, j] = (fevals or {}).get(s, 100.0)
        for p, (r, t) in enumerate(problem_index):
            k = solved[s][t][r]
            success[j, 0, n_t - k :, p] = 1.0

    data = xr.Dataset(
        {
            "success": (
                ("schedule", "output_dim", "target", "problem"),
                success,
            ),
            "cost": (
                ("metric", "schedule", "output_dim", "target", "problem"),
                cost,
            ),
        },
        coords={
            "schedule": schedules,
            "output_dim": ["y"],
            "target": _TARGETS,
            "metric": ["fevals", "iterations"],
            "problem": problem_index,
        },
    )
    labels = np.array([["sphere", 2]] * n_p, dtype=object)
    task_labels = xr.DataArray(
        labels,
        dims=["problem", "feature"],
        coords={
            "problem": problem_index,
            "feature": ["fn_name", "dimensionality"],
        },
    )
    return DataLoader(data=data, task_labels=task_labels)


def _fevals(prof: xr.DataArray) -> xr.DataArray:
    """Metric-selected profile, sorted by problem."""
    return prof.sel({"metric": "fevals"}).sortby(["problem"])


#                                                  Per-task virtual best solver
# =============================================================================


def test_steady_winner_beats_one_lucky_run() -> None:
    """``vbs_task`` pools over runs; ``vbs`` takes the lucky run's winner.

    ``lucky`` hits all 4 targets on run 0 and none after (4 in total);
    ``steady`` hits 2 on every run (8 in total).
    """
    d = _explicit_loader(
        {"lucky": {"t": [4, 0, 0, 0]}, "steady": {"t": [2, 2, 2, 2]}}
    )
    prof = _fevals(d.ertd_profile(include_best=True, include_task_best=True))

    vbs_task = prof.sel({"schedule": "vbs_task"})
    steady = prof.sel({"schedule": "steady"})
    assert np.array_equal(vbs_task.values, steady.values)
    # The per-run oracle does better than always running `steady`.
    vbs_auc = float(prof.sel({"schedule": "vbs"}).mean())
    assert vbs_auc > float(vbs_task.mean())


def test_vbs_task_is_one_schedule_per_task() -> None:
    """Each task's ``vbs_task`` rows are one portfolio schedule's rows.

    Checked on the rows the ``valid_files`` mask leaves alone --
    ``vbs_task`` carries no mask of its own, as ``vbs`` doesn't.
    """
    prof = _fevals(_loader(_TASK_IDS).ertd_profile(include_task_best=True))
    for task_id in _TASK_IDS:
        rows = prof.isel({"problem": prof["taskID"].values == task_id})
        pick = rows.sel({"schedule": "vbs_task"})
        # Identical over the task's runs: the profile has no realization
        # of its own, and the pick does not vary by run.
        assert (pick == pick.isel({"problem": 0})).all()
        matches = [
            s
            for s in rows["schedule"].values
            if s != "vbs_task"
            and np.array_equal(
                rows.sel({"schedule": s}).values, pick.values, equal_nan=True
            )
        ]
        assert matches, f"vbs_task on {task_id} matches no schedule's curve"


def test_vbs_task_differs_from_vbs_on_the_shared_fixture() -> None:
    """Guard: the shared fixture has tasks where the two oracles split.

    Without it the tests above could pass with ``vbs_task`` quietly equal
    to ``vbs``. No ordering is asserted: both oracles pick at the full
    budget, so neither bounds the other's area under the curve.
    """
    prof = _fevals(
        _loader(_TASK_IDS).ertd_profile(
            include_best=True, include_task_best=True
        )
    )
    assert not np.array_equal(
        prof.sel({"schedule": "vbs"}).values,
        prof.sel({"schedule": "vbs_task"}).values,
    )


@pytest.mark.parametrize("run", [0, 1, 2, 3])
def test_single_run_pool_reduces_vbs_task_to_vbs(run: int) -> None:
    """With one run in the pool, the per-task and per-run picks agree.

    Would fail if ``vbs_task`` pooled over every run regardless of
    ``realizations``: the full-pool pick differs from the single-run one
    on some task for every ``run`` of the shared fixture.
    """
    prof = _fevals(
        _loader(_TASK_IDS).ertd_profile(
            include_best=True, include_task_best=True, realizations=[run]
        )
    )
    np.testing.assert_array_equal(
        prof.sel({"schedule": "vbs_task"}).values,
        prof.sel({"schedule": "vbs"}).values,
    )


def test_include_task_best_leaves_the_portfolio_untouched() -> None:
    """Adding ``vbs_task`` changes no other schedule's rows."""
    base = _fevals(_loader(_TASK_IDS).ertd_profile(include_best=True))
    with_task = _fevals(
        _loader(_TASK_IDS).ertd_profile(
            include_best=True, include_task_best=True
        )
    )
    assert list(with_task["schedule"].values) == [
        *base["schedule"].values,
        "vbs_task",
    ]
    xr.testing.assert_identical(
        with_task.sel({"schedule": list(base["schedule"].values)}), base
    )


#                                                            Single best solver
# =============================================================================


def test_single_best_can_be_the_per_task_best_nowhere() -> None:
    """The SBS sums over tasks, so a runner-up everywhere can win it.

    ``a`` and ``b`` each dominate one task and fail the other (4 targets
    per run on one, 0 on the other: 8 in total); ``c`` is second on both
    (3 per run: 12 in total).
    """
    d = _explicit_loader(
        {
            "a": {"t0": [4, 4], "t1": [0, 0]},
            "b": {"t0": [0, 0], "t1": [4, 4]},
            "c": {"t0": [3, 3], "t1": [3, 3]},
        }
    )
    assert d.single_best_schedule() == "c"

    prof = _fevals(d.ertd_profile(include_task_best=True))
    for task_id, best in [("t0", "a"), ("t1", "b")]:
        rows = prof.isel({"problem": prof["taskID"].values == task_id})
        np.testing.assert_array_equal(
            rows.sel({"schedule": "vbs_task"}).values,
            rows.sel({"schedule": best}).values,
        )


def test_single_best_tie_goes_to_the_faster_schedule() -> None:
    """Equal target totals: the fewer mean fevals wins, in either order."""
    solved = {"slow": {"t0": [2, 2]}, "fast": {"t0": [2, 2]}}
    fevals = {"slow": 500.0, "fast": 50.0}
    assert _explicit_loader(solved, fevals).single_best_schedule() == "fast"
    flipped = {"fast": solved["fast"], "slow": solved["slow"]}
    assert _explicit_loader(flipped, fevals).single_best_schedule() == "fast"


def test_single_best_pools_only_the_given_runs() -> None:
    """``realizations`` narrows the pool, as for ``vbs_task``.

    Over all three runs ``lucky`` totals 4 and ``steady`` 3; over runs 1
    and 2 only, ``lucky`` totals 0.
    """
    d = _explicit_loader(
        {"lucky": {"t0": [4, 0, 0]}, "steady": {"t0": [1, 1, 1]}}
    )
    assert d.single_best_schedule() == "lucky"
    assert d.single_best_schedule(realizations=[1, 2]) == "steady"


def test_single_best_is_a_portfolio_schedule() -> None:
    """On the shared fixture: a real schedule, never an oracle name."""
    d = _loader(_TASK_IDS)
    assert d.single_best_schedule() in list(d.data["schedule"].values)


@pytest.mark.parametrize("shard_size", [1, 2, 4])
def test_single_best_scores_add_up_over_task_shards(shard_size: int) -> None:
    """Summed per-shard scores are the whole selection's, pick included.

    What lets a sharded caller pick the SBS without reading the whole
    selection again. ``shard_size=4`` leaves a ragged last shard.
    """
    whole = _loader(_TASK_IDS).single_best_scores()
    shards = [
        _loader(_TASK_IDS[i : i + shard_size]).single_best_scores()
        for i in range(0, len(_TASK_IDS), shard_size)
    ]
    summed = xr.concat(shards, dim="shard").sum("shard")
    xr.testing.assert_allclose(summed, whole)
    assert DataLoader.pick_single_best(summed) == (
        _loader(_TASK_IDS).single_best_schedule()
    )


#                                                                  Plot styling
# =============================================================================


@pytest.mark.parametrize("by", ["optimizer", "category"])
def test_the_three_oracles_draw_distinct_black_lines(by: str) -> None:
    """Each oracle keeps its own dash pattern, in either colouring mode.

    ``vbs_task`` must not collapse onto base ``vbs``'s style through the
    ``split("_")[0]`` fallback.
    """
    styles = get_schedule_styles(["sbs", "vbs", "vbs_task", "cmaes"], by=by)
    oracle = [styles[s] for s in ("sbs", "vbs", "vbs_task")]
    assert {st.color for st in oracle} == {"#000000"}
    assert len({st.linestyle for st in oracle}) == 3


def test_oracle_labels_name_their_granularity() -> None:
    assert display_name("sbs") == "SBS"
    assert display_name("vbs") == "VBS (per run)"
    assert display_name("vbs_task") == "VBS (per task)"
