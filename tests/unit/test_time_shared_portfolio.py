"""The time-shared portfolio pseudo-schedule (l2co#115).

A portfolio of K stored schedules, interleaved in round-robin over
function evaluations with no state transfer, is built offline from the
members' stored histories: ``best_portfolio(F) = min_k best_k(w_k F)``.
These tests pin it at its public seam, ``l2co.data.time_shared_portfolio``,
and at the consumers that read the pseudo-schedule unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from l2co_optimizers import OptimizationStep

from l2co._src.performance import calculate_ert
from l2co.data import scoring_loss, time_shared_portfolio
from l2co.databank import load_databank
from l2co.naming import (
    BUDGET,
    COST,
    FEVALS,
    ITERATIONS,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
    TASK_ID,
)
from tests.unit.test_databank import FakeTask, _dump
from tests.unit.test_noise_free_scoring import NOISY_TASK, _write_runs

pytestmark = pytest.mark.filterwarnings("ignore")


#                                                                      Fixtures
# =============================================================================


def _schedule(name, output_min, fevals, iterations=None) -> xr.Dataset:
    """One stored schedule: ``(realization, budget)`` arrays, one row per run."""
    output_min = np.atleast_2d(np.asarray(output_min, dtype=float))
    fevals = np.broadcast_to(
        np.atleast_2d(np.asarray(fevals, dtype=float)), output_min.shape
    )
    if iterations is None:
        iterations = np.arange(1, output_min.shape[1] + 1)
    iterations = np.broadcast_to(
        np.atleast_2d(np.asarray(iterations, dtype=float)), output_min.shape
    )
    dims = [REALIZATION, BUDGET]
    return xr.Dataset(
        {
            OUTPUT_MIN: (dims, output_min),
            FEVALS: (dims, fevals),
            ITERATIONS: (dims, iterations),
        },
        coords={
            REALIZATION: np.arange(output_min.shape[0]),
            BUDGET: np.linspace(0.0, 1.0, output_min.shape[1]),
        },
    ).expand_dims({SCHEDULE: [name]})


def _store(*schedules: xr.Dataset) -> xr.Dataset:
    return xr.concat(schedules, dim=SCHEDULE)


#                                                                    Properties
# =============================================================================


def test_a_one_member_portfolio_is_the_member():
    ds = _store(
        _schedule("adam", [[5.0, 3.0, 3.0, 1.0]], [2, 4, 6, 8]),
        _schedule("lbfgs", [[9.0, 0.5, 0.1, 0.0]], [1, 5, 9, 13]),
    )

    pf = time_shared_portfolio(ds, ["adam"])

    assert list(pf[SCHEDULE].values) == ["portfolio(adam)"]
    member = ds.sel({SCHEDULE: ["adam"]})
    for var in (OUTPUT_MIN, FEVALS, ITERATIONS):
        np.testing.assert_array_equal(pf[var].values, member[var].values)


def _random_members(rng, k: int, n_runs: int = 6, n_budget: int = 12):
    """K members with random losses; each draws its own feval grid."""
    names = [f"m{j}" for j in range(k)]
    schedules = [
        _schedule(
            name,
            rng.lognormal(0.0, 2.0, (n_runs, n_budget)),
            np.cumsum(rng.integers(1, 20, n_budget)),
        )
        for name in names
    ]
    return _store(*schedules), names


def _hitting_fevals(run: xr.Dataset, target: float) -> float:
    """Fevals at which a run's best-so-far first reaches ``target``."""
    best = np.fmin.accumulate(run[OUTPUT_MIN].values)
    hit = np.flatnonzero(best <= target)
    return run[FEVALS].values[hit[0]] if hit.size else np.inf


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [2, 3, 4])
def test_hitting_time_is_at_most_k_times_the_fastest_member(seed, k):
    """On a common feval grid, every target the fastest member reaches
    after h fevals the portfolio reaches within K * h."""
    rng = np.random.default_rng(seed)
    grid = np.cumsum(rng.integers(1, 20, 12))
    names = [f"m{j}" for j in range(k)]
    ds = _store(
        *(_schedule(n, rng.lognormal(0.0, 2.0, (6, 12)), grid) for n in names)
    )

    pf = time_shared_portfolio(ds, names).squeeze(SCHEDULE)

    for r in ds[REALIZATION].values:
        for target in ds[OUTPUT_MIN].sel({REALIZATION: r}).values.ravel():
            fastest = min(
                _hitting_fevals(ds.sel({SCHEDULE: n, REALIZATION: r}), target)
                for n in names
            )
            got = _hitting_fevals(pf.sel({REALIZATION: r}), target)
            assert got <= k * fastest


@pytest.mark.parametrize("granularity", ["fevals", "iterations"])
@pytest.mark.parametrize("seed", range(5))
def test_best_so_far_and_counters_are_monotone(seed, granularity):
    ds, names = _random_members(np.random.default_rng(seed), k=3)

    pf = time_shared_portfolio(ds, names, granularity=granularity)

    steps = pf.diff(BUDGET)
    assert (steps[OUTPUT_MIN] <= 0).all()
    assert (steps[FEVALS] >= 0).all()
    assert (steps[ITERATIONS] >= 0).all()


#                                                                Worked examples
# =============================================================================


def _two_members() -> xr.Dataset:
    """Two schedules over two paired runs; b spends 2 fevals per step."""
    return _store(
        _schedule("a", [[8, 6, 4, 2], [7, 7, 7, 7]], [1, 2, 3, 4]),
        _schedule("b", [[9, 5, 1, 0.5], [3, 3, 0, 0]], [2, 4, 6, 8]),
    )


def test_equal_shares_interleave_the_paired_runs_over_fevals():
    # Step i spends F_i = 3, 6, 9, 12 fevals in all, F_i/2 per member;
    # a stops at its last point (4 fevals) and leaves its share unused.
    pf = time_shared_portfolio(_two_members(), ["a", "b"])

    assert list(pf[SCHEDULE].values) == ["portfolio(a+b)"]
    run = pf.squeeze(SCHEDULE)
    np.testing.assert_array_equal(
        run[OUTPUT_MIN].values, [[8, 4, 2, 1], [7, 3, 3, 0]]
    )
    np.testing.assert_array_equal(run[FEVALS].values[0], [3, 6, 8.5, 10])
    np.testing.assert_array_equal(
        run[ITERATIONS].values[0], [2.25, 4.5, 6.25, 7]
    )


def test_weights_set_each_members_share_of_the_fevals():
    # Weights 3:1 give a 3/4 and b 1/4 of F_i = 3, 6, 9, 12.
    pf = time_shared_portfolio(_two_members(), ["a", "b"], weights=[3, 1])

    run = pf.squeeze(SCHEDULE)
    np.testing.assert_array_equal(
        run[OUTPUT_MIN].values, [[6, 2, 2, 2], [7, 7, 3, 3]]
    )
    np.testing.assert_array_equal(run[FEVALS].values[0], [3, 5.5, 6.25, 7])


@pytest.mark.parametrize(
    "unwritten",
    [(OUTPUT_MIN, FEVALS, ITERATIONS), (OUTPUT_MIN,)],
    ids=["unwritten-cell", "no-loss"],
)
def test_a_run_missing_from_any_member_is_missing_from_the_portfolio(
    unwritten,
):
    # An unwritten store cell reads back NaN in every variable; a run
    # with counters but no loss is just as missing.
    ds = _two_members()
    for var in unwritten:
        ds[var].loc[{SCHEDULE: "b", REALIZATION: 1}] = np.nan

    run = time_shared_portfolio(ds, ["a", "b"]).squeeze(SCHEDULE)

    for var in (OUTPUT_MIN, FEVALS, ITERATIONS):
        assert run[var].sel({REALIZATION: 1}).isnull().all()
    np.testing.assert_array_equal(
        run[OUTPUT_MIN].sel({REALIZATION: 0}).values, [8, 4, 2, 1]
    )


@pytest.mark.parametrize("granularity", ["fevals", "iterations"])
def test_a_member_run_missing_its_last_point_spoils_only_that_step(
    granularity,
):
    # Member a's last stored point lacks its feval count: the portfolio
    # knows its own fevals at every step before it.
    ds = _store(
        _schedule("a", [[8.0, 4.0, 2.0, 1.0]], [2, 4, 6, np.nan]),
        _schedule("b", [[9.0, 5.0, 3.0, 2.0]], [2, 4, 6, 8]),
    )

    run = time_shared_portfolio(ds, ["a", "b"], granularity=granularity)
    run = run.squeeze([SCHEDULE, REALIZATION])

    np.testing.assert_array_equal(run[FEVALS].values, [4, 8, 12, np.nan])
    np.testing.assert_array_equal(run[ITERATIONS].values, [2, 4, 6, np.nan])


def test_a_lazy_store_chunked_unevenly_across_variables_stays_lazy():
    eager = _two_members()
    lazy = eager.chunk({REALIZATION: 1})
    lazy[FEVALS] = lazy[FEVALS].chunk({REALIZATION: 2})

    run = time_shared_portfolio(lazy, ["a", "b"])

    assert run[OUTPUT_MIN].chunks is not None
    xr.testing.assert_equal(
        run.compute(), time_shared_portfolio(eager, ["a", "b"])
    )


def test_a_noisy_portfolio_is_scored_on_its_noise_free_series():
    # The noise-free series is _two_members' loss; the observed one dips
    # a thousandfold below it, as a noisy minimum over a population can.
    clean = _two_members()
    noisy = clean.assign(
        {
            OUTPUT_MIN: clean[OUTPUT_MIN] * 1e-3,
            OUTPUT_MIN_TRUE: clean[OUTPUT_MIN],
        }
    )

    run = time_shared_portfolio(noisy, ["a", "b"]).squeeze(SCHEDULE)

    want = np.array([[8, 4, 2, 1], [7, 3, 3, 0]])
    np.testing.assert_array_equal(scoring_loss(run).values, want)
    np.testing.assert_allclose(run[OUTPUT_MIN].values, want * 1e-3)
    assert OUTPUT_MIN_TRUE not in time_shared_portfolio(clean, ["a", "b"])


def test_the_portfolio_is_marked_as_derived_and_keeps_the_run_labels():
    ds = _two_members()
    ds = ds.assign(
        task_labels=(REALIZATION, ["r0", "r1"]),
        output_mean=ds[OUTPUT_MIN] * 2,
    ).assign_coords(global_min=0.0)

    pf = time_shared_portfolio(
        ds, ["a", "b"], weights=[3, 1], granularity="fevals"
    )

    assert pf.attrs == {
        "derived": "time_shared_portfolio",
        "portfolio_members": ["a", "b"],
        "portfolio_weights": [0.75, 0.25],
        "portfolio_granularity": "fevals",
    }
    xr.testing.assert_identical(pf["task_labels"], ds["task_labels"])
    assert float(pf["global_min"]) == 0.0
    # a per-schedule variable of the members has no portfolio counterpart
    assert "output_mean" not in pf


def test_whole_iteration_turns_whose_length_the_weights_scale():
    # Weights 1:2 give a one iteration per turn and b two, in member
    # order: a turn ends at 1, 5 | 6, 10 | 11 | 12 fevals in all. Each
    # step stops at the last turn that fits in F_i = 3, 6, 9, 12, so a
    # turn too long for what is left (b's at 9) is not started.
    pf = time_shared_portfolio(
        _two_members(), ["a", "b"], weights=[1, 2], granularity="iterations"
    )

    run = pf.squeeze(SCHEDULE)
    np.testing.assert_array_equal(
        run[OUTPUT_MIN].values, [[8, 5, 5, 0.5], [7, 3, 3, 0]]
    )
    np.testing.assert_array_equal(run[FEVALS].values[0], [1, 6, 6, 12])
    np.testing.assert_array_equal(run[ITERATIONS].values[0], [1, 4, 4, 8])


#                                                                    Consumers
# =============================================================================


def test_expected_running_time_reads_the_portfolio_as_a_schedule():
    ds = _two_members()
    both = xr.concat([ds, time_shared_portfolio(ds, ["a", "b"])], dim=SCHEDULE)

    ert = calculate_ert(both, targets=np.array([4.5, 1.5]), global_min=0.0)

    pf = ert.sel({SCHEDULE: "portfolio(a+b)"})
    assert pf[COST].dims == ert.sel({SCHEDULE: "a"})[COST].dims
    assert pf[SUCCESS].all()
    # run 0 reaches 4.5 at step 1 (6 fevals) and 1.5 at step 3 (10)
    np.testing.assert_array_equal(
        pf[COST].sel({"metric": FEVALS, REALIZATION: 0}).values, [6, 10]
    )


def test_portfolio_of_a_databank_read_scores_as_its_members(tmp_path):
    """A store with a noisy and a noise-free task, losses read back from
    float32 gaps to a nonzero f*: the portfolio of two members that ran
    alike reaches every target its members reach, at twice the fevals,
    and is scored on the series they are."""
    steps = [
        OptimizationStep(optimizer="randomsearch"),
        OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01}),
    ]
    shifted = FakeTask(
        hash="5h1f7d", fn_name="rastrigin", dimensionality=4, global_min=-3.0
    )
    runs = _write_runs(
        tmp_path / "runs", [(shifted, 7, False), (NOISY_TASK, 9, True)], steps
    )
    _dump(tmp_path, runs, mode="error")
    data = load_databank("bbob", steps, tmp_path).data
    names = [s.name for s in steps]

    pf = time_shared_portfolio(data, names)

    targets = np.logspace(-3, 1, 9)
    member = calculate_ert(data.sel({SCHEDULE: names[0]}), targets=targets)
    got = calculate_ert(pf.squeeze(SCHEDULE), targets=targets)
    member, got = member.drop_vars(SCHEDULE), got.drop_vars(SCHEDULE)
    for task in (shifted.hash, NOISY_TASK.hash):
        want = member.where(member[TASK_ID] == task, drop=True)
        run = got.where(got[TASK_ID] == task, drop=True)
        xr.testing.assert_equal(run[SUCCESS], want[SUCCESS])
        xr.testing.assert_equal(
            run[COST].sel({"metric": FEVALS}),
            2 * want[COST].sel({"metric": FEVALS}),
        )
