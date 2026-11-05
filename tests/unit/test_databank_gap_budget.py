"""The databank stores gaps, checks each cell's budget and cuts at it.

Three changes for the dataset paper's re-run (l2co ADR 0027; the spec's
decisions c, d and e):

* **The gap is a storage format.** The writer subtracts the task's f*
  (``global_min``) in float64 from ``output_min``, ``output_min_true`` and
  ``output_mean`` after the dump-time scores are computed, and stores the
  gap in float32. Every l2co reader adds f* back in float64, so callers
  keep getting losses -- now precise to float32 *of the gap*. Tasks
  without a finite f* are stored and returned raw. Each raw netCDF file
  carries ``f_star`` and ``value_kind`` attributes.
* **Each cell is checked against the task's evaluation budget B**, when
  the caller gives one: ``n == B`` for a cell costing one evaluation per
  iteration or a variable amount, ``n*lambda <= B < (n+1)*lambda`` for a
  fixed-cost cell. B is stored per task; an append refuses another B.
* **The cut at B**: ``calculate_ert`` ignores points past B and charges a
  missed run at most B. The breach check still reads every point.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from l2co import databank
from l2co._src.data_validation import _ert_from_quantile
from l2co._src.databank import layout
from l2co._src.databank.reader import (
    load_databank,
    load_databank_dataset,
    load_loss_raw,
)
from l2co._src.databank.records import BREACHES_ATTR, _build_task_records
from l2co._src.databank.verify import verify_databank
from l2co._src.databank.writer import (
    DumpPlan,
    _dump_task_runs,
    _dump_template_runs,
    dump_databank_finalize,
)
from l2co._src.performance import calculate_ert, draw_samples
from l2co.naming import (
    BUDGET,
    COST,
    EVALUATION_BUDGET,
    F_STAR,
    FEVALS,
    ITERATIONS,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_STD,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
    TARGET_VALUE,
    TASK_ID,
    VALUE_KIND,
)
from l2co.optimization import OptimizationStep

from .test_databank import (  # noqa: F401  (fixtures)
    FakeTask,
    _write_raw_runs,
    make_loss_ds,
    steps,
    tasks,
)

pytestmark = pytest.mark.filterwarnings("ignore")

NAME = "bbob"
BS = 16
F_STAR_VALUE = 100.0


def _dump(
    root: Path,
    runs,
    *,
    mode: str = "error",
    evaluation_budget=None,
    run_settings=None,
) -> Path:
    plan = _dump_template_runs(
        runs,
        root,
        NAME,
        mode=mode,
        easiest_quantile=0.75,
        taskset=None,
        budget_steps=BS,
        evaluation_budget=evaluation_budget,
        run_settings=run_settings,
    )
    return dump_databank_finalize(_dump_task_runs(runs, plan), plan)


def _snapshot(store: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(store)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(store.rglob("*"))
        if p.is_file()
    }


def _offset_runs(rundir: Path, task, steps, nb: int, offset: float):
    """Runs whose losses are ``offset`` plus :func:`make_loss_ds`'s."""
    rundir.mkdir(parents=True, exist_ok=True)
    step_paths = {}
    for s in steps:
        ds = make_loss_ds(nb, scale=10.0)
        for var in (OUTPUT_MIN, OUTPUT_MEAN):
            ds[var] = ds[var] + offset
        p = rundir / f"{task.hash}_{s.hash}.nc"
        ds.to_netcdf(p)
        step_paths[s] = p
    return [(task, step_paths)]


def _cell(n: int, increments, scale: float = 10.0) -> xr.Dataset:
    """A loss history of ``n`` iterations costing ``increments`` each.

    ``increments`` is one evaluation count per iteration (cycled); the
    ``fevals`` counter is their cumulative sum, as the run loop records
    it.
    """
    # A budget coordinate, as the run loop's histories have, so cells of
    # different lengths line up when a task's schedules are concatenated.
    ds = make_loss_ds(n, scale=scale).assign_coords({BUDGET: np.arange(n)})
    per_step = np.resize(np.asarray(increments, dtype=int), n)
    fevals = np.cumsum(per_step)
    ds[FEVALS] = ds[FEVALS].copy(
        data=np.broadcast_to(fevals[None, :, None], ds[FEVALS].shape).copy()
    )
    return ds


def _cell_runs(rundir: Path, task, cells: dict) -> list:
    rundir.mkdir(parents=True, exist_ok=True)
    step_paths = {}
    for step, ds in cells.items():
        p = rundir / f"{task.hash}_{step.hash}.nc"
        ds.to_netcdf(p)
        step_paths[step] = p
    return [(task, step_paths)]


RS = OptimizationStep(optimizer="randomsearch")
CMA = OptimizationStep(optimizer="cmaes", hyperparameters={"popsize": 6})
LBFGS = OptimizationStep(optimizer="lbfgs")


#                                                      c. the gap is storage
# =============================================================================


def test_the_store_holds_the_gap_and_readers_return_the_loss(tmp_path, steps):
    task = FakeTask("f1", "sphere", 10, global_min=F_STAR_VALUE)
    runs = _offset_runs(tmp_path / "runs", task, steps, 20, F_STAR_VALUE)
    root = tmp_path / "db"
    _dump(root, runs)

    want = make_loss_ds(20, scale=10.0)  # the gap, exactly
    stored = xr.open_zarr(layout.zarr_store_path(root, NAME))
    assert stored[F_STAR].dtype == np.float64
    assert float(stored[F_STAR].sel({TASK_ID: "f1"})) == F_STAR_VALUE
    gap = stored[OUTPUT_MIN].sel({TASK_ID: "f1"}).isel({SCHEDULE: 0})
    # The stored gap ends at the last raw value (the grid keeps it).
    np.testing.assert_allclose(
        float(gap.isel({BUDGET: -1}).max()),
        float(want[OUTPUT_MIN].isel({BUDGET: -1}).max()),
        rtol=1e-6,
    )

    ds = load_databank_dataset(NAME, [s.name for s in steps], root)
    loss = ds[OUTPUT_MIN].isel({SCHEDULE: 0}).squeeze(drop=True)
    assert loss.dtype == np.float64
    # Back in loss units, precise to float32 of the gap (about 1e-10
    # here), not of the loss (about 1e-5 at |f*| = 100).
    last = float(loss.isel({BUDGET: -1}).max())
    expect = F_STAR_VALUE + float(want[OUTPUT_MIN].isel({BUDGET: -1}).max())
    assert abs(last - expect) < 1e-9
    # output_std and the counters are not shifted.
    np.testing.assert_allclose(
        ds[OUTPUT_STD].isel({SCHEDULE: 0}).values, 0.01, rtol=1e-6
    )
    assert float(ds[FEVALS].max()) == 20
    mean = ds[OUTPUT_MEAN].isel({SCHEDULE: 0}).isel({BUDGET: -1})
    assert float(mean.min()) > F_STAR_VALUE
    # The loader too (DataLoader over the same helper), without f_star.
    loader = load_databank(NAME, [s.name for s in steps], root)
    assert F_STAR not in loader.data
    assert float(loader.data[OUTPUT_MIN].min()) > F_STAR_VALUE


def test_raw_files_hold_the_gap_and_describe_themselves(tmp_path, steps):
    task = FakeTask("f1", "sphere", 10, global_min=F_STAR_VALUE)
    runs = _offset_runs(tmp_path / "runs", task, steps, 13, F_STAR_VALUE)
    root = tmp_path / "db"
    _dump(root, runs)

    path = layout.loss_raw_path(
        layout.raw_root(root, NAME), "f1", steps[0].hash
    )
    on_disk = xr.load_dataset(path)
    assert on_disk.attrs[VALUE_KIND] == "gap"
    assert on_disk.attrs[F_STAR] == F_STAR_VALUE
    assert on_disk[OUTPUT_MIN].dtype == np.float32
    want = make_loss_ds(13, scale=10.0)
    np.testing.assert_allclose(
        on_disk[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET),
        want[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET),
        rtol=1e-6,
    )

    raw = load_loss_raw(NAME, root, "f1", steps[0].hash)
    assert raw[OUTPUT_MIN].dtype == np.float64
    assert VALUE_KIND not in raw.attrs
    got = raw[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET)
    gap = want[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET)
    # Precise to float32 *of the gap*: storing the loss in float32 would
    # be off by ~6e-6 at |f*| = 100, which is ~1e-2 of the smallest gaps.
    np.testing.assert_allclose(
        got.values - F_STAR_VALUE, gap.values, rtol=1e-6, atol=0
    )
    assert np.issubdtype(raw[FEVALS].dtype, np.integer)


def test_a_task_without_a_finite_f_star_stays_raw(tmp_path, steps):
    task = FakeTask("nf", "sphere", 10, global_min=float("nan"))
    runs = _offset_runs(tmp_path / "runs", task, steps, 13, 5.0)
    root = tmp_path / "db"
    _dump(root, runs)

    path = layout.loss_raw_path(
        layout.raw_root(root, NAME), "nf", steps[0].hash
    )
    on_disk = xr.load_dataset(path)
    assert on_disk.attrs[VALUE_KIND] == "raw"
    assert np.isnan(on_disk.attrs[F_STAR])
    want = make_loss_ds(13, scale=10.0)[OUTPUT_MIN] + 5.0
    np.testing.assert_allclose(
        on_disk[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET),
        want.squeeze(drop=True).transpose(REALIZATION, BUDGET),
        rtol=1e-6,
    )
    raw = load_loss_raw(NAME, root, "nf", steps[0].hash)
    np.testing.assert_allclose(
        raw[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET),
        want.squeeze(drop=True).transpose(REALIZATION, BUDGET),
        rtol=1e-6,
    )
    ds = load_databank_dataset(NAME, [s.name for s in steps], root)
    assert np.isnan(float(ds[F_STAR].sel({TASK_ID: "nf"})))
    assert float(ds[OUTPUT_MIN].min()) > 5.0


def test_dump_time_scores_come_from_the_losses_not_the_gaps(steps):
    """Shifted after scoring: the breach record holds losses, and the
    scores equal those computed straight from the unshifted runs."""
    task = FakeTask("br", "sphere", 10, global_min=F_STAR_VALUE)
    ds_by_step = {}
    for s in steps:
        ds = make_loss_ds(10, scale=10.0)
        ds[OUTPUT_MIN] = ds[OUTPUT_MIN] + F_STAR_VALUE
        ds_by_step[s] = ds
    # One run goes below f*: a breach of 0.5, in loss units.
    ds_by_step[steps[1]][OUTPUT_MIN][0, -1, 0] = F_STAR_VALUE - 0.5

    task_ds, loss, perf, _ = _build_task_records(
        task, ds_by_step, easiest_quantile=0.75
    )
    breach = json.loads(task_ds.attrs[BREACHES_ATTR])
    assert breach["schedules"][steps[1].name]["best"] == F_STAR_VALUE - 0.5

    concat = xr.concat(
        [ds.expand_dims({SCHEDULE: [s.name]}) for s, ds in ds_by_step.items()],
        dim=SCHEDULE,
    ).assign_coords(global_min=F_STAR_VALUE)
    want = _ert_from_quantile(concat, easiest_quantile=0.75)
    np.testing.assert_array_equal(
        task_ds[TARGET_VALUE].squeeze(drop=True).values,
        want[TARGET_VALUE].values,
    )
    for s in steps:
        got = perf[s.hash][SUCCESS].squeeze(drop=True)
        exp = (
            want[SUCCESS]
            .sel({SCHEDULE: s.name})
            .squeeze(drop=True)
            .transpose(*got.dims)
        )
        np.testing.assert_array_equal(got.values, exp.values)
        # The stored payload is the gap.
        gap = loss[s.hash][OUTPUT_MIN].squeeze(drop=True)
        assert float(gap.max()) < 20.0
    assert task_ds[F_STAR].dtype == np.float64


def test_append_keeps_f_star_of_old_tasks(tmp_path, steps):
    t0 = FakeTask("f0", "sphere", 10, global_min=F_STAR_VALUE)
    t1 = FakeTask("f1", "rastrigin", 10, global_min=-3.0)
    root = tmp_path / "db"
    _dump(root, _offset_runs(tmp_path / "a", t0, steps, 13, F_STAR_VALUE))
    _dump(
        root,
        _offset_runs(tmp_path / "b", t1, steps, 13, -3.0),
        mode="append",
    )
    ds = load_databank_dataset(NAME, [s.name for s in steps], root)
    np.testing.assert_array_equal(
        ds[F_STAR].sel({TASK_ID: ["f0", "f1"]}).values, [F_STAR_VALUE, -3.0]
    )
    assert float(ds[OUTPUT_MIN].sel({TASK_ID: "f0"}).min()) > F_STAR_VALUE
    assert -3.0 < float(ds[OUTPUT_MIN].sel({TASK_ID: "f1"}).min()) < 0.0


def test_verify_expects_f_star(tmp_path, tasks, steps):
    runs = _write_raw_runs(tmp_path / "runs", [(t, 13) for t in tasks], steps)
    root = tmp_path / "db"
    _dump(root, runs)
    report = verify_databank(root, NAME)
    assert report.ok, report.summary()
    assert not [f for f in report.findings if f.variable == F_STAR]


#                                         d. per-task B, checked per cell
# =============================================================================


def test_cells_matching_their_budget_are_stored_with_it(
    tmp_path, tasks, steps
):
    # make_loss_ds: one evaluation per iteration, n = 13.
    runs = _write_raw_runs(tmp_path / "runs", [(t, 13) for t in tasks], steps)
    root = tmp_path / "db"
    budget = {t.hash: 13 for t in tasks}
    _dump(root, runs, evaluation_budget=budget)
    ds = load_databank_dataset(NAME, [s.name for s in steps], root)
    assert ds[EVALUATION_BUDGET].dims == (TASK_ID,)
    assert np.issubdtype(ds[EVALUATION_BUDGET].dtype, np.integer)
    assert ds[EVALUATION_BUDGET].sel({TASK_ID: tasks[1].hash}).values == 13
    assert verify_databank(root, NAME).ok
    # Not part of the DataLoader contract.
    loader = load_databank(NAME, [s.name for s in steps], root)
    assert EVALUATION_BUDGET not in loader.data


@pytest.mark.parametrize("budget", [12, 14])
def test_a_one_evaluation_per_iteration_cell_needs_n_equal_b(
    tmp_path, tasks, steps, budget
):
    runs = _write_raw_runs(tmp_path / "runs", [(tasks[0], 13)], steps)
    with pytest.raises(ValueError, match="evaluation budget") as err:
        _dump(tmp_path / "db", runs, evaluation_budget={tasks[0].hash: budget})
    assert tasks[0].hash in str(err.value)
    assert "13 iteration" in str(err.value)


@pytest.mark.parametrize(
    ("budget", "ok"),
    [(60, True), (65, True), (59, False), (66, False)],
)
def test_a_fixed_cost_cell_needs_n_lambda_within_b(tmp_path, budget, ok):
    """lambda = 6 read from fevals; n = 10 iterations spend 60."""
    task = FakeTask("fx", "sphere", 10)
    cells = {RS: _cell(budget, [1]), CMA: _cell(10, [6])}
    runs = _cell_runs(tmp_path / "runs", task, cells)
    if ok:
        _dump(tmp_path / "db", runs, evaluation_budget={"fx": budget})
        return
    with pytest.raises(ValueError, match=CMA.name) as err:
        _dump(tmp_path / "db", runs, evaluation_budget={"fx": budget})
    assert "6 evaluations per iteration" in str(err.value)


def test_a_variable_cost_cell_needs_n_equal_b(tmp_path):
    """L-BFGS: 1 to 3 evaluations per step; budgeted at one per step."""
    task = FakeTask("vc", "sphere", 10)
    cells = {RS: _cell(20, [1]), LBFGS: _cell(20, [1, 3, 2])}
    runs = _cell_runs(tmp_path / "runs", task, cells)
    _dump(tmp_path / "db", runs, evaluation_budget={"vc": 20})

    spent = int(_cell(20, [1, 3, 2])[FEVALS].max())  # 39 evaluations
    runs = _cell_runs(
        tmp_path / "runs2",
        task,
        {RS: _cell(spent, [1]), LBFGS: _cell(20, [1, 3, 2])},
    )
    with pytest.raises(ValueError, match="varies"):
        _dump(tmp_path / "db2", runs, evaluation_budget={"vc": spent})


def test_a_variable_cost_cell_with_constant_steps_is_refused_loudly(tmp_path):
    """Fixed vs variable cost is read from the data (ADR 0027).

    A variable-cost run (budgeted at n == B) whose every step happens to
    cost the same lambda > 1 looks fixed-cost. Run at n == B it spends
    n * lambda > B, so the fixed-cost rule refuses it; it cannot pass
    silently.
    """
    task = FakeTask("eq", "sphere", 10)
    cells = {RS: _cell(20, [1]), LBFGS: _cell(20, [3])}
    runs = _cell_runs(tmp_path / "runs", task, cells)
    with pytest.raises(ValueError, match=LBFGS.name) as err:
        _dump(tmp_path / "db", runs, evaluation_budget={"eq": 20})
    assert "3 evaluations per iteration" in str(err.value)


def test_every_task_needs_a_budget_when_any_has_one(tmp_path, tasks, steps):
    runs = _write_raw_runs(tmp_path / "runs", [(t, 13) for t in tasks], steps)
    with pytest.raises(ValueError, match=tasks[1].hash):
        _dump(tmp_path / "db", runs, evaluation_budget={tasks[0].hash: 13})
    assert not layout.zarr_store_path(tmp_path / "db", NAME).exists()


def test_without_a_budget_nothing_is_checked_or_stored(tmp_path, tasks, steps):
    """Today's behaviour, until l2co_experiments passes B."""
    cells = {RS: _cell(13, [1]), CMA: _cell(10, [6])}
    runs = _cell_runs(tmp_path / "runs", tasks[0], cells)
    _dump(tmp_path / "db", runs)
    ds = load_databank_dataset(NAME, [RS.name, CMA.name], tmp_path / "db")
    assert EVALUATION_BUDGET not in ds


def _two_stores(tmp_path, tasks, steps, budget):
    runs_a = _write_raw_runs(tmp_path / "a", [(tasks[0], 13)], steps)
    root = tmp_path / "db"
    _dump(root, runs_a, evaluation_budget=budget)
    return root


def test_append_refuses_a_differing_budget_before_touching_the_store(
    tmp_path, tasks, steps
):
    root = _two_stores(tmp_path, tasks, steps, {tasks[0].hash: 13})
    extra = OptimizationStep(optimizer="sgd", hyperparameters={"lr": 0.1})
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[0], 14)], [extra])
    store = layout.zarr_store_path(root, NAME)
    before = _snapshot(store)
    with pytest.raises(ValueError, match="13") as err:
        _dump(
            root, runs_b, mode="append", evaluation_budget={tasks[0].hash: 14}
        )
    assert "14" in str(err.value)
    assert _snapshot(store) == before
    assert not layout.append_marker_path(root, NAME).exists()


def test_append_with_the_same_budget_extends_the_variable(
    tmp_path, tasks, steps
):
    root = _two_stores(tmp_path, tasks, steps, {tasks[0].hash: 13})
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 17)], steps)
    _dump(root, runs_b, mode="append", evaluation_budget={tasks[1].hash: 17})
    ds = load_databank_dataset(NAME, [s.name for s in steps], root)
    np.testing.assert_array_equal(
        ds[EVALUATION_BUDGET].sel({TASK_ID: [t.hash for t in tasks]}).values,
        [13, 17],
    )


def test_append_refuses_to_mix_budgeted_and_unbudgeted(tmp_path, tasks, steps):
    root = _two_stores(tmp_path, tasks, steps, {tasks[0].hash: 13})
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 13)], steps)
    with pytest.raises(ValueError, match="evaluation_budget"):
        _dump(root, runs_b, mode="append")

    root2 = tmp_path / "db2"
    _dump(root2, _write_raw_runs(tmp_path / "c", [(tasks[0], 13)], steps))
    with pytest.raises(ValueError, match="evaluation_budget"):
        _dump(
            root2,
            runs_b,
            mode="append",
            evaluation_budget={tasks[1].hash: 13},
        )


def test_the_budget_travels_in_the_serialized_plan(tmp_path, tasks, steps):
    """The parallel path: template -> JSON plan -> per-task -> finalize."""
    runs = _write_raw_runs(tmp_path / "runs", [(t, 13) for t in tasks], steps)
    root = tmp_path / "db"
    plan = _dump_template_runs(
        runs,
        root,
        NAME,
        mode="error",
        easiest_quantile=0.75,
        taskset=None,
        budget_steps=BS,
        evaluation_budget={t.hash: 13 for t in tasks},
    )
    plan = DumpPlan(**json.loads(json.dumps(dataclasses.asdict(plan))))
    smalls = [s for run in runs for s in _dump_task_runs([run], plan)]
    dump_databank_finalize(smalls, plan)
    ds = load_databank_dataset(NAME, [s.name for s in steps], root)
    assert ds[EVALUATION_BUDGET].values.tolist() == [13, 13]


def test_the_public_entry_points_take_the_budget():
    import inspect

    for fn in (
        databank.dump_databank,
        databank.dump_runs,
        databank.dump_databank_template,
    ):
        params = inspect.signature(fn).parameters
        assert params["evaluation_budget"].default is None
        assert params["run_settings"].default is None


#                                                        e. the cut at B
# =============================================================================


def _one_run(fevals, losses) -> xr.Dataset:
    fevals = np.asarray(fevals)
    n = fevals.size

    def _da(a):
        return xr.DataArray(
            np.asarray(a)[None, :, None],
            dims=[REALIZATION, BUDGET, "output_dim"],
            coords={REALIZATION: [0], "output_dim": ["y"]},
        )

    return xr.Dataset(
        {
            OUTPUT_MIN: _da(np.asarray(losses, dtype=float)),
            FEVALS: _da(fevals),
            ITERATIONS: _da(np.arange(1, n + 1)),
        }
    )


# fevals pass B = 10 at the fourth step; the hardest target (1e-3) is
# reached only there.
FEVALS_PAST_B = [3, 6, 9, 12, 15]
LOSSES = [1.0, 0.5, 0.1, 1e-4, 1e-4]
TARGETS = np.array([1e-3, 0.2, 2.0])


def test_a_target_first_reached_after_b_counts_as_missed():
    data = _one_run(FEVALS_PAST_B, LOSSES)
    free = calculate_ert(data, targets=TARGETS, global_min=0.0)
    cut = calculate_ert(
        data, targets=TARGETS, global_min=0.0, evaluation_budget=10
    )
    assert free[SUCCESS].squeeze().values.tolist() == [True, True, True]
    assert cut[SUCCESS].squeeze().values.tolist() == [False, True, True]
    # Solved before B: unchanged.
    xr.testing.assert_equal(
        cut[COST].isel(target=slice(1, None)),
        free[COST].isel(target=slice(1, None)),
    )


def test_a_missed_run_is_charged_at_most_b():
    # Never reaches 1e-3; last improvement at fevals 12 (> B = 10), so
    # without the cut it is charged 15, with it at most 10.
    data = _one_run(FEVALS_PAST_B, [1.0, 0.5, 0.1, 0.05, 0.05])
    free = calculate_ert(data, targets=TARGETS, global_min=0.0)
    cut = calculate_ert(
        data, targets=TARGETS, global_min=0.0, evaluation_budget=10
    )
    free_cost = free[COST].sel(metric=FEVALS).isel(target=0).item()
    cut_cost = cut[COST].sel(metric=FEVALS).isel(target=0).item()
    assert free_cost > 10
    assert cut_cost <= 10
    # Iterations follow the same truncated history.
    assert cut[COST].sel(metric=ITERATIONS).isel(target=0).item() == 3


def test_a_budget_above_every_point_changes_nothing():
    data = _one_run(FEVALS_PAST_B, LOSSES)
    free = calculate_ert(data, targets=TARGETS, global_min=0.0)
    cut = calculate_ert(
        data, targets=TARGETS, global_min=0.0, evaluation_budget=15
    )
    xr.testing.assert_identical(free, cut)


def test_the_dump_cuts_at_b_but_still_records_a_late_breach(tmp_path):
    """L-BFGS passes B; its late breach is still recorded (f* is wrong
    whatever the budget), but its late success is not counted."""
    task = FakeTask("lb", "sphere", 10, global_min=0.0)
    late = _cell(10, [1, 2])  # n = 10 = B, 15 evaluations
    late[OUTPUT_MIN][:, -1, :] = -1.0  # reached at fevals 15 > B
    early = _cell(10, [1])
    task_ds, _, perf, _ = _build_task_records(
        task,
        {RS: early, LBFGS: late},
        easiest_quantile=0.75,
        evaluation_budget=10,
    )
    breach = json.loads(task_ds.attrs[BREACHES_ATTR])
    assert LBFGS.name in breach["schedules"]
    hardest = perf[LBFGS.hash][SUCCESS].isel(target=0)
    assert not bool(hardest.any())
    assert int(task_ds[EVALUATION_BUDGET].item()) == 10
    _, _, perf_free, _ = _build_task_records(
        task, {RS: early, LBFGS: late}, easiest_quantile=0.75
    )
    assert bool(perf_free[LBFGS.hash][SUCCESS].isel(target=0).all())


#                              the simulated orders draw without replacement
# =============================================================================


@pytest.mark.parametrize("n", [1, 3, 25])
def test_simulated_orders_are_permutations_of_the_runs(n):
    for seed in range(5):
        order = draw_samples(seed=seed, n_realizations=n)
        assert sorted(order.tolist()) == list(range(n))
        np.testing.assert_array_equal(
            order, np.random.default_rng(seed).permutation(n)
        )


def _old_draw_samples(seed: int, n_realizations: int, n_picks: int = 50):
    """The removed implementation: a permutation, then a tail drawn with
    replacement up to ``n_picks``."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.permutation(n_realizations),
            rng.integers(
                0, n_realizations, size=max(n_picks - n_realizations, 0)
            ),
        ]
    )


@pytest.mark.parametrize("pool", [None, [0, 2]])
def test_the_dead_tail_changed_no_ertd_value(monkeypatch, pool):
    """The tail after the permutation was never reached: the ERTD with
    the old with-replacement draws is the ERTD without them, exactly."""
    from l2co._src import performance

    from .test_ertd_separability import _TASK_IDS, _loader

    new = _loader(_TASK_IDS).ertd_profile(realizations=pool)
    monkeypatch.setattr(
        performance,
        "draw_samples",
        lambda seed, n_realizations: _old_draw_samples(seed, n_realizations),
    )
    old = _loader(_TASK_IDS).ertd_profile(realizations=pool)
    xr.testing.assert_identical(new, old)
