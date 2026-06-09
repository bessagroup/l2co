"""Unit tests for the consolidated-zarr databank
(:mod:`l2co._src.databank`).

Covers the f3dasm-free record core (``_build_task_records``), budget
resampling (``loss_preprocess``), tasksets, the streaming creation /
append core (driven with hand-built ``runs`` so no f3dasm is needed),
the read path and loaders, the raw-history round-trip, and the
NaN-for-missing-pair semantics. The full ``dump_databank`` f3dasm
preamble is exercised end-to-end by the ``create_optimization_dataset``
pipeline, not here.

The databank takes an explicit ``root`` directory (no machine paths in
the library), so every test points it at the test's ``tmp_path``. The
store/read path is checked against the in-memory record pipeline
(``_build_task_records`` + ``_interp_loss``) rather than a second
backend: the two share that pipeline, so a cell read back from zarr must
equal what the pipeline produced for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from l2co import DataLoader, databank
from l2co._src.databank import layout
from l2co._src.databank.reader import (
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
)
from l2co._src.databank.records import _build_task_records, loss_preprocess
from l2co._src.databank.tasksets import load_taskset, save_taskset
from l2co._src.databank.writer import _dump_append, _dump_create, _interp_loss
from l2co.naming import (
    BUDGET,
    COST,
    FEATURE,
    FEVALS,
    ITERATIONS,
    METRIC,
    OPTIMIZATIONSTEP,
    OUTPUT_MIN,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
    TARGET,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)
from l2co.optimization import OptimizationStep

pytestmark = pytest.mark.filterwarnings("ignore")

N_REAL = 3
BS = 16  # small budget grid keeps the tests fast


#                                                                      Fixtures
# =============================================================================


@dataclass
class FakeTask:
    """Minimal stand-in for ``l2co_tasks.Task`` (hash/tag/global_min)."""

    hash: str
    fn_name: str
    dimensionality: int
    global_min: float = 0.0

    @property
    def tag(self) -> dict:
        return {"fn_name": self.fn_name, "dimensionality": self.dimensionality}


def make_loss_ds(n_budget: int, scale: float) -> xr.Dataset:
    """A raw per-run loss history: dims (realization, budget, output_dim).

    ``output_min`` decreases monotonically toward 0 so targets are
    reachable; ``fevals`` / ``iterations`` / ``update_step`` are integer
    cumulative counters.
    """
    budget = np.arange(n_budget)
    base = scale * np.exp(-budget / 2.0)[None, :]
    offset = (1 + np.arange(N_REAL))[:, None] * 1e-3
    output = (base + offset)[:, :, None]  # (real, budget, 1)

    def _da(arr):
        return xr.DataArray(
            arr,
            dims=[REALIZATION, BUDGET, "output_dim"],
            coords={REALIZATION: list(range(N_REAL)), "output_dim": ["y"]},
        )

    counters = np.broadcast_to(
        np.arange(1, n_budget + 1)[None, :, None], (N_REAL, n_budget, 1)
    )
    return xr.Dataset(
        {
            OUTPUT_MIN: _da(output),
            "output_mean": _da(output * 1.1),
            "output_std": _da(np.full_like(output, 0.01)),
            FEVALS: _da(counters.astype(int)),
            ITERATIONS: _da(counters.astype(int)),
            UPDATE_STEP: _da(counters.astype(int)),
        }
    )


@pytest.fixture
def steps() -> list[OptimizationStep]:
    return [
        OptimizationStep(optimizer="randomsearch"),
        OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01}),
    ]


@pytest.fixture
def tasks() -> list[FakeTask]:
    return [
        FakeTask(hash="a1b2c3", fn_name="sphere", dimensionality=10),
        FakeTask(hash="9f8e7d", fn_name="rastrigin", dimensionality=10),
    ]


def _write_raw_runs(rundir: Path, task_budgets: list, steps, omit=()) -> list:
    """Write raw loss netCDFs and return ``runs`` = [(task, {step: path})].

    ``task_budgets`` is a list of ``(task, raw_budget_length)`` pairs.
    ``omit`` is a set of ``(task_hash, step_name)`` pairs to leave out
    (to exercise ragged / missing-pair handling).
    """
    rundir.mkdir(parents=True, exist_ok=True)
    runs = []
    for task, nb in task_budgets:
        step_paths = {}
        for s in steps:
            if (task.hash, s.name) in omit:
                continue
            p = rundir / f"{task.hash}_{s.hash}.nc"
            make_loss_ds(nb, scale=10.0).to_netcdf(p)
            step_paths[s] = p
        runs.append((task, step_paths))
    return runs


def _create_zarr(root: Path, runs, budget_steps=BS):
    """Drive the streaming creation core directly (no f3dasm)."""
    store = layout.zarr_store_path(root, "bbob")
    rawroot = layout.raw_root(root, "bbob")
    rawroot.mkdir(parents=True, exist_ok=True)
    _dump_create(
        store, rawroot, runs, easiest_quantile=0.75, budget_steps=budget_steps
    )
    return store


def _records(task, nb, steps):
    """Expected per-record datasets for one task (the shared pipeline)."""
    ds_by_step = {s: make_loss_ds(nb, scale=10.0) for s in steps}
    return _build_task_records(task, ds_by_step, easiest_quantile=0.75)


def _assert_da_close(expected: xr.DataArray, got: xr.DataArray):
    """Compare two arrays NaN-aware, tolerant of size-1 dims / dim order."""
    e = expected.astype(float).squeeze(drop=True)
    g = got.astype(float).squeeze(drop=True).transpose(*e.dims)
    # The store keeps numerical variables as float32, so the round-trip
    # through it loses ~7 significant digits; the default rtol (~1e-7) sits
    # right at float32 epsilon and would flake.
    np.testing.assert_allclose(
        e.values, g.values, rtol=1e-6, atol=1e-7, equal_nan=True
    )


#                                                          _build_task_records
# =============================================================================


def test_build_task_records_schema(tasks, steps):
    task = tasks[0]
    ds_by_step = {s: make_loss_ds(8, scale=10.0) for s in steps}
    task_ds, loss_by_hash, perf_by_hash, opt_by_hash = _build_task_records(
        task, ds_by_step, easiest_quantile=0.75
    )

    # task file: labels (taskID, feature) + target_value (taskID, target)
    assert set(task_ds[TASK_LABELS].dims) == {TASK_ID, FEATURE}
    assert set(task_ds[TARGET_VALUE].dims) == {TASK_ID, TARGET}
    assert "global_min" in list(task_ds[FEATURE].values)

    rs_hash = steps[0].hash
    loss = loss_by_hash[rs_hash]
    assert loss[OUTPUT_MIN].dims[:2] == (TASK_ID, SCHEDULE)
    assert {REALIZATION, BUDGET}.issubset(set(loss[OUTPUT_MIN].dims))
    assert UPDATE_STEP in loss  # kept (user override)

    perf = perf_by_hash[rs_hash]
    assert SUCCESS in perf and COST in perf
    assert TARGET in perf[SUCCESS].dims
    assert METRIC in perf[COST].dims
    assert TARGET_VALUE not in perf  # lives in the task file only
    assert bool(perf[SUCCESS].dtype == bool)

    opt = opt_by_hash[rs_hash]
    assert set(opt[OPTIMIZATIONSTEP].dims) == {SCHEDULE}
    assert str(opt[SCHEDULE].values[0]) == "randomsearch"


#                                                               loss_preprocess
# =============================================================================


def test_loss_preprocess_common_grid_and_int():
    short = make_loss_ds(7, scale=1.0)
    long = make_loss_ds(13, scale=1.0)

    a = loss_preprocess(short, budget_steps=5)
    b = loss_preprocess(long, budget_steps=5)

    assert a.sizes[BUDGET] == b.sizes[BUDGET] == 5
    np.testing.assert_array_equal(a[BUDGET].values, b[BUDGET].values)
    for var in (FEVALS, ITERATIONS, UPDATE_STEP):
        assert np.issubdtype(a[var].dtype, np.integer)
        assert not np.isnan(a[var].values).any()
    # the two resampled files merge cleanly along schedule
    merged = xr.concat(
        [a.expand_dims({SCHEDULE: ["x"]}), b.expand_dims({SCHEDULE: ["y"]})],
        dim=SCHEDULE,
    )
    assert merged.sizes[SCHEDULE] == 2


#                                                             Streaming create
# =============================================================================


def test_create_stores_records(tmp_path, tasks, steps):
    # The same data goes through the in-memory record pipeline and the
    # streaming zarr create; every stored cell must match the pipeline.
    task_budgets = [(tasks[0], 7), (tasks[1], 13)]
    runs = _write_raw_runs(tmp_path / "runs", task_budgets, steps)
    _create_zarr(tmp_path, runs)
    zd = load_databank_dataset("bbob", steps, tmp_path).load()

    assert zd.sizes[BUDGET] == BS
    # numerical vars are stored / returned as float32 (missing cell = NaN)
    assert zd[FEVALS].dtype == np.float32
    assert zd[SUCCESS].dtype == np.float32

    for task, nb in task_budgets:
        task_ds, loss, perf, _ = _records(task, nb, steps)
        _assert_da_close(
            task_ds[TARGET_VALUE],
            zd[TARGET_VALUE].sel({TASK_ID: task.hash}),
        )
        for s in steps:
            sel = {TASK_ID: task.hash, SCHEDULE: s.name}
            loss_interp = _interp_loss(loss[s.hash], BS)
            # loss arrays (interpolated) and performance arrays
            for v in (OUTPUT_MIN, FEVALS):
                _assert_da_close(loss_interp[v], zd[v].sel(sel))
            for v in (SUCCESS, COST):
                _assert_da_close(perf[s.hash][v], zd[v].sel(sel))


def test_missing_pair_is_nan(tmp_path, tasks, steps):
    # task1 has no "adam" run -> that cell must read back NaN.
    runs = _write_raw_runs(
        tmp_path / "runs",
        [(tasks[0], 7), (tasks[1], 7)],
        steps,
        omit={(tasks[1].hash, "adam_lr=0.01")},
    )
    _create_zarr(tmp_path, runs)
    zd = load_databank_dataset("bbob", steps, tmp_path).load()

    cell = zd[OUTPUT_MIN].sel(
        {TASK_ID: tasks[1].hash, SCHEDULE: "adam_lr=0.01"}
    )
    assert bool(np.isnan(cell).all())
    present = zd[OUTPUT_MIN].sel(
        {TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"}
    )
    assert not bool(np.isnan(present).all())


def test_populated_pairs(tmp_path, tasks, steps):
    # Missing store -> empty set (the "run everything" signal).
    assert databank.populated_pairs(tmp_path, "bbob") == set()

    # One omitted cell -> every other (task, optimizer) pair is reported.
    runs = _write_raw_runs(
        tmp_path / "runs",
        [(tasks[0], 7), (tasks[1], 7)],
        steps,
        omit={(tasks[1].hash, "adam_lr=0.01")},
    )
    _create_zarr(tmp_path, runs)

    assert databank.populated_pairs(tmp_path, "bbob") == {
        (tasks[0].hash, "randomsearch"),
        (tasks[0].hash, "adam_lr=0.01"),
        (tasks[1].hash, "randomsearch"),
    }


#                                                                  Reader path
# =============================================================================


def test_returns_compatible_dataloader(tmp_path, tasks, steps):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)
    d = load_databank("bbob", steps, tmp_path)

    assert isinstance(d, DataLoader)
    assert "problem" in d.data.dims
    assert OPTIMIZATIONSTEP not in d.data
    assert d.data.sizes[SCHEDULE] == len(steps)
    assert FEATURE in d.task_labels.dims
    assert "fn_name" in list(d.task_labels[FEATURE].values)
    # The single loader carries both the loss and performance families.
    assert SUCCESS in d.data
    assert OUTPUT_MIN in d.data
    # The selection stays lazy: the heavy loss variable is still
    # dask-backed (not materialised on construction).
    assert d.data[OUTPUT_MIN].chunks is not None


def test_performance_loader_stays_lazy_and_matches_full(
    tmp_path, tasks, steps
):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)

    pl = load_databank_performance("bbob", steps, tmp_path)
    assert isinstance(pl, DataLoader)
    # performance-only loader carries success/cost, not the loss arrays
    assert SUCCESS in pl.data and COST in pl.data
    assert OUTPUT_MIN not in pl.data
    assert BUDGET not in pl.data.dims
    # stays lazy (no eager .load())
    assert pl.data[SUCCESS].chunks is not None
    # values agree with the full loader's performance variables
    full = load_databank("bbob", steps, tmp_path)
    xr.testing.assert_allclose(
        pl.data[SUCCESS].astype(float), full.data[SUCCESS].astype(float)
    )


def test_subset_by_task_and_optimizer(tmp_path, tasks, steps):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)

    zd = load_databank_dataset(
        "bbob", [steps[0]], tmp_path, task_ids=[tasks[0].hash]
    )
    assert zd.sizes[TASK_ID] == 1
    assert zd.sizes[SCHEDULE] == 1
    assert str(zd[TASK_ID].values[0]) == tasks[0].hash
    assert str(zd[SCHEDULE].values[0]) == "randomsearch"


def test_facade_load_databank_dataset(tmp_path, tasks, steps):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)
    zd = databank.load_databank_dataset("bbob", steps, tmp_path)
    assert zd.sizes[SCHEDULE] == len(steps)


def test_select_optimizer_by_int_hash(tmp_path, tasks, steps):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)
    # pass the integer hash instead of the OptimizationStep
    zd = load_databank_dataset("bbob", [steps[1].hash], tmp_path)
    assert str(zd[SCHEDULE].values[0]) == "adam_lr=0.01"


def test_select_optimizer_by_name(tmp_path, tasks, steps):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)
    # pass the schedule-name string (legacy ergonomics)
    zd = load_databank_dataset("bbob", ["randomsearch"], tmp_path)
    assert str(zd[SCHEDULE].values[0]) == "randomsearch"
    assert zd.sizes[SCHEDULE] == 1


def test_recipes_of_differing_lengths_not_truncated(tmp_path, tasks):
    # Optimizers whose JSON recipes differ in length must round-trip
    # intact: fixed-width string storage would cap them at the first
    # recipe's width and corrupt the longer ones (unterminated JSON).
    import json

    short = OptimizationStep(optimizer="sepcmaes")
    long = OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 0.01}
    )
    assert len(json.dumps(long.to_dict())) > len(json.dumps(short.to_dict()))
    runs = _write_raw_runs(tmp_path / "runs", [(tasks[0], 7)], [short, long])
    _create_zarr(tmp_path, runs)

    # both selectable (round-tripped recipes parse cleanly)
    zd = load_databank_dataset("bbob", [short, long], tmp_path)
    assert zd.sizes[SCHEDULE] == 2
    # and selectable by the longer one's int hash (needs valid JSON)
    zl = load_databank_dataset("bbob", [long.hash], tmp_path)
    assert str(zl[SCHEDULE].values[0]) == long.name


def test_inconsistent_feature_axis_raises(tmp_path, steps):
    # two tasks with different tag keys -> different feature axis
    t0 = FakeTask("aaa", "sphere", 10)

    @dataclass
    class OddTask:
        hash: str = "bbb"
        global_min: float = 0.0

        @property
        def tag(self):
            return {"different_key": "x"}

    runs = _write_raw_runs(tmp_path / "runs", [(t0, 7), (OddTask(), 7)], steps)
    with pytest.raises(ValueError, match="feature"):
        _create_zarr(tmp_path, runs)


#                                                                  Raw history
# =============================================================================


def test_loss_raw_roundtrips(tmp_path, tasks, steps):
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)

    raw = load_loss_raw("bbob", tmp_path, tasks[1].hash, steps[0].hash)
    assert raw.sizes[BUDGET] == 13  # original variable length preserved
    # Float vars are downcast to float32 on the sidecar to save disk, but
    # the integer counters keep their integer dtype (selective downcast).
    assert raw[OUTPUT_MIN].dtype == np.float32
    for var in (FEVALS, ITERATIONS, UPDATE_STEP):
        assert np.issubdtype(raw[var].dtype, np.integer)


def test_regenerate_from_raw(tmp_path, tasks, steps):
    # Re-interp the raw history at a different budget grid and check it
    # matches resampling the original raw run at that grid.
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)

    reint = load_loss_raw(
        "bbob", tmp_path, tasks[1].hash, steps[0].hash, budget_steps=9
    )
    expect = loss_preprocess(make_loss_ds(13, scale=10.0), budget_steps=9)
    r = reint[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET)
    e = expect[OUTPUT_MIN].squeeze(drop=True).transpose(REALIZATION, BUDGET)
    np.testing.assert_allclose(r.values, e.values, equal_nan=True)


#                                                                       Append
# =============================================================================


def test_append_optimizer_and_task(tmp_path, tasks, steps):
    rs, adam = steps
    # create: task0 with rs only
    runs0 = _write_raw_runs(tmp_path / "r0", [(tasks[0], 7)], [rs])
    store = _create_zarr(tmp_path, runs0)
    rawroot = layout.raw_root(tmp_path, "bbob")

    base = load_databank_dataset("bbob", [rs], tmp_path).load()
    target0_before = base[TARGET_VALUE].sel({TASK_ID: tasks[0].hash}).values

    # append: task0 gains adam, and a brand-new task1 gains rs + adam
    runs1 = _write_raw_runs(
        tmp_path / "r1", [(tasks[0], 7)], [adam]
    ) + _write_raw_runs(tmp_path / "r2", [(tasks[1], 7)], [rs, adam])
    _dump_append(store, rawroot, runs1, easiest_quantile=0.75)

    zd = load_databank_dataset("bbob", steps, tmp_path).load()
    assert zd.sizes[TASK_ID] == 2
    assert zd.sizes[SCHEDULE] == 2

    # old (task0, rs) cell preserved
    old_cell = zd[OUTPUT_MIN].sel(
        {TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"}
    )
    assert not bool(np.isnan(old_cell).all())
    # newly appended cells populated
    for tid, sch in [
        (tasks[0].hash, "adam_lr=0.01"),
        (tasks[1].hash, "randomsearch"),
        (tasks[1].hash, "adam_lr=0.01"),
    ]:
        c = zd[OUTPUT_MIN].sel({TASK_ID: tid, SCHEDULE: sch})
        assert not bool(np.isnan(c).all())

    # task0 kept its original ERT targets (append reuses stored targets)
    target0_after = zd[TARGET_VALUE].sel({TASK_ID: tasks[0].hash}).values
    np.testing.assert_array_equal(target0_before, target0_after)


#                                                                      Tasksets
# =============================================================================


def test_load_taskset_strips_comments(tmp_path):
    d = tmp_path / "tasksets"
    d.mkdir()
    (d / "bbob.txt").write_text(
        "# header\na1b2c3   # sphere 10D\n\n9f8e7d\n# trailing comment only\n"
    )
    assert load_taskset(tmp_path, "bbob") == ["a1b2c3", "9f8e7d"]


def test_load_taskset_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_taskset(tmp_path, "nope")


def test_load_taskset_empty_raises(tmp_path):
    d = tmp_path / "tasksets"
    d.mkdir()
    (d / "empty.txt").write_text("# only comments\n\n")
    with pytest.raises(ValueError):
        load_taskset(tmp_path, "empty")


def test_save_taskset_roundtrip(tmp_path):
    save_taskset(
        tmp_path,
        "sel",
        ["a1b2c3", "9f8e7d"],
        descriptions={"a1b2c3": "sphere 10D"},
    )
    assert load_taskset(tmp_path, "sel") == ["a1b2c3", "9f8e7d"]
