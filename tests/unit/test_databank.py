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

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from l2co import DataLoader, databank
from l2co._src.databank import layout
from l2co._src.databank import writer as databank_writer
from l2co._src.databank.reader import (
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
)
from l2co._src.databank.records import _build_task_records, loss_preprocess
from l2co._src.databank.tasksets import load_taskset, save_taskset
from l2co._src.databank.verify import verify_databank
from l2co._src.databank.writer import (
    _dump_append,
    _dump_create,
    _dump_task_runs,
    _dump_template_runs,
    _interp_loss,
    _minimal_sample_steps,
    dump_databank_finalize,
)
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


def test_populated_iterations(tmp_path, tasks, steps):
    # Missing store -> empty dict (the "run everything" signal).
    assert databank.populated_iterations(tmp_path, "bbob") == {}

    # Each populated cell reports the stored run's iteration count, which
    # is the ITERATIONS counter's max == the raw budget length here
    # (make_loss_ds sets ITERATIONS = arange(1, nb + 1)). The omitted cell
    # is absent, just like populated_pairs.
    runs = _write_raw_runs(
        tmp_path / "runs",
        [(tasks[0], 7), (tasks[1], 13)],
        steps,
        omit={(tasks[1].hash, "adam_lr=0.01")},
    )
    _create_zarr(tmp_path, runs)

    assert databank.populated_iterations(tmp_path, "bbob") == {
        (tasks[0].hash, "randomsearch"): 7,
        (tasks[0].hash, "adam_lr=0.01"): 7,
        (tasks[1].hash, "randomsearch"): 13,
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


def test_load_databank_to_params(tmp_path, tasks, steps):
    """The databank loader supports the training pipelines' ``to_params``.

    End-to-end contract the supervised training experiments rely on: a
    databank selection produces ``OptimizationDynamics`` whose label
    axis is exactly the selected schedules, NaN-free after ``filter``.
    Also a regression test for the ``FeatureProcessor`` call sites in
    ``task_features`` / ``optimizer_features``, which kept passing the
    pre-refactor ``tags`` kwarg (and the ``"categorical"`` string spec)
    after the vocabs-only refactor — breaking every ``to_params`` call.
    """
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 13)], steps
    )
    _create_zarr(tmp_path, runs)
    d = load_databank("bbob", steps, tmp_path)

    dynamics = d.to_params(
        t=0.25,
        t_end=100,
        schema={
            "dimensionality": "continuous",
            "fn_name": ["sphere", "rastrigin"],
        },
    )
    labels = [str(v) for v in dynamics.y["label"].values]
    assert sorted(labels) == sorted(s.name for s in steps)
    # The optimizer vocabulary is the data's schedule names.
    assert set(dynamics.feature_processor_optimizer.vocabs["optimizer"]) == {
        s.name for s in steps
    }

    filtered = dynamics.filter()
    assert filtered.y.sizes["label"] == len(steps)
    assert filtered.y.sizes["sample"] > 0
    assert not np.isnan(np.asarray(filtered.y)).any()
    assert not np.isnan(np.asarray(filtered.x_history)).any()


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


def test_append_overwrites_higher_budget(tmp_path, tasks, steps):
    # create: task0 with rs + adam at a low budget (nb=7) ...
    runs0 = _write_raw_runs(tmp_path / "r0", [(tasks[0], 7)], steps)
    store = _create_zarr(tmp_path, runs0)
    rawroot = layout.raw_root(tmp_path, "bbob")
    # ... plus an unrelated task1 (also nb=7) that is never re-run.
    runs0b = _write_raw_runs(tmp_path / "r0b", [(tasks[1], 7)], steps)
    _dump_append(store, rawroot, runs0b, easiest_quantile=0.75)

    # append: re-run task0's cells at a HIGHER budget (nb=13).
    runs1 = _write_raw_runs(tmp_path / "r1", [(tasks[0], 13)], steps)
    _dump_append(store, rawroot, runs1, easiest_quantile=0.75)

    # the re-run cells now report the higher iteration count; task1 is
    # untouched.
    iters = databank.populated_iterations(tmp_path, "bbob")
    for s in steps:
        assert iters[(tasks[0].hash, s.name)] == 13
        assert iters[(tasks[1].hash, s.name)] == 7

    # the stored loss matches the nb=13 records, not the stale nb=7 ones --
    # i.e. the copy-old-cells step did not clobber the re-run. (OUTPUT_MIN
    # is target-independent, so reusing task0's old ERT targets on the
    # re-run does not affect this comparison.)
    zd = load_databank_dataset("bbob", steps, tmp_path).load()
    _, loss13, _, _ = _records(tasks[0], 13, steps)
    for s in steps:
        sel = {TASK_ID: tasks[0].hash, SCHEDULE: s.name}
        _assert_da_close(
            _interp_loss(loss13[s.hash], BS)[OUTPUT_MIN],
            zd[OUTPUT_MIN].sel(sel),
        )


def test_append_carries_old_cells_as_file_copies(
    tmp_path, tasks, steps, monkeypatch
):
    """Untouched cells are carried over as chunk-file copies: the append
    only region-writes the appended runs' cells, and the carried cells
    read back unchanged."""
    runs0 = _write_raw_runs(tmp_path / "r0", [(tasks[0], 7)], steps)
    store = _create_zarr(tmp_path, runs0)
    rawroot = layout.raw_root(tmp_path, "bbob")
    before = load_databank_dataset("bbob", steps, tmp_path).load()

    calls: list[tuple[int, int]] = []
    orig = databank_writer._write_pair

    def counting(arrays, ds, ti, sj):
        calls.append((ti, sj))
        return orig(arrays, ds, ti, sj)

    monkeypatch.setattr(databank_writer, "_write_pair", counting)
    runs1 = _write_raw_runs(tmp_path / "r1", [(tasks[1], 7)], steps)
    _dump_append(store, rawroot, runs1, easiest_quantile=0.75)

    # Only the appended task's cells were written (one loss + one perf
    # write each); task0's cells never went through _write_pair.
    assert len(calls) == 2 * len(steps)
    assert {ti for ti, _ in calls} == {1}

    zd = load_databank_dataset("bbob", steps, tmp_path).load()
    for s in steps:
        sel = {TASK_ID: tasks[0].hash, SCHEDULE: s.name}
        _assert_da_close(before[OUTPUT_MIN].sel(sel), zd[OUTPUT_MIN].sel(sel))


def test_append_fallback_mismatched_realizations(tmp_path, tasks, steps):
    """An appended run with a different realization count rebuilds the
    loss arrays with a new cell layout, so old loss cells take the
    decode/re-encode fallback: carried values intact, extra realization
    rows NaN."""
    runs0 = _write_raw_runs(tmp_path / "r0", [(tasks[0], 7)], steps)
    store = _create_zarr(tmp_path, runs0)
    rawroot = layout.raw_root(tmp_path, "bbob")
    before = load_databank_dataset("bbob", steps, tmp_path).load()

    rundir = tmp_path / "r1"
    rundir.mkdir()
    step_paths = {}
    for s in steps:
        ds6 = xr.concat(
            [make_loss_ds(7, scale=10.0)] * 2, dim=REALIZATION
        ).assign_coords({REALIZATION: list(range(2 * N_REAL))})
        p = rundir / f"{tasks[1].hash}_{s.hash}.nc"
        ds6.to_netcdf(p)
        step_paths[s] = p
    _dump_append(
        store, rawroot, [(tasks[1], step_paths)], easiest_quantile=0.75
    )

    zd = load_databank_dataset("bbob", steps, tmp_path).load()
    assert zd.sizes[REALIZATION] == 2 * N_REAL
    for s in steps:
        old_cell = zd[OUTPUT_MIN].sel(
            {TASK_ID: tasks[0].hash, SCHEDULE: s.name}
        )
        # The old store's realizations carry over ...
        _assert_da_close(
            before[OUTPUT_MIN].sel({TASK_ID: tasks[0].hash, SCHEDULE: s.name}),
            old_cell.isel({REALIZATION: slice(0, N_REAL)}),
        )
        # ... and the rows beyond its realization count stay NaN.
        assert bool(
            np.isnan(old_cell.isel({REALIZATION: slice(N_REAL, None)})).all()
        )
        new_cell = zd[OUTPUT_MIN].sel(
            {TASK_ID: tasks[1].hash, SCHEDULE: s.name}
        )
        assert not bool(np.isnan(new_cell).any())


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


#                                                                  Verification
# =============================================================================


def _full_store(tmp_path, tasks, steps, omit=()):
    """A complete 2 tasks x 2 schedules store under ``tmp_path``."""
    runs = _write_raw_runs(
        tmp_path / "runs", [(tasks[0], 7), (tasks[1], 7)], steps, omit=omit
    )
    return _create_zarr(tmp_path, runs)


def _zarr_corrupt(store: Path, var: str, value, **sel):
    """Write ``value`` into ``var`` at per-dim positions.

    ``sel`` maps a dim name to a coordinate label (``taskID`` /
    ``schedule`` strings) or an integer/slice position; unspecified dims
    take the full slice. Value writes through zarr leave the
    consolidated metadata valid, so the corrupted store still opens.
    """
    with xr.open_zarr(store) as ds:
        dims = ds[var].dims
        labels = {
            d: [str(x) for x in ds[d].values]
            for d in (TASK_ID, SCHEDULE)
            if d in dims
        }
    idx = tuple(
        labels[d].index(sel[d])
        if isinstance(sel.get(d), str)
        else sel.get(d, slice(None))
        for d in dims
    )
    zarr.open_group(str(store), mode="r+")[var][idx] = value


def _by_check(report, check: str):
    return [f for f in report.findings if f.check == check]


def test_verify_healthy_store_ok(tmp_path, tasks, steps):
    _full_store(tmp_path, tasks, steps)
    report = verify_databank(tmp_path, "bbob")

    assert report.ok
    assert report.errors == ()
    assert report.warnings == ()
    assert report.n_pairs_expected == report.n_pairs_populated == 4
    assert report.missing_pairs == ()
    assert _by_check(report, "completeness.summary")
    text = str(report)
    assert "bbob" in text and "OK" in text


def test_verify_missing_store_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_databank(tmp_path, "bbob")


def test_verify_scattered_nan_is_warning(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        OUTPUT_MIN,
        np.nan,
        **{TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch", BUDGET: 0},
    )
    report = verify_databank(tmp_path, "bbob")

    assert report.ok  # scattered non-finite loss values are expected
    found = _by_check(report, "loss.scattered_nonfinite")
    assert len(found) == 1
    f = found[0]
    assert f.severity == "warning"
    assert f.task_id == tasks[0].hash
    assert f.schedule == "randomsearch"
    assert f.variable == OUTPUT_MIN
    assert f.count == N_REAL  # one budget slot across all realizations


def test_verify_inf_treated_as_nonfinite(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    # inf in a loss variable behaves like NaN: a warning, not an error
    _zarr_corrupt(
        store,
        OUTPUT_MIN,
        np.inf,
        **{TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch", BUDGET: 0},
    )
    # inf in a performance variable is an error like NaN
    _zarr_corrupt(
        store,
        COST,
        np.inf,
        **{
            TASK_ID: tasks[1].hash,
            SCHEDULE: "randomsearch",
            REALIZATION: 0,
            "output_dim": 0,
            TARGET: 0,
            METRIC: 0,
        },
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    assert _by_check(report, "loss.scattered_nonfinite")
    errs = _by_check(report, "perf.nonfinite")
    assert len(errs) == 1
    assert errs[0].variable == COST
    assert errs[0].count == 1


def test_verify_whole_var_nan_is_error(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        "output_mean",
        np.nan,
        **{TASK_ID: tasks[0].hash, SCHEDULE: "adam_lr=0.01"},
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    errs = _by_check(report, "loss.all_nonfinite")
    assert len(errs) == 1
    assert errs[0].variable == "output_mean"
    assert errs[0].task_id == tasks[0].hash
    assert errs[0].schedule == "adam_lr=0.01"
    # the cell is not flagged as a partial write (other loss vars intact)
    assert not _by_check(report, "pair.partial_loss_missing")


def test_verify_nan_in_cost_is_error(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        COST,
        np.nan,
        **{
            TASK_ID: tasks[0].hash,
            SCHEDULE: "randomsearch",
            REALIZATION: 0,
            "output_dim": 0,
            TARGET: 0,
            METRIC: 0,
        },
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    errs = _by_check(report, "perf.nonfinite")
    assert len(errs) == 1
    assert errs[0].variable == COST
    assert errs[0].count == 1


def test_verify_partial_write_loss_missing(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    sel = {TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"}
    for var in (
        OUTPUT_MIN,
        "output_mean",
        "output_std",
        FEVALS,
        ITERATIONS,
        UPDATE_STEP,
    ):
        _zarr_corrupt(store, var, np.nan, **sel)
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    errs = _by_check(report, "pair.partial_loss_missing")
    assert len(errs) == 1
    assert (errs[0].task_id, errs[0].schedule) == (
        tasks[0].hash,
        "randomsearch",
    )
    # one root cause, one finding: per-variable errors are suppressed
    assert not _by_check(report, "loss.all_nonfinite")
    assert report.n_pairs_populated == 4  # success is still non-null


def test_verify_partial_write_perf_missing(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    sel = {TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"}
    _zarr_corrupt(store, SUCCESS, np.nan, **sel)
    _zarr_corrupt(store, COST, np.nan, **sel)
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    errs = _by_check(report, "pair.partial_perf_missing")
    assert len(errs) == 1
    assert (errs[0].task_id, errs[0].schedule) == (
        tasks[0].hash,
        "randomsearch",
    )
    assert report.n_pairs_populated == 3
    # a half-written cell is not a benign missing pair
    assert (tasks[0].hash, "randomsearch") not in report.missing_pairs


def test_verify_unwritten_pair_is_info(tmp_path, tasks, steps):
    _full_store(tmp_path, tasks, steps, omit={(tasks[1].hash, "adam_lr=0.01")})
    report = verify_databank(tmp_path, "bbob")

    assert report.ok
    assert report.errors == ()
    assert report.n_pairs_populated == 3
    assert report.missing_pairs == ((tasks[1].hash, "adam_lr=0.01"),)
    summary = _by_check(report, "completeness.summary")[0]
    assert tasks[1].hash in summary.message


def test_verify_success_non_monotone_is_error(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    sel = {TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"}
    # solved at every target except the easiest (last) one: impossible
    _zarr_corrupt(store, SUCCESS, 1.0, **sel)
    _zarr_corrupt(store, SUCCESS, 0.0, **sel, **{TARGET: -1})
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    assert _by_check(report, "consistency.success_not_monotone")
    assert not _by_check(report, "consistency.success_not_binary")


def test_verify_success_non_binary_is_error(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        SUCCESS,
        0.5,
        **{
            TASK_ID: tasks[0].hash,
            SCHEDULE: "randomsearch",
            REALIZATION: 0,
            "output_dim": 0,
            TARGET: 0,
        },
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    errs = _by_check(report, "consistency.success_not_binary")
    assert len(errs) == 1
    assert errs[0].count == 1


def test_verify_negative_std_is_error(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        "output_std",
        -1.0,
        **{
            TASK_ID: tasks[0].hash,
            SCHEDULE: "randomsearch",
            REALIZATION: 0,
            "output_dim": 0,
            BUDGET: 0,
        },
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    errs = _by_check(report, "consistency.negative_std")
    assert len(errs) == 1
    assert errs[0].count == 1


def test_verify_counter_checks(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    cell = {TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"}
    point = {REALIZATION: 0, "output_dim": 0}
    # negative counter (budget slot 0, so no decreasing diff to it)
    _zarr_corrupt(store, FEVALS, -3.0, **cell, **point, **{BUDGET: 0})
    # non-integer counter, still below its successor (no decrease)
    _zarr_corrupt(store, ITERATIONS, 0.5, **cell, **point, **{BUDGET: 0})
    # spike that the next budget step falls back from (a decrease)
    _zarr_corrupt(store, UPDATE_STEP, 1000.0, **cell, **point, **{BUDGET: 8})
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    neg = _by_check(report, f"consistency.counter_negative.{FEVALS}")
    assert len(neg) == 1 and neg[0].severity == "error"
    nonint = _by_check(report, f"consistency.counter_noninteger.{ITERATIONS}")
    assert len(nonint) == 1 and nonint[0].severity == "warning"
    dec = _by_check(report, f"consistency.counter_decreasing.{UPDATE_STEP}")
    assert len(dec) == 1 and dec[0].severity == "warning"


def test_verify_target_value_corruption(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        TARGET_VALUE,
        np.nan,
        **{TASK_ID: tasks[0].hash, TARGET: 0},
    )
    # a huge hardest target breaks the ascending (hardest->easiest) order
    _zarr_corrupt(
        store,
        TARGET_VALUE,
        1e9,
        **{TASK_ID: tasks[1].hash, TARGET: 0},
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    nonfinite = _by_check(report, "task.target_value_nonfinite")
    assert len(nonfinite) == 1
    assert nonfinite[0].task_id == tasks[0].hash
    assert nonfinite[0].schedule is None
    unsorted = _by_check(report, "task.target_value_unsorted")
    assert len(unsorted) == 1
    assert unsorted[0].task_id == tasks[1].hash


def test_verify_subset_filters(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    _zarr_corrupt(
        store,
        "output_mean",
        np.nan,
        **{TASK_ID: tasks[0].hash, SCHEDULE: "randomsearch"},
    )
    assert not verify_databank(tmp_path, "bbob").ok

    # the corruption sits outside both of these selections
    by_task = verify_databank(tmp_path, "bbob", task_ids=[tasks[1].hash])
    assert by_task.ok and by_task.n_pairs_expected == 2
    by_sched = verify_databank(tmp_path, "bbob", schedules=["adam_lr=0.01"])
    assert by_sched.ok and by_sched.n_pairs_expected == 2

    with pytest.raises(ValueError):
        verify_databank(
            tmp_path, "bbob", task_ids=[tasks[1].hash], taskset="sel"
        )

    # unknown requested ids: warned and skipped, not fatal
    partial = verify_databank(
        tmp_path, "bbob", task_ids=[tasks[1].hash, "zzz"]
    )
    assert partial.ok
    assert _by_check(partial, "subset.unknown_task")
    with pytest.raises(ValueError):
        verify_databank(tmp_path, "bbob", task_ids=["zzz"])


def test_verify_taskset_unknown_hash_warning(tmp_path, tasks, steps):
    _full_store(tmp_path, tasks, steps)
    save_taskset(
        layout.databank_root(tmp_path, "bbob"),
        "sel",
        [tasks[0].hash, "deadbeef"],
    )
    report = verify_databank(tmp_path, "bbob")

    assert report.ok  # a stale taskset is a warning, not an error
    found = _by_check(report, "taskset.unknown_hash")
    assert len(found) == 1
    assert "deadbeef" in found[0].message


def test_verify_missing_variable_is_error(tmp_path, tasks, steps):
    store = _full_store(tmp_path, tasks, steps)
    shutil.rmtree(store / COST)
    zarr.consolidate_metadata(str(store))
    report = verify_databank(tmp_path, "bbob")  # must not crash

    assert not report.ok
    errs = _by_check(report, "structure.missing_var")
    assert len(errs) == 1
    assert errs[0].variable == COST
    # pair analysis still ran on the remaining variables
    assert report.n_pairs_populated == 4


def test_verify_budget_nondefault_is_info(tmp_path, tasks, steps):
    _full_store(tmp_path, tasks, steps)  # BS=16 != BUDGET_STEPS
    report = verify_databank(tmp_path, "bbob")

    found = _by_check(report, "structure.budget_nondefault")
    assert len(found) == 1
    assert found[0].severity == "info"
    assert report.ok


def test_verify_facade(tmp_path, tasks, steps):
    _full_store(tmp_path, tasks, steps)
    report = databank.verify_databank(tmp_path, "bbob")
    assert isinstance(report, databank.DatabankVerificationReport)
    assert isinstance(report.findings[0], databank.VerificationFinding)


#                                                        overlay_roots (union)
# =============================================================================
# Combining a read-only baseline databank with a separate learned-optimizer
# runs store: the two stores hold a disjoint slice of the requested schedules
# and load_databank_dataset(..., overlay_roots=[runs]) unions them lazily.


def _overlay_step() -> OptimizationStep:
    """A schedule absent from the `steps` fixture, standing in for a
    learned-optimizer run dumped into the separate runs store."""
    return OptimizationStep(optimizer="lbfgs")


def test_overlay_unions_schedules(tmp_path, tasks, steps):
    primary = tmp_path / "databank"
    runs = tmp_path / "runs"
    _create_zarr(
        primary,
        _write_raw_runs(
            primary / "raw", [(tasks[0], 12), (tasks[1], 12)], steps
        ),
    )
    ostep = _overlay_step()
    _create_zarr(
        runs,
        _write_raw_runs(
            runs / "raw", [(tasks[0], 12), (tasks[1], 12)], [ostep]
        ),
    )

    requested = steps + [ostep]
    ds = load_databank_dataset(
        "bbob", requested, primary, overlay_roots=[runs]
    ).load()

    assert {str(s) for s in ds[SCHEDULE].values} == {s.name for s in requested}
    assert ds.sizes[TASK_ID] == 2


def test_overlay_restricts_to_task_intersection(tmp_path, tasks, steps):
    primary = tmp_path / "databank"
    runs = tmp_path / "runs"
    _create_zarr(
        primary,
        _write_raw_runs(
            primary / "raw", [(tasks[0], 12), (tasks[1], 12)], steps
        ),
    )
    ostep = _overlay_step()
    # Runs store evaluated only the first task.
    _create_zarr(
        runs, _write_raw_runs(runs / "raw", [(tasks[0], 12)], [ostep])
    )

    ds = load_databank_dataset(
        "bbob", steps + [ostep], primary, overlay_roots=[runs]
    ).load()

    assert [str(t) for t in ds[TASK_ID].values] == [tasks[0].hash]
    assert {str(s) for s in ds[SCHEDULE].values} == {
        s.name for s in steps + [ostep]
    }


def test_overlay_dedups_primary_wins(tmp_path, tasks, steps):
    primary = tmp_path / "databank"
    runs = tmp_path / "runs"
    _create_zarr(
        primary,
        _write_raw_runs(
            primary / "raw", [(tasks[0], 12), (tasks[1], 12)], steps
        ),
    )
    ostep = _overlay_step()
    # Runs store carries a duplicate of adam plus the fresh lbfgs.
    _create_zarr(
        runs,
        _write_raw_runs(
            runs / "raw", [(tasks[0], 12), (tasks[1], 12)], [steps[1], ostep]
        ),
    )

    ds = load_databank_dataset(
        "bbob", steps + [ostep], primary, overlay_roots=[runs]
    ).load()

    names = [str(s) for s in ds[SCHEDULE].values]
    assert names.count("adam_lr=0.01") == 1  # primary wins, no duplicate
    assert set(names) == {s.name for s in steps + [ostep]}


def test_overlay_budget_grid_mismatch_raises(tmp_path, tasks, steps):
    primary = tmp_path / "databank"
    runs = tmp_path / "runs"
    _create_zarr(
        primary,
        _write_raw_runs(
            primary / "raw", [(tasks[0], 12), (tasks[1], 12)], steps
        ),
        budget_steps=BS,
    )
    ostep = _overlay_step()
    # A store dumped on a different budget grid must fail loudly, not
    # silently drop points (join="exact" inside the concat).
    _create_zarr(
        runs,
        _write_raw_runs(
            runs / "raw", [(tasks[0], 12), (tasks[1], 12)], [ostep]
        ),
        budget_steps=BS * 2,
    )

    with pytest.raises(ValueError):
        load_databank_dataset(
            "bbob", steps + [ostep], primary, overlay_roots=[runs]
        )


def test_overlay_performance_loader(tmp_path, tasks, steps):
    primary = tmp_path / "databank"
    runs = tmp_path / "runs"
    _create_zarr(
        primary,
        _write_raw_runs(
            primary / "raw", [(tasks[0], 12), (tasks[1], 12)], steps
        ),
    )
    ostep = _overlay_step()
    _create_zarr(
        runs,
        _write_raw_runs(
            runs / "raw", [(tasks[0], 12), (tasks[1], 12)], [ostep]
        ),
    )

    pl = load_databank_performance(
        "bbob", steps + [ostep], primary, overlay_roots=[runs]
    )
    assert pl.data.sizes[SCHEDULE] == len(steps) + 1


#                    Parallel dump: template / per-task / finalize
# =============================================================================
# The three-phase parallel dump must produce a byte-for-byte equivalent
# store to the serial streaming path. The per-task phase is exercised
# across real processes to validate the "one shard file per (task,
# schedule) cell, so disjoint tasks never touch the same file" contract
# that lets a SLURM array write one store concurrently.


def _dump_one_task_worker(run, plan):
    """Process-pool worker: region-write ONE task's cells into plan.store."""
    payloads = _dump_task_runs([run], plan)
    return payloads[0] if payloads else None


def _assert_stores_equal(store_a: Path, store_b: Path) -> None:
    """Assert two zarr databank stores are equal (NaN-aware, order-strict)."""
    a = xr.open_zarr(store_a, consolidated=True).load()
    b = xr.open_zarr(store_b, consolidated=True).load()
    assert set(a.data_vars) == set(b.data_vars)
    assert list(a[TASK_ID].values) == list(b[TASK_ID].values)
    assert list(a[SCHEDULE].values) == list(b[SCHEDULE].values)
    for v in a.data_vars:
        av = a[v]
        bv = b[v].transpose(*av.dims)
        if np.issubdtype(av.dtype, np.floating):
            np.testing.assert_allclose(
                av.values, bv.values, rtol=1e-6, atol=1e-7, equal_nan=True
            )
        else:
            assert np.array_equal(av.values, bv.values), v


def _parallel_dump(root: Path, runs, *, mode: str, concurrent: bool):
    """Drive template -> per-task (optionally concurrent) -> finalize."""
    plan = _dump_template_runs(
        runs,
        root,
        "bbob",
        mode=mode,
        easiest_quantile=0.75,
        taskset=None,
        budget_steps=BS,
    )
    if concurrent:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=4) as ex:
            payloads = list(
                ex.map(_dump_one_task_worker, list(runs), [plan] * len(runs))
            )
        payloads = [p for p in payloads if p is not None]
    else:
        payloads = [_dump_one_task_worker(r, plan) for r in runs]
        payloads = [p for p in payloads if p is not None]
    return dump_databank_finalize(payloads, plan)


def _many_task_runs(rundir: Path, steps, n_tasks: int = 6):
    """A wider set of tasks so the concurrent write actually contends."""
    tasks_ = [
        FakeTask(hash=f"task{ti:02d}", fn_name="sphere", dimensionality=10)
        for ti in range(n_tasks)
    ]
    return _write_raw_runs(rundir, [(t, 8) for t in tasks_], steps)


def test_parallel_dump_create_matches_serial(tmp_path, steps):
    """Concurrent per-task writes reproduce the serial streaming store."""
    runs = _many_task_runs(tmp_path / "runs", steps, n_tasks=6)

    serial_root = tmp_path / "serial"
    _create_zarr(serial_root, runs, budget_steps=BS)

    par_root = tmp_path / "parallel"
    _parallel_dump(par_root, runs, mode="error", concurrent=True)

    _assert_stores_equal(
        layout.zarr_store_path(par_root, "bbob"),
        layout.zarr_store_path(serial_root, "bbob"),
    )


def test_parallel_dump_append_matches_serial(tmp_path, tasks, steps):
    """Parallel append (tmp rebuild + carry-over) matches serial append."""
    base = _write_raw_runs(tmp_path / "base", [(tasks[0], 8)], steps)
    extra = _write_raw_runs(tmp_path / "extra", [(tasks[1], 8)], steps)

    # Serial: create then append.
    serial_root = tmp_path / "serial"
    _create_zarr(serial_root, base, budget_steps=BS)
    store_s = layout.zarr_store_path(serial_root, "bbob")
    rawroot_s = layout.raw_root(serial_root, "bbob")
    _dump_append(store_s, rawroot_s, extra, easiest_quantile=0.75)

    # Parallel: template(create)+finalize, then template(append)+finalize.
    par_root = tmp_path / "parallel"
    _parallel_dump(par_root, base, mode="error", concurrent=False)
    store_p = _parallel_dump(par_root, extra, mode="append", concurrent=False)

    _assert_stores_equal(Path(store_p), store_s)


def test_parallel_dump_template_is_shape_only(tmp_path):
    """The template loads only a 1-2 optimizer subset yet lays out a store
    byte-equal to the serial (full-optimizer) dump."""
    steps3 = [
        OptimizationStep(optimizer="randomsearch"),
        OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01}),
        OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.1}),
    ]
    task = FakeTask(hash="t0", fn_name="sphere", dimensionality=10)
    runs = _write_raw_runs(tmp_path / "runs", [(task, 8)], steps3)

    # The template's sample subset is randomsearch + one representative
    # optimizer -- NOT the task's full optimizer set.
    subset = _minimal_sample_steps(runs[0][1])
    assert len(subset) == 2
    assert any(s.optimizer == "randomsearch" for s in subset)
    assert any(s.optimizer != "randomsearch" for s in subset)

    serial_root = tmp_path / "serial"
    _create_zarr(serial_root, runs, budget_steps=BS)

    par_root = tmp_path / "parallel"
    _parallel_dump(par_root, runs, mode="error", concurrent=False)

    _assert_stores_equal(
        layout.zarr_store_path(par_root, "bbob"),
        layout.zarr_store_path(serial_root, "bbob"),
    )
