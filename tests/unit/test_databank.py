"""Unit tests for the per-record databank (:mod:`l2co._src.databank`).

Covers the read path, path selection, tasksets, budget resampling, and
the f3dasm-free core of the writer (``_build_task_records``). The full
``dump_databank`` f3dasm preamble is exercised end-to-end by the
``create_optimization_dataset`` pipeline, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from l2co import DataLoader, databank
from l2co._src.databank import layout
from l2co._src.databank.reader import (
    load_databank,
    load_databank_dataset,
    loss_preprocess,
)
from l2co._src.databank.tasksets import load_taskset, save_taskset
from l2co._src.databank.writer import _build_task_records
from l2co.naming import (
    BUDGET,
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

#                                                                      Fixtures
# =============================================================================

N_REAL = 3


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
    # decreasing loss per realization, slightly offset by realization
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


def _write_synthetic_databank(rootdir, tasks, steps, task_budgets):
    """Write a full per-record databank via the writer's core helper.

    ``task_budgets[i]`` is the raw budget length for task ``i``. All
    optimizers within a task share a budget length (as in the real
    pipeline, where budget depends on the task, not the optimizer);
    budget varies *across* tasks, which exercises read-time resampling.
    """
    for sub in (
        layout.TASK_DIR,
        layout.OPTIMIZER_DIR,
        layout.LOSS_DIR,
        layout.PERF_DIR,
    ):
        (rootdir / sub).mkdir(parents=True, exist_ok=True)

    for ti, task in enumerate(tasks):
        ds_by_step = {
            s: make_loss_ds(task_budgets[ti], scale=10.0 ** (ti + 1))
            for s in steps
        }
        task_ds, loss_by_hash, perf_by_hash, opt_by_hash = _build_task_records(
            task, ds_by_step, easiest_quantile=0.75
        )
        task_ds.to_netcdf(layout.task_path(rootdir, task.hash))
        for h in loss_by_hash:
            loss_by_hash[h].to_netcdf(layout.loss_path(rootdir, task.hash, h))
            perf_by_hash[h].to_netcdf(layout.perf_path(rootdir, task.hash, h))
            opt_by_hash[h].to_netcdf(layout.optimizer_path(rootdir, h))


@pytest.fixture
def databank_root(tmp_path, monkeypatch, tasks, steps):
    """A populated databank rooted under tmp_path, reachable via "local"."""
    monkeypatch.setitem(layout._LOCATION_ROOTS, "local", tmp_path)
    rootdir = layout.databank_root("local", "bbob")
    task_budgets = [7, 13]  # per-task budget lengths (differ across tasks)
    _write_synthetic_databank(rootdir, tasks, steps, task_budgets)
    return rootdir


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
    assert SUCCESS in perf and "cost" in perf
    assert TARGET in perf[SUCCESS].dims
    assert METRIC in perf["cost"].dims
    assert TARGET_VALUE not in perf  # lives in the task file only
    assert bool(perf[SUCCESS].dtype == bool)

    opt = opt_by_hash[rs_hash]
    assert set(opt[OPTIMIZATIONSTEP].dims) == {SCHEDULE}
    assert str(opt[SCHEDULE].values[0]) == "randomsearch"


#                                                                  Reader path
# =============================================================================


def test_load_databank_dataset_roundtrip(databank_root, steps, tasks):
    ds = load_databank_dataset("bbob", steps, "local", budget_steps=5)

    assert ds.sizes[TASK_ID] == len(tasks)
    assert ds.sizes[SCHEDULE] == len(steps)
    assert ds.sizes[BUDGET] == 5  # resampled to common grid
    for var in (
        OUTPUT_MIN,
        FEVALS,
        SUCCESS,
        "cost",
        TASK_LABELS,
        TARGET_VALUE,
    ):
        assert var in ds
    assert set(str(s) for s in ds[SCHEDULE].values) == {
        "randomsearch",
        "adam_lr=0.01",
    }
    # int counters survive the resample as ints
    assert np.issubdtype(ds[FEVALS].dtype, np.integer)


def test_load_databank_returns_compatible_dataloader(databank_root, steps):
    d = load_databank("bbob", steps, "local", budget_steps=5)
    assert isinstance(d, DataLoader)
    # stacked into PROBLEM, optimizationstep dropped, schedules present
    assert "problem" in d.data.dims
    assert OPTIMIZATIONSTEP not in d.data
    assert d.data.sizes[SCHEDULE] == len(steps)
    # task labels carried through and parsed (mixed-type -> object dtype)
    assert FEATURE in d.task_labels.dims
    assert "fn_name" in list(d.task_labels[FEATURE].values)


def test_load_databank_dataset_subset_by_task(databank_root, steps, tasks):
    ds = load_databank_dataset(
        "bbob", steps, "local", task_ids=[tasks[0].hash], budget_steps=5
    )
    assert ds.sizes[TASK_ID] == 1
    assert str(ds[TASK_ID].values[0]) == tasks[0].hash


def test_load_databank_dataset_facade(databank_root, steps):
    ds = databank.load_databank_dataset("bbob", steps, "local", budget_steps=5)
    assert ds.sizes[SCHEDULE] == len(steps)


#                                                                Path selection
# =============================================================================


def test_select_paths_existing_and_missing(databank_root, steps, tasks):
    task_ids = [t.hash for t in tasks]
    hashes = [s.hash for s in steps]
    # one extra optimizer hash that was never written
    hashes_with_gap = hashes + [123456]

    paths = layout.select_paths(databank_root, task_ids, hashes_with_gap)
    assert len(paths["loss_history"]) == len(task_ids) * len(hashes)
    assert len(paths["loss_history_missing"]) == len(task_ids)  # 2 gaps
    assert len(paths["task"]) == len(task_ids)
    assert len(paths["optimizer"]) == len(hashes)


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
