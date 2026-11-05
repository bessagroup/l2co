"""Noisy tasks are scored on their noise-free value (``docs/adr/0024``).

A run on a noisy task records two series: ``output_min``, the lowest
*observed* (noisy) loss per step, which is what the optimizer saw, and
``output_min_true``, the lowest *noise-free* loss of the same evaluated
population. Both are stored. Every score -- ERT targets (including the
random-search anchor of the easiest one), success, cost and the
``global_min`` breach record -- is computed from the noise-free series
when it is there, and from the observed one otherwise.

The toy noisy data here has an observed series 1000x *below* its
noise-free one, so scoring on the wrong series would show. The
noise-free data it is compared against is the same runs with the
noise-free series as their only series: scoring a noisy run must give
exactly the scores of that run. Noise-free tasks must be unaffected: no
new variable, no new file, the same scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from l2co import DataLoader
from l2co._src.data.report_helpers import shifted_final_min
from l2co._src.data_validation import _ert_from_quantile
from l2co._src.databank import layout
from l2co._src.databank.reader import (
    load_databank,
    load_databank_dataset,
    load_databank_performance,
    load_loss_raw,
)
from l2co._src.databank.records import _build_task_records
from l2co._src.databank.verify import verify_databank
from l2co._src.databank.writer import _interp_loss, _template_sample_run
from l2co._src.performance import (
    calculate_ert,
    compute_task_targets,
    global_min_breaches,
    shift_to_quality_value,
)
from l2co.data import scoring_loss
from l2co.history import (
    HistoryState,
    history_to_dataloader,
    history_to_xarray,
    history_to_xarray_realizations,
)
from l2co.naming import (
    BUDGET,
    COST,
    FEVALS,
    ITERATIONS,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    OUTPUT_STD,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
    TARGET_VALUE,
    TASK_ID,
    UPDATE_STEP,
)
from l2co.optimization import OptimizationStep
from tests.unit.test_databank import (
    BS,
    FakeTask,
    _assert_da_close,
    _by_check,
    _dump,
    _zarr_corrupt,
    make_loss_ds,
)

pytestmark = pytest.mark.filterwarnings("ignore")

#: The observed series of the toy noisy data is the noise-free one times
#: this: well below it, as a noisy minimum over a population tends to be.
DIP = 1e-3


#                                                                      Fixtures
# =============================================================================


@dataclass
class FakeNoisyTask(FakeTask):
    """A stand-in for a task with a stochastic loss: it carries a
    noise-free loss, as ``l2co_tasks.Task.loss_fn_true``."""

    @property
    def loss_fn_true(self):
        return lambda model: 0.0


def _noisy(ds: xr.Dataset) -> xr.Dataset:
    """``ds``'s loss as the noise-free series, observed dipping below."""
    return ds.assign(
        {OUTPUT_MIN: ds[OUTPUT_MIN] * DIP, OUTPUT_MIN_TRUE: ds[OUTPUT_MIN]}
    )


def _observed_only(ds: xr.Dataset) -> xr.Dataset:
    """A noisy dataset with its noise-free series dropped."""
    return ds.drop_vars(OUTPUT_MIN_TRUE)


@pytest.fixture
def steps() -> list[OptimizationStep]:
    return [
        OptimizationStep(optimizer="randomsearch"),
        OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01}),
    ]


def _write_runs(rundir: Path, specs, steps) -> list:
    """Raw loss netCDFs for ``specs = [(task, n_budget, noisy)]``."""
    rundir.mkdir(parents=True, exist_ok=True)
    runs = []
    for task, nb, noisy in specs:
        step_paths = {}
        for s in steps:
            ds = make_loss_ds(nb, scale=10.0)
            if noisy:
                ds = _noisy(ds)
            p = rundir / f"{task.hash}_{s.hash}.nc"
            ds.to_netcdf(p)
            step_paths[s] = p
        runs.append((task, step_paths))
    return runs


def _store(root: Path) -> Path:
    return layout.zarr_store_path(root, "bbob")


def _task_data(noisy: bool, n_budget: int = 9) -> xr.Dataset:
    """One task's runs concatenated over two schedules, as the dump does."""
    parts = []
    for name in ("randomsearch", "adam"):
        ds = make_loss_ds(n_budget, scale=10.0)
        parts.append(
            (_noisy(ds) if noisy else ds).expand_dims({SCHEDULE: [name]})
        )
    return xr.concat(parts, dim=SCHEDULE)


NOISY_TASK = FakeNoisyTask(hash="n015e7", fn_name="sphere", dimensionality=4)
CLEAN_TASK = FakeTask(hash="c1ea17", fn_name="rastrigin", dimensionality=4)
#: ``NOISY_TASK`` without its noise-free loss: the same task identity,
#: for the runs that hold the noise-free series as their only series.
NOISY_TASK_AS_CLEAN = FakeTask(
    hash=NOISY_TASK.hash, fn_name="sphere", dimensionality=4
)


#                                                               History export
# =============================================================================


def _history(with_true: bool, batched: bool = True) -> HistoryState:
    rng = np.random.default_rng(0)
    shape = (2, 6) if batched else (6,)
    true = jnp.asarray(rng.uniform(1.0, 5.0, shape))
    ints = jnp.ones(shape, dtype=int)
    return HistoryState.from_reduced(
        true * DIP,
        true,
        true * 0.1,
        ints,
        ints,
        ints,
        output_min_true=true if with_true else None,
    )


_SIX = {OUTPUT_MIN, OUTPUT_MEAN, OUTPUT_STD, UPDATE_STEP, FEVALS, ITERATIONS}


@pytest.mark.parametrize(
    "export, batched",
    [(history_to_xarray_realizations, True), (history_to_xarray, False)],
    ids=["realizations", "single"],
)
def test_history_export_carries_the_noise_free_series(export, batched):
    hs = _history(with_true=True, batched=batched)
    ds = export(hs)

    assert set(ds.data_vars) == _SIX | {OUTPUT_MIN_TRUE}
    assert ds[OUTPUT_MIN_TRUE].dims == ds[OUTPUT_MIN].dims
    # ``output_dim`` leads; the rest is the buffer's own shape.
    np.testing.assert_array_equal(
        ds[OUTPUT_MIN_TRUE].values.squeeze(0), np.asarray(hs.output_min_true)
    )


@pytest.mark.parametrize(
    "export, batched",
    [(history_to_xarray_realizations, True), (history_to_xarray, False)],
    ids=["realizations", "single"],
)
def test_history_export_of_a_noise_free_run_is_unchanged(export, batched):
    ds = export(_history(with_true=False, batched=batched))
    assert set(ds.data_vars) == _SIX


def test_history_export_interpolates_the_noise_free_series():
    ds = history_to_xarray_realizations(
        _history(with_true=True), budget_steps=4
    )
    assert ds.sizes[BUDGET] == 4
    assert ds[OUTPUT_MIN_TRUE].dims == ds[OUTPUT_MIN].dims


def test_history_to_dataloader_scores_the_noise_free_series():
    class _Task:
        hash = "tt"
        tag = {"fn_name": "x", "dimensionality": 2}
        global_min = 1.0

    step = OptimizationStep(optimizer="adam")
    hs = _history(with_true=True)
    noisy = history_to_dataloader(hs, _Task(), step)
    clean = history_to_dataloader(
        HistoryState.from_reduced(
            hs.output_min_true,
            hs.output_mean,
            hs.output_std,
            hs.update_step,
            hs.fevals,
            hs.iterations,
        ),
        _Task(),
        step,
    )

    assert OUTPUT_MIN_TRUE in noisy.data
    for var in (SUCCESS, COST):
        xr.testing.assert_identical(noisy.data[var], clean.data[var])


#                                                                 scoring_loss
# =============================================================================


def test_scoring_loss_without_the_series_is_output_min():
    ds = _task_data(noisy=False)
    xr.testing.assert_identical(scoring_loss(ds), ds[OUTPUT_MIN])


def test_scoring_loss_prefers_the_noise_free_series():
    ds = _task_data(noisy=True)
    xr.testing.assert_identical(scoring_loss(ds), ds[OUTPUT_MIN_TRUE])


def test_scoring_loss_falls_back_per_task_where_the_series_is_absent():
    """A task none of whose runs has the series (a noise-free task's
    cells of a mixed store, read back as NaN) is scored on its observed
    series, next to a noisy task scored on its noise-free one."""
    noisy = _task_data(noisy=True)
    clean = _task_data(noisy=True)
    clean = clean.assign({OUTPUT_MIN_TRUE: clean[OUTPUT_MIN_TRUE] * np.nan})
    ds = xr.concat([noisy, clean], dim=TASK_ID).assign_coords(
        {TASK_ID: ["noisy", "clean"]}
    )

    got = scoring_loss(ds)
    xr.testing.assert_equal(
        got.sel({TASK_ID: "noisy"}),
        ds[OUTPUT_MIN_TRUE].sel({TASK_ID: "noisy"}),
    )
    xr.testing.assert_equal(
        got.sel({TASK_ID: "clean"}), ds[OUTPUT_MIN].sel({TASK_ID: "clean"})
    )


def test_scoring_loss_refuses_a_task_with_the_series_on_some_runs_only():
    """One task's runs scored on two series, silently: refused. The dump
    refuses it per store; this catches it where data is combined after
    (an overlay, a joined loader)."""
    ds = _task_data(noisy=True)
    ds[OUTPUT_MIN_TRUE].loc[{SCHEDULE: "adam"}] = np.nan

    with pytest.raises(ValueError, match="some of its runs"):
        scoring_loss(ds)


def test_scoring_loss_ignores_unpopulated_runs_of_a_noisy_task():
    """A run with neither series (an unwritten pair, NaN on read) is not
    a run without the noise-free series."""
    ds = _task_data(noisy=True)
    for var in (OUTPUT_MIN, OUTPUT_MIN_TRUE):
        ds[var].loc[{SCHEDULE: "adam"}] = np.nan

    xr.testing.assert_identical(scoring_loss(ds), ds[OUTPUT_MIN_TRUE])


def test_legacy_preprocess_refuses_a_task_with_the_series_on_some_runs():
    """``create_combined_odata`` scores each task through
    ``_ert_from_quantile``; concatenating a schedule without the series
    NaN-fills it there, which the scoring rule refuses."""
    parts = [
        _noisy(make_loss_ds(9, 10.0)).expand_dims(
            {SCHEDULE: ["randomsearch"]}
        ),
        make_loss_ds(9, 10.0).expand_dims({SCHEDULE: ["adam"]}),
    ]
    ds = xr.concat(parts, dim=SCHEDULE, join="outer")
    assert OUTPUT_MIN_TRUE in ds

    with pytest.raises(ValueError, match="some of its runs"):
        _ert_from_quantile(ds, easiest_quantile=0.75)


#                                                     Targets, ERT and breaches
# =============================================================================


def test_shift_scores_the_noise_free_series_and_drops_it():
    got = shift_to_quality_value(_task_data(noisy=True), 0.0)
    want = shift_to_quality_value(_task_data(noisy=False), 0.0)
    assert OUTPUT_MIN_TRUE not in got
    xr.testing.assert_identical(got, want)


def test_shift_of_noise_free_data_is_the_old_formula():
    """Pinned: the shift a noise-free task got before this change."""
    ds = _task_data(noisy=False)
    for g_min in (0.0, 1e-3, np.nan):
        g = float(ds[OUTPUT_MIN].min()) if np.isnan(g_min) else g_min
        want = ds.assign({OUTPUT_MIN: (ds[OUTPUT_MIN] - g).clip(min=0.0)})
        xr.testing.assert_identical(shift_to_quality_value(ds, g_min), want)


def test_nan_global_min_falls_back_to_the_noise_free_minimum():
    got = shift_to_quality_value(_task_data(noisy=True), np.nan)
    want = shift_to_quality_value(_task_data(noisy=False), np.nan)
    xr.testing.assert_identical(got, want)


def test_calculate_ert_scores_the_noise_free_series():
    targets = np.logspace(-8, 2, 11)
    noisy = calculate_ert(_task_data(True), targets=targets, global_min=0.0)
    clean = calculate_ert(_task_data(False), targets=targets, global_min=0.0)
    observed = calculate_ert(
        _observed_only(_task_data(True)), targets=targets, global_min=0.0
    )

    xr.testing.assert_identical(noisy, clean)
    # Not vacuous: on the observed series more targets count as solved.
    assert int(observed[SUCCESS].sum()) > int(noisy[SUCCESS].sum())


def test_random_search_anchor_uses_the_noise_free_series():
    """The easiest target is a quantile of random search's losses; on a
    noisy task those are its noise-free losses."""
    noisy = _ert_from_quantile(_task_data(True), easiest_quantile=0.75)
    clean = _ert_from_quantile(_task_data(False), easiest_quantile=0.75)
    observed = _ert_from_quantile(
        _observed_only(_task_data(True)), easiest_quantile=0.75
    )

    xr.testing.assert_identical(noisy, clean)
    assert not np.allclose(
        observed[TARGET_VALUE].values, noisy[TARGET_VALUE].values
    )


def test_compute_task_targets_refuses_unshifted_noisy_data():
    with pytest.raises(ValueError, match="shift_to_quality_value"):
        compute_task_targets(_task_data(noisy=True))


def test_breaches_are_judged_on_the_noise_free_series():
    """An observed value below ``global_min`` is expected noise on a
    noisy task, not a ``global_min`` to re-tune."""
    ds = _task_data(noisy=True)
    g_min = float(ds[OUTPUT_MIN_TRUE].min()) / 2  # between the two series

    assert global_min_breaches(_observed_only(ds), g_min)
    assert global_min_breaches(ds, g_min) == {}


#                                                                      Records
# =============================================================================


def test_records_of_a_noisy_task_score_the_noise_free_series(steps):
    noisy = {s: _noisy(make_loss_ds(8, 10.0)) for s in steps}
    clean = {s: make_loss_ds(8, 10.0) for s in steps}
    task_n, loss_n, perf_n, _ = _build_task_records(NOISY_TASK, noisy, 0.75)
    task_c, _, perf_c, _ = _build_task_records(
        NOISY_TASK_AS_CLEAN, clean, 0.75
    )

    xr.testing.assert_identical(task_n[TARGET_VALUE], task_c[TARGET_VALUE])
    for s in steps:
        xr.testing.assert_identical(perf_n[s.hash], perf_c[s.hash])
        # both series are kept in the loss payload
        assert {OUTPUT_MIN, OUTPUT_MIN_TRUE} <= set(loss_n[s.hash].data_vars)


def test_records_refuse_a_task_with_the_series_on_some_runs_only(steps):
    """A schedule without the series would silently score on its
    observed values next to schedules scored on noise-free ones."""
    ds_by_step = {
        steps[0]: _noisy(make_loss_ds(8, 10.0)),
        steps[1]: make_loss_ds(8, 10.0),
    }
    with pytest.raises(ValueError, match="noise-free series"):
        _build_task_records(CLEAN_TASK, ds_by_step, 0.75)


def test_records_refuse_a_noisy_task_without_the_series(steps):
    ds_by_step = {s: make_loss_ds(8, 10.0) for s in steps}
    with pytest.raises(ValueError, match="loss_fn_true"):
        _build_task_records(NOISY_TASK, ds_by_step, 0.75)


def test_records_refuse_the_series_on_a_task_without_a_noise_free_loss(
    steps,
):
    """The store takes the variable from its noisy tasks, those with a
    ``loss_fn_true`` (:func:`_template_sample_run`); a series on any
    other task is refused rather than written into a store sized
    without it."""
    ds_by_step = {s: _noisy(make_loss_ds(8, 10.0)) for s in steps}
    with pytest.raises(ValueError, match="loss_fn_true"):
        _build_task_records(NOISY_TASK_AS_CLEAN, ds_by_step, 0.75)


#                                                                     Databank
# =============================================================================


def test_noisy_store_round_trips_both_series(tmp_path, steps):
    runs = _write_runs(tmp_path / "runs", [(NOISY_TASK, 9, True)], steps)
    _dump(tmp_path, runs, mode="error")

    ds = load_databank_dataset("bbob", steps, tmp_path)
    assert OUTPUT_MIN_TRUE in ds
    for s in steps:
        raw = xr.load_dataset(runs[0][1][s])
        cell = ds.sel({TASK_ID: NOISY_TASK.hash, SCHEDULE: s.name})
        want = _interp_loss(raw, BS)
        for var in (OUTPUT_MIN, OUTPUT_MIN_TRUE):
            _assert_da_close(want[var], cell[var])
        sidecar = load_loss_raw("bbob", tmp_path, NOISY_TASK.hash, s.hash)
        _assert_da_close(raw[OUTPUT_MIN_TRUE], sidecar[OUTPUT_MIN_TRUE])

    # the performance-only loader stays an allow-list
    pl = load_databank_performance("bbob", steps, tmp_path)
    assert OUTPUT_MIN_TRUE not in pl.data


def test_noisy_store_scores_equal_the_noise_free_runs(tmp_path, steps):
    """The stored targets, success and cost of a noisy store are those of
    a store holding its noise-free series as the only series; so are the
    loader's final-loss scores."""
    noisy_root, clean_root = tmp_path / "noisy", tmp_path / "clean"
    _dump(
        noisy_root,
        _write_runs(noisy_root / "runs", [(NOISY_TASK, 9, True)], steps),
        mode="error",
    )
    _dump(
        clean_root,
        _write_runs(
            clean_root / "runs", [(NOISY_TASK_AS_CLEAN, 9, False)], steps
        ),
        mode="error",
    )
    noisy = load_databank_dataset("bbob", steps, noisy_root).load()
    clean = load_databank_dataset("bbob", steps, clean_root).load()
    for var in (TARGET_VALUE, SUCCESS, COST):
        xr.testing.assert_identical(noisy[var], clean[var])

    dl_n = load_databank("bbob", steps, noisy_root)
    dl_c = load_databank("bbob", steps, clean_root)
    xr.testing.assert_identical(
        dl_n.performance[OUTPUT_MIN].load(),
        dl_c.performance[OUTPUT_MIN].load(),
    )
    # equal, not identical: the noisy one is named after its series
    xr.testing.assert_equal(
        shifted_final_min(dl_n.data, dl_n.task_labels).load(),
        shifted_final_min(dl_c.data, dl_c.task_labels).load(),
    )


def test_noisy_store_writes_no_breach_record_for_noise(tmp_path, steps):
    task = FakeNoisyTask(
        hash="n015e7", fn_name="sphere", dimensionality=4, global_min=5e-4
    )
    runs = _write_runs(tmp_path / "runs", [(task, 9, True)], steps)
    _dump(tmp_path, runs, mode="error")
    path = layout.global_min_breaches_path(tmp_path, "bbob")
    assert json.loads(path.read_text()) == {}


def test_noise_free_store_has_no_noise_free_variable(tmp_path, steps):
    runs = _write_runs(tmp_path / "runs", [(CLEAN_TASK, 9, False)], steps)
    store = _dump(tmp_path, runs, mode="error")

    assert OUTPUT_MIN_TRUE not in load_databank_dataset(
        "bbob", steps, tmp_path
    )
    assert not (store / OUTPUT_MIN_TRUE).exists()


def test_template_sample_is_chosen_without_reading_runs(tmp_path, steps):
    """The template's sample is the first noisy task, told by its
    ``loss_fn_true``: no run file is opened to choose it, so a store of
    noise-free tasks costs no extra reads."""
    missing = {s: tmp_path / f"missing_{s.hash}.nc" for s in steps}
    clean, noisy = (CLEAN_TASK, missing), (NOISY_TASK, missing)

    assert _template_sample_run([clean, noisy])[0] is NOISY_TASK
    assert _template_sample_run([clean])[0] is CLEAN_TASK


@pytest.mark.parametrize("noisy_first", [True, False], ids=["noisy", "clean"])
def test_mixed_store_in_either_order(tmp_path, steps, noisy_first):
    """The store's variables follow every run, not the first one: a
    noise-free task first must not leave the series out of the store."""
    specs = [(NOISY_TASK, 9, True), (CLEAN_TASK, 7, False)]
    if not noisy_first:
        specs.reverse()
    runs = _write_runs(tmp_path / "runs", specs, steps)
    store = _dump(tmp_path, runs, mode="error")

    ds = load_databank_dataset("bbob", steps, tmp_path).load()
    true = ds[OUTPUT_MIN_TRUE]
    assert true.sel({TASK_ID: NOISY_TASK.hash}).notnull().all()
    assert true.sel({TASK_ID: CLEAN_TASK.hash}).isnull().all()
    # One chunk file per noisy cell, none for the noise-free task's.
    chunks = [p for p in (store / OUTPUT_MIN_TRUE).rglob("*") if p.is_file()]
    chunks = [p for p in chunks if p.name != "zarr.json"]
    assert len(chunks) == len(steps)

    # The noise-free task scores exactly as in a store of its own.
    alone = tmp_path / "alone"
    _dump(
        alone,
        _write_runs(alone / "runs", [(CLEAN_TASK, 7, False)], steps),
        mode="error",
    )
    own = load_databank_dataset("bbob", steps, alone).load()
    for var in (TARGET_VALUE, SUCCESS, COST, OUTPUT_MIN):
        xr.testing.assert_identical(
            ds[var].sel({TASK_ID: [CLEAN_TASK.hash]}), own[var]
        )


def test_append_noise_free_runs_to_a_store_with_the_series(tmp_path, steps):
    _dump(
        tmp_path,
        _write_runs(tmp_path / "r1", [(NOISY_TASK, 9, True)], steps),
        mode="error",
    )
    before = load_databank_dataset("bbob", steps, tmp_path).load()
    _dump(
        tmp_path,
        _write_runs(tmp_path / "r2", [(CLEAN_TASK, 7, False)], steps),
        mode="append",
    )
    after = load_databank_dataset("bbob", steps, tmp_path).load()

    assert (
        after[OUTPUT_MIN_TRUE].sel({TASK_ID: CLEAN_TASK.hash}).isnull().all()
    )
    xr.testing.assert_identical(
        after[OUTPUT_MIN_TRUE].sel({TASK_ID: [NOISY_TASK.hash]}),
        before[OUTPUT_MIN_TRUE],
    )
    assert verify_databank(tmp_path, "bbob").ok


def test_append_noisy_runs_to_a_store_with_the_series(tmp_path, steps):
    """A new schedule on a stored noisy task and a new noisy task: their
    series are written, while the stored cells and the stored task's
    targets stay as they were."""
    sgd = [OptimizationStep(optimizer="sgd", hyperparameters={"lr": 0.02})]
    other = FakeNoisyTask(hash="0th3r1", fn_name="ellipsoid", dimensionality=4)
    _dump(
        tmp_path,
        _write_runs(tmp_path / "r1", [(NOISY_TASK, 9, True)], steps),
        mode="error",
    )
    before = load_databank_dataset("bbob", steps, tmp_path).load()
    incoming = _write_runs(
        tmp_path / "r2", [(NOISY_TASK, 9, True)], sgd
    ) + _write_runs(tmp_path / "r3", [(other, 8, True)], steps + sgd)
    _dump(tmp_path, incoming, mode="append")
    after = load_databank_dataset("bbob", steps + sgd, tmp_path).load()

    stored = {TASK_ID: [NOISY_TASK.hash], SCHEDULE: [s.name for s in steps]}
    xr.testing.assert_equal(
        after[OUTPUT_MIN_TRUE].sel(stored), before[OUTPUT_MIN_TRUE]
    )
    xr.testing.assert_equal(
        after[TARGET_VALUE].sel({TASK_ID: [NOISY_TASK.hash]}),
        before[TARGET_VALUE],
    )
    for task, step_paths in incoming:
        for s, path in step_paths.items():
            want = _interp_loss(xr.load_dataset(path), BS)[OUTPUT_MIN_TRUE]
            cell = after.sel({TASK_ID: task.hash, SCHEDULE: s.name})
            _assert_da_close(want, cell[OUTPUT_MIN_TRUE])
    assert verify_databank(tmp_path, "bbob").ok


@pytest.mark.parametrize(
    "incoming",
    [[(NOISY_TASK, 9, True)], [(CLEAN_TASK, 7, False), (NOISY_TASK, 9, True)]],
    ids=["noisy", "noisy-after-clean"],
)
def test_append_noisy_runs_to_a_store_without_the_series_is_refused(
    tmp_path, steps, incoming
):
    """The store's noisy cells, if any, and its targets were scored on
    observed values; mixing would score one task two ways. Refused before
    the store is touched."""
    seed = FakeTask(hash="5eed00", fn_name="sphere", dimensionality=4)
    _dump(
        tmp_path,
        _write_runs(tmp_path / "r1", [(seed, 7, False)], steps),
        mode="error",
    )
    files_before = sorted(p for p in tmp_path.joinpath("bbob").rglob("*"))
    with pytest.raises(ValueError, match="noise-free series"):
        _dump(
            tmp_path,
            _write_runs(tmp_path / "r2", incoming, steps),
            mode="append",
        )
    assert sorted(p for p in tmp_path.joinpath("bbob").rglob("*")) == (
        files_before
    )
    assert not layout.append_marker_path(tmp_path, "bbob").exists()


#                                                                       Verify
# =============================================================================


def _mixed_store(root: Path, steps) -> Path:
    runs = _write_runs(
        root / "runs", [(CLEAN_TASK, 7, False), (NOISY_TASK, 9, True)], steps
    )
    return _dump(root, runs, mode="error")


def _series_findings(report):
    return [f for f in report.findings if f.variable == OUTPUT_MIN_TRUE]


def test_verify_mixed_store_ok(tmp_path, steps):
    _mixed_store(tmp_path, steps)
    report = verify_databank(tmp_path, "bbob")

    assert report.ok
    assert report.warnings == ()
    assert _series_findings(report) == []


def test_verify_noise_free_store_says_nothing_about_the_series(
    tmp_path, steps
):
    _dump(
        tmp_path,
        _write_runs(tmp_path / "runs", [(CLEAN_TASK, 7, False)], steps),
        mode="error",
    )
    report = verify_databank(tmp_path, "bbob")
    assert report.ok
    assert _series_findings(report) == []


def test_verify_flags_a_task_with_the_series_on_some_pairs_only(
    tmp_path, steps
):
    store = _mixed_store(tmp_path, steps)
    _zarr_corrupt(
        store,
        OUTPUT_MIN_TRUE,
        np.nan,
        **{TASK_ID: NOISY_TASK.hash, SCHEDULE: steps[1].name},
    )
    report = verify_databank(tmp_path, "bbob")

    assert not report.ok
    (finding,) = _by_check(report, "task.noise_free_partial")
    assert finding.task_id == NOISY_TASK.hash


def test_verify_warns_on_scattered_nonfinite_noise_free_values(
    tmp_path, steps
):
    store = _mixed_store(tmp_path, steps)
    _zarr_corrupt(
        store,
        OUTPUT_MIN_TRUE,
        np.inf,
        **{
            TASK_ID: NOISY_TASK.hash,
            SCHEDULE: steps[0].name,
            REALIZATION: 0,
            BUDGET: 3,
        },
    )
    report = verify_databank(tmp_path, "bbob")

    (finding,) = _series_findings(report)
    assert finding.check == "loss.scattered_nonfinite"
    assert report.ok


#                                                                   DataLoader
# =============================================================================


def test_dataloader_join_of_noisy_and_noise_free_loaders(tmp_path, steps):
    """Concatenating a noisy and a noise-free loader NaN-fills the series
    on the noise-free side, which then scores on its observed one."""
    a, b = tmp_path / "a", tmp_path / "b"
    runs_a = _write_runs(a / "runs", [(NOISY_TASK, 9, True)], steps)
    runs_b = _write_runs(b / "runs", [(CLEAN_TASK, 9, False)], steps)
    _dump(a, runs_a, mode="error")
    _dump(b, runs_b, mode="error")
    dl = load_databank("bbob", steps, a) + load_databank("bbob", steps, b)
    assert isinstance(dl, DataLoader)

    scored = scoring_loss(dl.data).load()
    data = dl.data.load()
    xr.testing.assert_equal(
        scored.sel({TASK_ID: CLEAN_TASK.hash}),
        data[OUTPUT_MIN].sel({TASK_ID: CLEAN_TASK.hash}),
    )
    xr.testing.assert_equal(
        scored.sel({TASK_ID: NOISY_TASK.hash}),
        data[OUTPUT_MIN_TRUE].sel({TASK_ID: NOISY_TASK.hash}),
    )


#: A schedule only the second source of a combined read holds.
SGD = [OptimizationStep(optimizer="sgd", hyperparameters={"lr": 0.02})]


def _noisy_store_and_one_without_the_series(tmp_path, steps):
    """Two stores holding the same noisy task: the first with the
    series, the second (made by a path that does not record it, e.g.
    rl2co's environment) with another schedule and observed values
    only."""
    a, b = tmp_path / "a", tmp_path / "b"
    _dump(
        a,
        _write_runs(a / "runs", [(NOISY_TASK, 9, True)], steps),
        mode="error",
    )
    _dump(
        b,
        _write_runs(b / "runs", [(NOISY_TASK_AS_CLEAN, 9, False)], SGD),
        mode="error",
    )
    return a, b


def test_overlay_scoring_a_task_on_two_series_is_refused(tmp_path, steps):
    """Each store passed the dump's check on its own; their union would
    score one task's schedules on different series."""
    a, b = _noisy_store_and_one_without_the_series(tmp_path, steps)

    ds = load_databank_dataset("bbob", steps + SGD, a, overlay_roots=[b])
    with pytest.raises(ValueError, match=NOISY_TASK.hash):
        scoring_loss(ds)
    dl = load_databank("bbob", steps + SGD, a, overlay_roots=[b])
    with pytest.raises(ValueError, match="some of its runs"):
        dl.performance  # noqa: B018


def test_dataloader_join_scoring_a_task_on_two_series_is_refused(
    tmp_path, steps
):
    a, b = _noisy_store_and_one_without_the_series(tmp_path, steps)

    dl = load_databank("bbob", steps, a).join(load_databank("bbob", SGD, b))
    with pytest.raises(ValueError, match=NOISY_TASK.hash):
        scoring_loss(dl.data)
