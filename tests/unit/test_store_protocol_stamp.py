"""Stores carry a protocol stamp, and the databank refuses another protocol.

The dataset paper's protocol revision re-runs every store under a new
run protocol (budget, starting points, noise-free scoring, ...). Numbers
written under two protocols are not comparable, and the new stores keep
the old names under a new root (re-run plan, decision D1). So each store
carries a stamp -- a small JSON file inside it, next to the append marker
-- and l2co refuses to read, append to or replace a store whose stamp is
missing or names another protocol. ``verify_databank`` audits any store
and reports the stamp as a finding instead. ADR 0025.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from l2co import databank
from l2co._src.databank import layout
from l2co._src.databank.reader import (
    load_databank_dataset,
    load_databank_performance,
    populated_iterations,
    populated_pairs,
)
from l2co._src.databank.stamp import (
    DATABANK_PROTOCOL,
    FStarError,
    RunSettingsError,
    StoreProtocolError,
    check_f_star,
    check_run_settings,
    store_stamp,
)
from l2co._src.databank.writer import (
    _dump_task_runs,
    _dump_template_runs,
    dump_databank_finalize,
)
from l2co._src.databank.verify import verify_databank

from .test_databank import (  # noqa: F401  (fixtures)
    BS,
    _append_zarr,
    _create_zarr,
    _dump,
    _write_raw_runs,
    steps,
    tasks,
)

pytestmark = pytest.mark.filterwarnings("ignore")

NAME = "bbob"  # the dataset name the shared ``_dump`` helper writes


def _stamp_file(root: Path) -> Path:
    return layout.zarr_store_path(root, NAME) / layout.STAMP_NAME


def _store(tmp_path, tasks, steps) -> Path:
    runs = _write_raw_runs(tmp_path / "runs", [(t, 20) for t in tasks], steps)
    root = tmp_path / "db"
    _create_zarr(root, runs)
    return root


def _make_unstamped(root: Path) -> None:
    """What every store written before stamps existed looks like."""
    _stamp_file(root).unlink()


def _restamp(root: Path, protocol: str) -> None:
    path = _stamp_file(root)
    stamp = json.loads(path.read_text())
    stamp["protocol"] = protocol
    path.write_text(json.dumps(stamp))


def _snapshot(store: Path) -> dict[str, int]:
    """Every file in the store with its size and mtime, to show it was
    left exactly as it was."""
    return {
        str(p.relative_to(store)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(store.rglob("*"))
        if p.is_file()
    }


#                                                              writing a stamp
# =============================================================================


def test_a_new_store_is_stamped_with_the_running_protocol(
    tmp_path, tasks, steps
):
    root = _store(tmp_path, tasks, steps)
    stamp = json.loads(_stamp_file(root).read_text())
    assert stamp["protocol"] == DATABANK_PROTOCOL
    assert set(stamp) == {"protocol", "l2co_version", "created"}
    assert store_stamp(root, NAME) == stamp


def test_the_stamp_survives_an_append(tmp_path, tasks, steps):
    runs_a = _write_raw_runs(tmp_path / "a", [(tasks[0], 20)], steps)
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 20)], steps)
    root = tmp_path / "db"
    _create_zarr(root, runs_a)
    before = _stamp_file(root).read_text()
    _append_zarr(root, runs_b)
    assert _stamp_file(root).read_text() == before
    assert populated_pairs(root, NAME)  # still readable


def test_replace_over_the_same_protocol_still_works(tmp_path, tasks, steps):
    root = _store(tmp_path, tasks, steps)
    runs = _write_raw_runs(tmp_path / "again", [(tasks[0], 20)], steps)
    _dump(root, runs, mode="replace")
    assert store_stamp(root, NAME)["protocol"] == DATABANK_PROTOCOL
    assert {t for t, _ in populated_pairs(root, NAME)} == {tasks[0].hash}


def test_error_mode_on_an_existing_store_is_unchanged(tmp_path, tasks, steps):
    root = _store(tmp_path, tasks, steps)
    _make_unstamped(root)
    runs = _write_raw_runs(tmp_path / "again", [(tasks[0], 20)], steps)
    with pytest.raises(FileExistsError):
        _dump(root, runs, mode="error")


#                                                                     reading
# =============================================================================

READERS = {
    "populated_pairs": lambda root: populated_pairs(root, NAME),
    "populated_iterations": lambda root: populated_iterations(root, NAME),
    "load_databank_dataset": lambda root: load_databank_dataset(
        NAME, ["randomsearch"], root
    ),
    "load_databank_performance": lambda root: load_databank_performance(
        NAME, ["randomsearch"], root
    ),
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_reads_refuse_an_unstamped_store(tmp_path, tasks, steps, reader):
    root = _store(tmp_path, tasks, steps)
    _make_unstamped(root)
    with pytest.raises(StoreProtocolError) as err:
        READERS[reader](root)
    message = str(err.value)
    assert "no protocol stamp" in message
    assert DATABANK_PROTOCOL in message
    assert "1.8" in message


@pytest.mark.parametrize("reader", sorted(READERS))
def test_reads_refuse_another_protocol(tmp_path, tasks, steps, reader):
    root = _store(tmp_path, tasks, steps)
    _restamp(root, "some-older-protocol")
    with pytest.raises(StoreProtocolError) as err:
        READERS[reader](root)
    assert "some-older-protocol" in str(err.value)
    assert DATABANK_PROTOCOL in str(err.value)


@pytest.mark.parametrize("protocol", [None, "some-older-protocol"])
def test_raw_histories_are_refused_with_their_store(
    tmp_path, tasks, steps, protocol
):
    """``load_loss_raw`` reads the raw sidecar, not the zarr store, but the
    sidecar belongs to the store and was written under its protocol."""
    root = _store(tmp_path, tasks, steps)
    task_id, step_hash = tasks[0].hash, steps[0].hash
    assert databank.load_loss_raw(NAME, root, task_id, step_hash) is not None
    if protocol is None:
        _make_unstamped(root)
    else:
        _restamp(root, protocol)
    with pytest.raises(StoreProtocolError, match=DATABANK_PROTOCOL):
        databank.load_loss_raw(NAME, root, task_id, step_hash)


def test_an_unreadable_stamp_is_refused(tmp_path, tasks, steps):
    root = _store(tmp_path, tasks, steps)
    _stamp_file(root).write_text("{not json")
    with pytest.raises(StoreProtocolError, match="unreadable"):
        populated_pairs(root, NAME)


def test_a_missing_store_still_reads_as_empty(tmp_path):
    """The create step's fresh-run path: no store means no cells yet."""
    assert populated_pairs(tmp_path / "nowhere", NAME) == set()
    assert populated_iterations(tmp_path / "nowhere", NAME) == {}


def test_store_stamp_is_none_for_an_unstamped_store(tmp_path, tasks, steps):
    root = _store(tmp_path, tasks, steps)
    _make_unstamped(root)
    assert store_stamp(root, NAME) is None


#                                                          append and replace
# =============================================================================


@pytest.mark.parametrize("protocol", [None, "some-older-protocol"])
def test_append_refuses_before_touching_the_store(
    tmp_path, tasks, steps, protocol
):
    runs_a = _write_raw_runs(tmp_path / "a", [(tasks[0], 20)], steps)
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 20)], steps)
    root = tmp_path / "db"
    _create_zarr(root, runs_a)
    if protocol is None:
        _make_unstamped(root)
    else:
        _restamp(root, protocol)
    store = layout.zarr_store_path(root, NAME)
    before = _snapshot(store)
    with pytest.raises(StoreProtocolError, match="Nothing was written"):
        _append_zarr(root, runs_b)
    assert _snapshot(store) == before
    assert not layout.append_marker_path(root, NAME).exists()
    assert not layout.rollback_path(root, NAME).exists()


@pytest.mark.parametrize("protocol", [None, "some-older-protocol"])
def test_replace_refuses_and_deletes_nothing(tmp_path, tasks, steps, protocol):
    root = _store(tmp_path, tasks, steps)
    if protocol is None:
        _make_unstamped(root)
    else:
        _restamp(root, protocol)
    store = layout.zarr_store_path(root, NAME)
    rawroot = layout.raw_root(root, NAME)
    before = _snapshot(store)
    raw_before = _snapshot(rawroot)
    runs = _write_raw_runs(tmp_path / "again", [(tasks[0], 20)], steps)
    with pytest.raises(StoreProtocolError, match="delete it by hand"):
        _dump(root, runs, mode="replace")
    assert _snapshot(store) == before
    assert _snapshot(rawroot) == raw_before


#                                                                      verify
# =============================================================================


def _protocol_findings(report):
    return [f for f in report.findings if f.check == "store.protocol"]


def test_verify_is_silent_on_a_matching_stamp(tmp_path, tasks, steps):
    root = _store(tmp_path, tasks, steps)
    report = verify_databank(root, NAME)
    assert _protocol_findings(report) == []
    assert report.ok


@pytest.mark.parametrize("protocol", [None, "some-older-protocol"])
def test_verify_audits_another_protocol_and_reports_it(
    tmp_path, tasks, steps, protocol
):
    root = _store(tmp_path, tasks, steps)
    if protocol is None:
        _make_unstamped(root)
    else:
        _restamp(root, protocol)
    report = verify_databank(root, NAME)
    (finding,) = _protocol_findings(report)
    assert finding.severity == "error"
    assert DATABANK_PROTOCOL in finding.message
    assert not report.ok
    # The rest of the audit still ran.
    assert report.n_pairs_populated == len(tasks) * len(steps)


#                                                                      facade
# =============================================================================


def test_the_facade_exports_the_stamp():
    assert databank.DATABANK_PROTOCOL == DATABANK_PROTOCOL
    assert databank.store_stamp is store_stamp
    assert databank.StoreProtocolError is StoreProtocolError
    assert issubclass(StoreProtocolError, RuntimeError)
    assert databank.check_run_settings is check_run_settings
    assert databank.RunSettingsError is RunSettingsError
    assert issubclass(RunSettingsError, StoreProtocolError)
    assert databank.check_f_star is check_f_star
    assert databank.FStarError is FStarError
    assert issubclass(FStarError, ValueError)


def test_the_gap_change_bumped_the_protocol():
    """Stores now hold f - f* (ADR 0027): a stored number means something
    else, so the stamp's own rule bumps the protocol."""
    assert DATABANK_PROTOCOL == "dataset-paper-2026-11.1"


#                                                     run settings (decision g)
# =============================================================================

SETTINGS = {
    "budget": {"evaluations_per_dimension": 1000, "max_evaluations": 100000},
    "sampler": {"name": "uniform", "low": 0.0, "high": 1.0},
    "seed": 42,
    "n_realizations": 25,
    "dimensions": [2, 10],
}


def _dump_with(root: Path, runs, *, mode: str, run_settings=None) -> Path:
    plan = _dump_template_runs(
        runs,
        root,
        NAME,
        mode=mode,
        easiest_quantile=0.75,
        taskset=None,
        budget_steps=BS,
        run_settings=run_settings,
    )
    return dump_databank_finalize(_dump_task_runs(runs, plan), plan)


def _settings_store(tmp_path, tasks, steps, settings=SETTINGS) -> Path:
    runs = _write_raw_runs(tmp_path / "a", [(tasks[0], 20)], steps)
    root = tmp_path / "db"
    _dump_with(root, runs, mode="error", run_settings=settings)
    return root


def test_the_stamp_records_the_run_settings(tmp_path, tasks, steps):
    root = _settings_store(tmp_path, tasks, steps)
    stamp = store_stamp(root, NAME)
    assert stamp["run_settings"] == SETTINGS
    assert stamp["protocol"] == DATABANK_PROTOCOL


def test_append_with_the_same_settings_works(tmp_path, tasks, steps):
    root = _settings_store(tmp_path, tasks, steps)
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 20)], steps)
    # Same mapping, other key order and a tuple for a list: equal as JSON.
    same = dict(reversed(list(json.loads(json.dumps(SETTINGS)).items())))
    same["dimensions"] = (2, 10)
    _dump_with(root, runs_b, mode="append", run_settings=same)
    assert len(populated_pairs(root, NAME)) == 2 * len(steps)
    assert store_stamp(root, NAME)["run_settings"] == SETTINGS


@pytest.mark.parametrize(
    "incoming",
    [{**SETTINGS, "seed": 145}, None],
    ids=["other-seed", "none"],
)
def test_append_refuses_other_settings_before_touching_the_store(
    tmp_path, tasks, steps, incoming
):
    root = _settings_store(tmp_path, tasks, steps)
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 20)], steps)
    store = layout.zarr_store_path(root, NAME)
    before = _snapshot(store)
    with pytest.raises(RunSettingsError, match="Nothing was written"):
        _dump_with(root, runs_b, mode="append", run_settings=incoming)
    assert _snapshot(store) == before
    assert not layout.append_marker_path(root, NAME).exists()


def test_append_with_settings_into_a_store_without_refuses(
    tmp_path, tasks, steps
):
    root = _store(tmp_path, tasks, steps)
    runs_b = _write_raw_runs(tmp_path / "b", [(tasks[1], 20)], steps)
    with pytest.raises(RunSettingsError, match="no run settings"):
        _dump_with(root, runs_b, mode="append", run_settings=SETTINGS)


def test_check_run_settings_for_the_create_step(tmp_path, tasks, steps):
    # No store yet: nothing to compare with.
    assert check_run_settings(tmp_path / "nowhere", NAME, SETTINGS) is None
    root = _settings_store(tmp_path, tasks, steps)
    assert check_run_settings(root, NAME, dict(SETTINGS)) is None
    with pytest.raises(RunSettingsError) as err:
        check_run_settings(root, NAME, {**SETTINGS, "n_realizations": 10})
    assert "n_realizations" in str(err.value)
    # Neither side has settings: as before.
    plain = _store(tmp_path / "plain", tasks, steps)
    assert check_run_settings(plain, NAME, None) is None


def test_check_run_settings_refuses_another_protocol_first(
    tmp_path, tasks, steps
):
    root = _settings_store(tmp_path, tasks, steps)
    _restamp(root, "some-older-protocol")
    with pytest.raises(StoreProtocolError, match="some-older-protocol"):
        check_run_settings(root, NAME, SETTINGS)


#                                                     f* per task (ADR 0027)
# =============================================================================


def test_check_f_star_for_the_create_step(tmp_path, tasks, steps):
    """A create step learns before any cell runs that a task's corrected
    minimum would be refused by the dump's append."""
    t0, t1 = tasks  # both stored with global_min 0.0
    # No store yet: nothing to compare with.
    assert (
        databank.check_f_star(tmp_path / "nowhere", NAME, {t0.hash: 1.0})
        is None
    )
    root = _store(tmp_path, tasks, steps)
    assert databank.check_f_star(root, NAME, {t0.hash: 0.0}) is None
    # A task the store does not hold passes, whatever its f*.
    assert databank.check_f_star(root, NAME, {"not-stored": 7.0}) is None
    with pytest.raises(databank.FStarError) as err:
        databank.check_f_star(
            root, NAME, {t0.hash: -2.5, t1.hash: float("nan")}
        )
    msg = str(err.value)
    assert f"{t0.hash}: stored 0.0, incoming -2.5" in msg
    assert f"{t1.hash}: stored 0.0, incoming nan" in msg
    assert "cannot change" in msg


def test_check_f_star_refuses_a_store_left_mid_append(tmp_path, tasks, steps):
    """A crashed append's half-grown store holds f* NaN for the tasks it
    added, so the check says the store is mid-append instead of comparing."""
    root = _store(tmp_path, tasks, steps)
    layout.append_marker_path(root, NAME).write_text("{}")
    with pytest.raises(RuntimeError, match="mid-append"):
        databank.check_f_star(root, NAME, {tasks[0].hash: 0.0})


def test_check_f_star_refuses_another_protocol_first(tmp_path, tasks, steps):
    root = _store(tmp_path, tasks, steps)
    _restamp(root, "some-older-protocol")
    with pytest.raises(StoreProtocolError, match="some-older-protocol"):
        databank.check_f_star(root, NAME, {tasks[0].hash: 0.0})


def test_run_settings_must_be_a_json_mapping(tmp_path, tasks, steps):
    runs = _write_raw_runs(tmp_path / "a", [(tasks[0], 20)], steps)
    for bad in ({"sampler": object()}, ["seed", 1], [("seed", 1)]):
        with pytest.raises(TypeError, match="run_settings"):
            _dump_with(tmp_path / "db", runs, mode="error", run_settings=bad)
    assert not layout.zarr_store_path(tmp_path / "db", NAME).exists()
