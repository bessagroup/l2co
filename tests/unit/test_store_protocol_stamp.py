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
    StoreProtocolError,
    store_stamp,
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
