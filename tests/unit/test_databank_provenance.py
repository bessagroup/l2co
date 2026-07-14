"""Unit tests for policy-artifact provenance
(:mod:`l2co._src.databank.provenance` + the writer/reader surfaces).

Mirrors the ``test_databank.py`` battery: the streaming creation /
append cores are driven with hand-built ``runs`` (no f3dasm), pointing
the databank at ``tmp_path``. Covers the manifest lifecycle (write /
load / verify / bump), the record helpers (composite hashing, task
union, path-free identity hash), the store schema additions
(``provenance`` / ``in_training``), the alias collision guard, the
admissibility backstop, artifact archiving, and the reader helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from l2co._src.databank import layout
from l2co._src.databank.provenance import (
    MissingManifestError,
    WeightsHashMismatchError,
    bump_manifest,
    dumps_provenance,
    load_manifest,
    manifest_path,
    mint_run_id,
    parse_provenance,
    policy_identity_hash,
    provenance_record,
    record_task_hashes,
    record_weights_hash,
    weights_hash,
    write_manifest,
)
from l2co._src.databank.reader import (
    held_out_task_ids,
    load_databank,
    load_provenance,
)
from l2co._src.databank.verify import verify_databank
from l2co._src.databank.writer import (
    _archive_policy_artifacts,
    _dump_append,
    _dump_create,
)
from l2co.naming import (
    BUDGET,
    FEVALS,
    IN_TRAINING,
    ITERATIONS,
    OUTPUT_MIN,
    PROVENANCE,
    REALIZATION,
    SCHEDULE,
    TASK_ID,
    UPDATE_STEP,
)
from l2co.optimization import OptimizationStep

pytestmark = pytest.mark.filterwarnings("ignore")

N_REAL = 3
BS = 16


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
    """A raw per-run loss history: dims (realization, budget, output_dim)."""
    budget = np.arange(n_budget)
    base = scale * np.exp(-budget / 2.0)[None, :]
    offset = (1 + np.arange(N_REAL))[:, None] * 1e-3
    output = (base + offset)[:, :, None]

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
def tasks() -> list[FakeTask]:
    return [
        FakeTask(hash="a1b2c3", fn_name="sphere", dimensionality=10),
        FakeTask(hash="9f8e7d", fn_name="rastrigin", dimensionality=10),
    ]


def _make_checkpoint(dirpath: Path, stem: str, payload: bytes) -> Path:
    """Write a fake ``.eqx`` checkpoint and return its stem path."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{stem}.eqx").write_bytes(payload)
    return dirpath / stem


def _make_artifact(
    dirpath: Path,
    stem: str,
    *,
    payload: bytes = b"weights-bytes",
    task_hashes: tuple[str, ...] = ("a1b2c3",),
    run_id: str | None = None,
) -> tuple[Path, dict]:
    """Fake checkpoint + manifest; returns ``(stem_path, manifest)``."""
    ckpt = _make_checkpoint(dirpath, stem, payload)
    write_manifest(
        ckpt,
        run_id=run_id or mint_run_id("test_experiment"),
        experiment="test_experiment",
        training_step=500,
        config={"model": {"hidden": 8}, "discount": 0.99},
        git={"l2co": {"sha": "abc", "dirty": False}},
        training_task_hashes=list(task_hashes),
        training_data={"tasks_choice": "bbob"},
    )
    return ckpt, load_manifest(ckpt)


def _make_record(
    dirpath: Path,
    stem: str = "policy",
    *,
    payload: bytes = b"weights-bytes",
    task_hashes: tuple[str, ...] = ("a1b2c3",),
) -> dict:
    """A full single-component provenance record with artifacts listed."""
    ckpt, manifest = _make_artifact(
        dirpath, stem, payload=payload, task_hashes=task_hashes
    )
    return provenance_record(
        training=manifest,
        evaluation={
            "git": {"l2co": {"sha": "abc", "dirty": False}},
            "evaluated_at": "2026-07-13T00:00:00+00:00",
            "artifacts": {
                "model": {
                    "path": str(ckpt),
                    "weights_hash": manifest["weights_hash"],
                }
            },
        },
    )


def _policy_step(model_stem: str = "policy") -> OptimizationStep:
    return OptimizationStep(
        optimizer="rl2co",
        hyperparameters={
            "model": f"/some/dir/{model_stem}",
            "n_iterations_per_action": [5, 25],
        },
    )


def _write_runs(
    rundir: Path,
    tasks: list[FakeTask],
    steps_with_prov: list[tuple[OptimizationStep, dict | None]],
) -> list:
    """Write raw loss netCDFs (attaching provenance attrs) -> ``runs``."""
    rundir.mkdir(parents=True, exist_ok=True)
    runs = []
    for task in tasks:
        step_paths = {}
        for s, record in steps_with_prov:
            p = rundir / f"{task.hash}_{s.hash}.nc"
            ds = make_loss_ds(8, scale=10.0)
            if record is not None:
                ds.attrs[PROVENANCE] = dumps_provenance(record)
            ds.to_netcdf(p)
            step_paths[s] = p
        runs.append((task, step_paths))
    return runs


def _create(root: Path, runs) -> tuple[Path, dict[str, str]]:
    store = layout.zarr_store_path(root, "bbob")
    rawroot = layout.raw_root(root, "bbob")
    rawroot.mkdir(parents=True, exist_ok=True)
    _, provs = _dump_create(
        store, rawroot, runs, easiest_quantile=0.75, budget_steps=BS
    )
    return store, provs


#                                                            Manifest lifecycle
# =============================================================================


def test_manifest_roundtrip(tmp_path):
    ckpt, manifest = _make_artifact(tmp_path, "model_a")
    assert manifest["run_id"].startswith("test_experiment-")
    assert manifest["training_step"] == 500
    assert manifest["config"]["discount"] == 0.99
    assert manifest["training_data"]["task_hashes"] == ["a1b2c3"]
    assert manifest["weights_hash"] == weights_hash(ckpt)
    # The manifest sits next to the checkpoint, keyed by stem.
    assert manifest_path(ckpt) == tmp_path / "model_a.manifest.json"
    assert manifest_path(tmp_path / "model_a.eqx") == manifest_path(ckpt)


def test_missing_manifest_raises(tmp_path):
    ckpt = _make_checkpoint(tmp_path, "orphan", b"bytes")
    with pytest.raises(MissingManifestError, match="not admissible"):
        load_manifest(ckpt)


def test_weights_mismatch_raises(tmp_path):
    ckpt, _ = _make_artifact(tmp_path, "model_a")
    (tmp_path / "model_a.eqx").write_bytes(b"tampered")
    with pytest.raises(WeightsHashMismatchError):
        load_manifest(ckpt)


def test_bump_manifest_keeps_run_id(tmp_path):
    ckpt, before = _make_artifact(tmp_path, "current_model")
    # In-place overwrite (the multitask trainer's save pattern) ...
    (tmp_path / "current_model.eqx").write_bytes(b"new-weights")
    bump_manifest(ckpt, training_step=501)
    after = load_manifest(ckpt)
    assert after["run_id"] == before["run_id"]
    assert after["training_step"] == 501
    assert after["weights_hash"] != before["weights_hash"]
    assert after["weights_hash"] == weights_hash(ckpt)


def test_weights_hash_covers_sidecar_files(tmp_path):
    ckpt = _make_checkpoint(tmp_path, "stage0", b"eqx-header")
    h_single = weights_hash(ckpt)
    (tmp_path / "stage0.joblib").write_bytes(b"sklearn-bytes")
    h_multi = weights_hash(ckpt)
    assert h_single != h_multi
    (tmp_path / "stage0.joblib").write_bytes(b"other-bytes")
    assert weights_hash(ckpt) != h_multi


def test_mint_run_id_unique_and_prefixed():
    a = mint_run_id("exp")
    b = mint_run_id("exp")
    assert a.startswith("exp-") and b.startswith("exp-")
    assert a != b


#                                                                Record helpers
# =============================================================================


def test_composite_record_hash_and_task_union(tmp_path):
    _, m0 = _make_artifact(tmp_path / "s0", "stage0", task_hashes=("t1", "t2"))
    _, m1 = _make_artifact(
        tmp_path / "s1", "stage1", payload=b"other", task_hashes=("t2", "t3")
    )
    record = provenance_record(
        training={"components": {"stage0": m0, "stage1": m1}},
        evaluation={"git": {}, "evaluated_at": "now", "artifacts": {}},
    )
    # Composite hash differs from either component and is order-stable.
    combined = record_weights_hash(record)
    assert combined not in (m0["weights_hash"], m1["weights_hash"])
    # In-training = union over components (seen by any component).
    assert record_task_hashes(record) == {"t1", "t2", "t3"}


def test_identity_hash_ignores_paths(tmp_path):
    record = _make_record(tmp_path)
    step_a = _policy_step("policy")
    step_b = OptimizationStep(
        optimizer="rl2co",
        hyperparameters={
            "model": "/moved/elsewhere/policy",  # path changed only
            "n_iterations_per_action": [5, 25],
        },
    )
    assert policy_identity_hash(record, step_a) == policy_identity_hash(
        record, step_b
    )
    # Different eval settings -> different identity.
    step_c = OptimizationStep(
        optimizer="rl2co",
        hyperparameters={
            "model": "/some/dir/policy",
            "n_iterations_per_action": [10, 50],
        },
    )
    assert policy_identity_hash(record, step_a) != policy_identity_hash(
        record, step_c
    )
    # Different weights -> different identity.
    other = _make_record(tmp_path / "other", payload=b"other-weights")
    assert policy_identity_hash(record, step_a) != policy_identity_hash(
        other, step_a
    )


def test_parse_provenance_empty_and_nan():
    assert parse_provenance("") is None
    assert parse_provenance(None) is None
    assert parse_provenance(float("nan")) is None
    assert parse_provenance('{"training": {}}') == {"training": {}}


#                                                             Store integration
# =============================================================================


def test_create_stores_provenance_and_in_training(tmp_path, tasks):
    # Policy trained on tasks[0] only -> tasks[1] is held out.
    record = _make_record(tmp_path / "ckpt", task_hashes=(tasks[0].hash,))
    static = OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01})
    policy = _policy_step()
    runs = _write_runs(
        tmp_path / "runs", tasks, [(static, None), (policy, record)]
    )
    store, provs = _create(tmp_path, runs)

    ds = xr.open_zarr(store, consolidated=True)
    # Provenance: JSON for the policy schedule, empty for the static one.
    prov = {
        str(s): str(v)
        for s, v in zip(ds[SCHEDULE].values, ds[PROVENANCE].values)
    }
    assert prov[static.name] == ""
    stored = json.loads(prov[policy.name])
    assert record_weights_hash(stored) == record_weights_hash(record)
    # in_training: NaN for static, 1/0 per task for the policy.
    it = ds[IN_TRAINING].load()
    assert it.sel({SCHEDULE: static.name}).isnull().all()
    assert (
        float(it.sel({SCHEDULE: policy.name, TASK_ID: tasks[0].hash})) == 1.0
    )
    assert (
        float(it.sel({SCHEDULE: policy.name, TASK_ID: tasks[1].hash})) == 0.0
    )
    # The dump reports exactly the policy schedules for archiving.
    assert set(provs) == {policy.name}
    ds.close()


def test_policy_raw_filename_uses_identity_hash(tmp_path, tasks):
    record = _make_record(tmp_path / "ckpt")
    policy = _policy_step()
    runs = _write_runs(tmp_path / "runs", tasks, [(policy, record)])
    _create(tmp_path, runs)

    rawroot = layout.raw_root(tmp_path, "bbob")
    id_hash = policy_identity_hash(record, policy)
    assert layout.loss_raw_path(rawroot, tasks[0].hash, id_hash).exists()
    # The recipe hash is path-derived for rl2co steps -- must NOT be used.
    assert not layout.loss_raw_path(
        rawroot, tasks[0].hash, policy.hash
    ).exists()


def test_policy_without_provenance_rejected(tmp_path, tasks):
    policy = _policy_step()
    runs = _write_runs(tmp_path / "runs", tasks, [(policy, None)])
    with pytest.raises(ValueError, match="not admissible"):
        _create(tmp_path, runs)


def test_same_artifact_different_eval_timestamps_ok(tmp_path, tasks):
    # Two records of the SAME artifact differing in evaluation section
    # (per-pair evaluated_at) must not trip the alias guard.
    record = _make_record(tmp_path / "ckpt")
    other = json.loads(dumps_provenance(record))
    other["evaluation"]["evaluated_at"] = "2026-07-14T00:00:00+00:00"
    policy = _policy_step()
    rundir = tmp_path / "runs"
    rundir.mkdir()
    runs = []
    for task, rec in zip(tasks, (record, other)):
        p = rundir / f"{task.hash}.nc"
        ds = make_loss_ds(8, scale=10.0)
        ds.attrs[PROVENANCE] = dumps_provenance(rec)
        ds.to_netcdf(p)
        runs.append((task, {policy: p}))
    store, provs = _create(tmp_path, runs)
    assert set(provs) == {policy.name}


def test_alias_collision_on_create_rejected(tmp_path, tasks):
    # Same schedule name, two different artifacts -> hard error.
    rec_a = _make_record(tmp_path / "a", payload=b"weights-a")
    rec_b = _make_record(tmp_path / "b", payload=b"weights-b")
    policy = _policy_step()
    rundir = tmp_path / "runs"
    rundir.mkdir()
    runs = []
    for task, rec in zip(tasks, (rec_a, rec_b)):
        p = rundir / f"{task.hash}.nc"
        ds = make_loss_ds(8, scale=10.0)
        ds.attrs[PROVENANCE] = dumps_provenance(rec)
        ds.to_netcdf(p)
        runs.append((task, {policy: p}))
    with pytest.raises(ValueError, match="one alias must always mean"):
        _create(tmp_path, runs)


def test_alias_collision_on_append_rejected(tmp_path, tasks):
    record = _make_record(tmp_path / "ckpt", task_hashes=(tasks[0].hash,))
    policy = _policy_step()
    runs = _write_runs(tmp_path / "runs", tasks[:1], [(policy, record)])
    store, _ = _create(tmp_path, runs)

    # A retrained checkpoint under the SAME alias must be refused.
    other = _make_record(tmp_path / "ckpt2", payload=b"retrained")
    runs2 = _write_runs(tmp_path / "runs2", tasks[1:], [(policy, other)])
    rawroot = layout.raw_root(tmp_path, "bbob")
    with pytest.raises(ValueError, match="one alias must always mean"):
        _dump_append(store, rawroot, runs2, easiest_quantile=0.75)


def test_append_new_task_keeps_provenance(tmp_path, tasks):
    record = _make_record(tmp_path / "ckpt", task_hashes=(tasks[0].hash,))
    static = OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01})
    policy = _policy_step()
    runs = _write_runs(
        tmp_path / "runs", tasks[:1], [(static, None), (policy, record)]
    )
    store, _ = _create(tmp_path, runs)

    # Append the second task for the same artifact (same alias) ...
    runs2 = _write_runs(
        tmp_path / "runs2", tasks[1:], [(static, None), (policy, record)]
    )
    rawroot = layout.raw_root(tmp_path, "bbob")
    _, provs = _dump_append(store, rawroot, runs2, easiest_quantile=0.75)
    assert set(provs) == {policy.name}

    ds = xr.open_zarr(store, consolidated=True)
    it = ds[IN_TRAINING].load()
    # ... and the mask is recomputed over the union of tasks.
    assert (
        float(it.sel({SCHEDULE: policy.name, TASK_ID: tasks[0].hash})) == 1.0
    )
    assert (
        float(it.sel({SCHEDULE: policy.name, TASK_ID: tasks[1].hash})) == 0.0
    )
    stored = parse_provenance(
        str(ds[PROVENANCE].sel({SCHEDULE: policy.name}).values)
    )
    assert record_weights_hash(stored) == record_weights_hash(record)
    ds.close()


#                                                            Artifact archiving
# =============================================================================


def test_archive_copies_checkpoint_manifest_and_record(tmp_path):
    record = _make_record(tmp_path / "ckpt")
    _archive_policy_artifacts(
        tmp_path, "bbob", {"rl2co_policy": dumps_provenance(record)}
    )
    dest = layout.policy_artifacts_root(tmp_path, "bbob") / "rl2co_policy"
    assert (dest / "policy.eqx").exists()
    assert (dest / "policy.manifest.json").exists()
    stored = json.loads((dest / "provenance.json").read_text())
    assert record_weights_hash(stored) == record_weights_hash(record)


def test_archive_missing_checkpoint_raises(tmp_path):
    record = _make_record(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "policy.eqx").unlink()  # scratch purge
    with pytest.raises(FileNotFoundError, match="purged"):
        _archive_policy_artifacts(
            tmp_path, "bbob", {"rl2co_policy": dumps_provenance(record)}
        )


def test_archive_tampered_checkpoint_raises(tmp_path):
    record = _make_record(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "policy.eqx").write_bytes(b"overwritten")
    with pytest.raises(ValueError, match="refusing to archive"):
        _archive_policy_artifacts(
            tmp_path, "bbob", {"rl2co_policy": dumps_provenance(record)}
        )


def test_archive_without_artifacts_map_raises(tmp_path):
    record = _make_record(tmp_path / "ckpt")
    record["evaluation"]["artifacts"] = {}
    with pytest.raises(ValueError, match="cannot be archived"):
        _archive_policy_artifacts(
            tmp_path, "bbob", {"rl2co_policy": dumps_provenance(record)}
        )


#                                                                       Readers
# =============================================================================


def _mixed_store(tmp_path, tasks):
    record = _make_record(tmp_path / "ckpt", task_hashes=(tasks[0].hash,))
    static = OptimizationStep(optimizer="adam", hyperparameters={"lr": 0.01})
    policy = _policy_step()
    runs = _write_runs(
        tmp_path / "runs", tasks, [(static, None), (policy, record)]
    )
    store, _ = _create(tmp_path, runs)
    return store, static, policy, record


def test_load_provenance_maps_policy_schedules_only(tmp_path, tasks):
    _, static, policy, record = _mixed_store(tmp_path, tasks)
    provs = load_provenance(tmp_path, "bbob")
    assert set(provs) == {policy.name}
    assert record_weights_hash(provs[policy.name]) == record_weights_hash(
        record
    )


def test_held_out_task_ids(tmp_path, tasks):
    _, static, policy, _ = _mixed_store(tmp_path, tasks)
    assert held_out_task_ids(tmp_path, "bbob", policy.name) == [tasks[1].hash]
    with pytest.raises(ValueError, match="no training set"):
        held_out_task_ids(tmp_path, "bbob", static.name)
    with pytest.raises(ValueError, match="not present"):
        held_out_task_ids(tmp_path, "bbob", "nope")


def test_load_databank_drops_provenance_vars(tmp_path, tasks):
    _, static, policy, _ = _mixed_store(tmp_path, tasks)
    d = load_databank("bbob", [static.name, policy.name], root=tmp_path)
    assert PROVENANCE not in d.data
    assert IN_TRAINING not in d.data


def test_verify_provenance_store_ok(tmp_path, tasks):
    _mixed_store(tmp_path, tasks)
    report = verify_databank(tmp_path, "bbob")
    assert report.ok, report.summary()
