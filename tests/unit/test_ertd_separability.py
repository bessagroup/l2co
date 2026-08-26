"""Exactness invariants of the ERTD bootstrap.

Two properties that consumers rely on to keep the ERTD tractable on a
full databank selection, where the unbatched, whole-selection form
allocates tens of GB to produce a sub-GB profile:

seed-batch invariance
    :func:`l2co._src.performance.ertd` evaluates the bootstrap in batches
    of ``seed_batch`` replicates. Batching must not perturb the result --
    only when the resampled arrays are allocated.

task separability
    :meth:`DataLoader.ertd_profile` never reduces over ``taskID``, and the
    bootstrap draws depend on the *number* of realizations rather than on
    which tasks are selected. So the profile rows of a task subset equal
    the corresponding rows of the full profile, which is what lets a
    caller shard the computation over tasks and concatenate the partials
    (``l2co_experiments``' parallel ``ertd_report``). Every report panel is
    a mean over some task subset of that array, so overlapping tag groups
    cost nothing extra.

Both are asserted **bit-exact**: these are reorderings of the same
arithmetic, not approximations, so any drift signals a real change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from l2co import DataLoader
from l2co._src.performance import ERTD_N_SEEDS, ERTD_SEED_BATCH

#                                                                      Fixtures
# =============================================================================

_SCHEDULES = ["randomsearch", "good_opt", "bad_opt"]
_TASK_IDS = ["task_0", "task_1", "task_2", "task_3", "task_4", "task_5"]
_N_REALIZATIONS = 4
_TARGETS = np.array([1e-8, 1e-4, 1.0, 10.0])
_DIM_MAP = {
    "task_0": 2,
    "task_1": 5,
    "task_2": 2,
    "task_3": 10,
    "task_4": 5,
    "task_5": 10,
}


def _loader(task_ids: list[str]) -> DataLoader:
    """Build a performance-only loader over the given tasks.

    Values are a deterministic function of ``(schedule, task, realization,
    target)`` only -- never of the selection -- so a subset loader carries
    exactly the rows the full loader does. Some ``(task, schedule)`` cells
    never succeed and one is all-NaN, exercising the NaN-skipping
    ``mean(SEED)`` and the ``valid_files`` mask.
    """
    problem_index = pd.MultiIndex.from_product(
        [list(range(_N_REALIZATIONS)), task_ids],
        names=["realization", "taskID"],
    )
    n_problems = len(problem_index)
    n_s, n_t = len(_SCHEDULES), _TARGETS.size

    success = np.zeros((n_s, 1, n_t, n_problems), dtype=float)
    cost = np.zeros((2, n_s, 1, n_t, n_problems), dtype=float)

    for p, (real, tid) in enumerate(problem_index):
        t_idx = _TASK_IDS.index(tid)
        for j, sched in enumerate(_SCHEDULES):
            for k in range(n_t):
                # Deterministic, selection-independent pseudo-values.
                h = (t_idx * 7 + j * 13 + real * 3 + k * 5) % 11
                cost[:, j, 0, k, p] = 100.0 + h * 10.0 + k
                if sched == "bad_opt":
                    success[j, 0, k, p] = 0.0  # never solves anything
                elif sched == "good_opt":
                    success[j, 0, k, p] = float(h % 3 != 0)
                else:
                    success[j, 0, k, p] = float(k < 2)

    # One all-NaN cell: good_opt on task_0 crashed outright.
    crashed = [
        p for p, (_, tid) in enumerate(problem_index) if tid == "task_0"
    ]
    success[1, 0, :, crashed] = np.nan

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
            "schedule": _SCHEDULES,
            "output_dim": ["y"],
            "target": _TARGETS,
            "metric": ["fevals", "iterations"],
            "problem": problem_index,
        },
    )

    features = ["fn_name", "dimensionality"]
    labels = np.empty((n_problems, len(features)), dtype=object)
    for p, (_real, tid) in enumerate(problem_index):
        labels[p, 0] = "sphere" if _TASK_IDS.index(tid) % 2 == 0 else "rosen"
        labels[p, 1] = _DIM_MAP[tid]

    task_labels = xr.DataArray(
        labels,
        dims=["problem", "feature"],
        coords={"problem": problem_index, "feature": features},
    )
    return DataLoader(data=data, task_labels=task_labels)


def _profile(
    task_ids: list[str], include_best: bool = False, **kwargs
) -> xr.DataArray:
    """Metric-selected ERTD profile over ``task_ids``, sorted by problem."""
    prof = _loader(task_ids).ertd_profile(include_best=include_best, **kwargs)
    return prof.sel({"metric": "fevals"}).sortby(["problem"])


def _assert_bit_identical(a: xr.DataArray, b: xr.DataArray) -> None:
    """Same coords, same NaN pattern, bit-identical finite values."""
    b = b.transpose(*a.dims)
    assert list(a["problem"].values) == list(b["problem"].values)
    assert list(a["schedule"].values) == list(b["schedule"].values)
    av, bv = a.values, b.values
    assert av.shape == bv.shape
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    assert np.array_equal(av, bv, equal_nan=True)


#                                                        Seed-batch invariance
# =============================================================================


@pytest.mark.parametrize("seed_batch", [1, 3, 5, 7, ERTD_N_SEEDS])
def test_seed_batching_is_exact(seed_batch: int) -> None:
    """Any ``seed_batch`` reproduces the unbatched bootstrap exactly.

    ``ERTD_N_SEEDS`` is the unbatched reference; 7 does not divide 50, so
    the ragged final batch is covered too.
    """
    reference = _profile(_TASK_IDS, seed_batch=ERTD_N_SEEDS)
    _assert_bit_identical(
        reference, _profile(_TASK_IDS, seed_batch=seed_batch)
    )


def test_default_seed_batch_divides_the_seed_axis() -> None:
    """The shipped default keeps every batch the same size."""
    assert ERTD_N_SEEDS % ERTD_SEED_BATCH == 0


def test_seed_batch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="seed_batch must be >= 1"):
        _profile(_TASK_IDS, seed_batch=0)


#                                                            Task separability
# =============================================================================


def test_task_subset_matches_full_profile() -> None:
    """A subset loader's profile rows equal the full profile's rows."""
    full = _profile(_TASK_IDS)
    subset_ids = ["task_1", "task_4"]
    subset = _profile(subset_ids)

    keep = np.flatnonzero(full["taskID"].isin(subset_ids).values)
    _assert_bit_identical(
        full.isel({"problem": keep}).sortby(["problem"]), subset
    )


@pytest.mark.parametrize("include_best", [False, True])
@pytest.mark.parametrize("shard_size", [1, 2, 4])
def test_disjoint_shards_reassemble_the_full_profile(
    shard_size: int, include_best: bool
) -> None:
    """Concatenating per-shard profiles reproduces the whole computation.

    This is the invariant the parallel ``ertd_report`` rests on: shard the
    tasks, compute each shard's profile in its own process, concatenate
    along ``problem``. ``shard_size=4`` does not divide the six tasks, so
    the ragged last shard is covered. ``include_best`` adds the ``vbs``
    envelope, whose per-problem ``idxmax(SCHEDULE)`` must also stay
    shard-local.
    """
    full = _profile(_TASK_IDS, include_best=include_best)
    shards = [
        _TASK_IDS[i : i + shard_size]
        for i in range(0, len(_TASK_IDS), shard_size)
    ]
    assert sorted(t for s in shards for t in s) == sorted(_TASK_IDS)

    reassembled = xr.concat(
        [_profile(shard, include_best=include_best) for shard in shards],
        dim="problem",
    ).sortby(["problem"])
    _assert_bit_identical(full, reassembled)


def test_panel_means_are_shard_invariant() -> None:
    """Tag-group panels agree whether or not the profile was sharded.

    A panel is a mean over the tasks carrying one tag value; groups
    overlap across features, and each is a cheap subset-mean of the same
    array either way.
    """
    full = _profile(_TASK_IDS)
    reassembled = xr.concat(
        [_profile(_TASK_IDS[i : i + 2]) for i in range(0, 6, 2)],
        dim="problem",
    ).sortby(["problem"])

    for group in ([t for t in _TASK_IDS if _DIM_MAP[t] == 5], _TASK_IDS[:1]):
        mask = np.flatnonzero(full["taskID"].isin(group).values)
        a = full.isel({"problem": mask}).mean("problem")
        b = reassembled.isel({"problem": mask}).mean("problem")
        assert np.array_equal(
            a.values, b.transpose(*a.dims).values, equal_nan=True
        )
