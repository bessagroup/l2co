"""The ERTD's restart bootstrap can be restricted to a subset of runs.

The bootstrap in :func:`l2co._src.performance.ertd` resamples the
``realization`` axis, so a single call over the whole run pool returns one
number per (task, schedule) with no attached uncertainty. Restricting the
pool to disjoint halves produces two independent estimates of the same
profile, and their spread is the estimate's own sampling error.

Three properties matter to callers:

pass-through
    Naming every run must reproduce the default exactly. Otherwise the
    parameter would silently change results wherever it is threaded but
    left unset.

selectivity
    The subset must actually reach the bootstrap. A half drawn from runs
    that never succeed and a half drawn from runs that always do cannot
    agree, so disagreement is the evidence that the pool was honoured.

separability is preserved
    Task sharding still holds under a realization subset, since the two
    axes are independent. The parallel report path relies on this.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from l2co._src.performance import ertd

from .test_ertd_separability import _DIM_MAP, _TASK_IDS, _loader

#                                                                      Fixtures
# =============================================================================


def _profile(realizations=None, task_ids=None) -> xr.DataArray:
    """ERTD profile of the shared synthetic loader."""
    return _loader(task_ids or _TASK_IDS).ertd_profile(
        realizations=realizations
    )


#                                                                         Tests
# =============================================================================


def test_naming_every_realization_matches_the_default() -> None:
    """The full index set is a no-op, bit-exact."""
    d = _loader(_TASK_IDS)
    full = np.arange(d.n_realization)
    xr.testing.assert_identical(
        _profile(), _profile(realizations=full)
    )


def test_permuting_the_index_set_is_a_no_op() -> None:
    """Selection is a set, not an order: the pool is what matters."""
    d = _loader(_TASK_IDS)
    shuffled = np.arange(d.n_realization)[::-1]
    xr.testing.assert_identical(
        _profile(), _profile(realizations=shuffled)
    )


def test_disjoint_halves_are_independent_estimates() -> None:
    """Two halves cover the axis and need not agree.

    Agreement everywhere would mean the subset never reached the
    bootstrap, which is the failure this parameter exists to avoid.
    """
    d = _loader(_TASK_IDS)
    n = d.n_realization
    lo, hi = np.arange(0, n, 2), np.arange(1, n, 2)
    assert sorted([*lo, *hi]) == list(range(n))

    first, second = _profile(realizations=lo), _profile(realizations=hi)
    assert first.shape == second.shape
    assert not np.allclose(
        first.values, second.values, equal_nan=True
    ), "halves are identical -- the subset never reached the bootstrap"


def test_a_single_realization_is_admissible() -> None:
    """The smallest usable pool is one run, not a special case."""
    profile = _profile(realizations=[0])
    assert profile.notnull().any()


def test_empty_selection_is_rejected() -> None:
    """An empty pool is a caller error, not an all-NaN profile."""
    d = _loader(_TASK_IDS)
    dims = d.task_labels.sel({"feature": "dimensionality"}, drop=True).astype(
        int
    )
    with pytest.raises(ValueError, match="at least one run"):
        ertd(performance=d.data, dims=dims, realizations=[])


@pytest.mark.parametrize("shards", [([_TASK_IDS[:2], _TASK_IDS[2:]])])
def test_task_sharding_still_holds_within_a_half(
    shards: list[list[str]],
) -> None:
    """Realization subsetting and task sharding are independent.

    The parallel report computes one shard per array job; restricting the
    run pool must not couple the shards to each other.
    """
    half = np.arange(0, 4, 2)
    full = _profile(realizations=half)
    parts = [_profile(realizations=half, task_ids=s) for s in shards]
    joined = xr.concat(parts, dim="problem").sortby("taskID")
    xr.testing.assert_allclose(full.sortby("taskID"), joined)


def test_dim_map_covers_the_fixture_tasks() -> None:
    """Guard the imported fixture constants against drift."""
    assert set(_DIM_MAP) == set(_TASK_IDS)
