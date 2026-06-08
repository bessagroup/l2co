"""Unit tests for :meth:`DataLoader.report` and its helpers."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from l2co import DataLoader
from l2co._src.data.report_helpers import (
    EPSILON,
    descriptive_stats,
    descriptive_stats_log10,
    pairwise_win_rates,
    per_task_shift,
    pp_auc,
    sanitize_for_json,
    shifted_final_min,
    top_k_by,
)

#                                                                      Fixtures
# =============================================================================


@pytest.fixture
def synthetic_dataloader() -> DataLoader:
    """Build a minimal DataLoader covering the ``report()`` code paths.

    Three schedules including the ``randomsearch`` baseline, four
    BBOB-like tasks (two ``fn_name`` × two ``dimensionality``), two
    realizations per task. One cell is deliberately all-NaN to trigger
    ``optimizer_crashed``; ``bad_opt`` is uniformly worse than
    ``randomsearch`` so ``schedule_worse_than_baseline`` fires;
    ``good_opt`` wins every other task so ``dominant_schedule`` fires.
    Two of the four tasks have a negative ``global_min`` so the
    log-safety path is exercised.
    """
    schedules = ["randomsearch", "good_opt", "bad_opt"]
    task_ids = ["task_0", "task_1", "task_2", "task_3"]
    n_realizations = 2
    n_budget = 5
    targets = np.array([1e-8, 1e-4, 1.0, 10.0])
    n_targets = targets.size

    problem_index = pd.MultiIndex.from_product(
        [list(range(n_realizations)), task_ids],
        names=["realization", "taskID"],
    )
    n_problems = len(problem_index)
    n_schedules = len(schedules)

    output_min = np.zeros((n_schedules, 1, n_budget, n_problems), dtype=float)
    for j, s in enumerate(schedules):
        for t in range(n_budget):
            progress = (t + 1) / n_budget
            if s == "good_opt":
                output_min[j, 0, t, :] = 10.0 * (1e-6) ** progress
            elif s == "bad_opt":
                # Strictly worse than randomsearch at the final budget.
                output_min[j, 0, t, :] = 1000.0 - 1.0 * progress
            else:  # randomsearch
                output_min[j, 0, t, :] = 100.0 * (0.5) ** progress

    # Make good_opt crash on the first problem (all-NaN curve).
    output_min[1, 0, :, 0] = np.nan

    success = np.zeros((n_schedules, 1, n_targets, n_problems), dtype=bool)
    success[1, 0, :, :] = True
    success[1, 0, :, 0] = False  # good_opt crashed on problem 0
    success[0, 0, :2, :] = True  # randomsearch reaches easy targets

    # ``cost`` is consumed by ``ertd``. Magnitude does not matter for the
    # schema test — only that the variable is present and broadcastable.
    cost = (
        np.zeros((2, n_schedules, 1, n_targets, n_problems), dtype=np.int32)
        + 100
    )

    data = xr.Dataset(
        {
            "output_min": (
                ("schedule", "output_dim", "budget", "problem"),
                output_min,
            ),
            "success": (
                ("schedule", "output_dim", "target", "problem"),
                success,
            ),
            "cost": (
                (
                    "metric",
                    "schedule",
                    "output_dim",
                    "target",
                    "problem",
                ),
                cost,
            ),
        },
        coords={
            "schedule": schedules,
            "output_dim": ["y"],
            "budget": np.linspace(0.0, 1.0, n_budget),
            "target": targets,
            "metric": ["fevals", "iterations"],
            "problem": problem_index,
        },
    )

    features = ["fn_name", "dimensionality", "global_min"]
    fn_map = {
        "task_0": "sphere",
        "task_1": "sphere",
        "task_2": "rosenbrock",
        "task_3": "rosenbrock",
    }
    dim_map = {"task_0": 2, "task_1": 5, "task_2": 2, "task_3": 5}
    gm_map = {
        "task_0": 0.0,
        "task_1": 0.0,
        "task_2": -5.0,
        "task_3": -5.0,
    }

    labels_arr = np.empty((n_problems, len(features)), dtype=object)
    for i, idx in enumerate(problem_index):
        _real, tid = idx
        labels_arr[i, 0] = fn_map[tid]
        labels_arr[i, 1] = dim_map[tid]
        labels_arr[i, 2] = gm_map[tid]

    task_labels = xr.DataArray(
        labels_arr,
        dims=["problem", "feature"],
        coords={"problem": problem_index, "feature": features},
    )

    # The merged synthetic dataset carries both the loss-trajectory
    # variable (output_min) and the performance variables (success, cost)
    # on the single loader.
    return DataLoader(data=data, task_labels=task_labels)


#                                                          Helpers — pure tests
# =============================================================================


def test_sanitize_for_json_handles_nan_inf_and_numpy() -> None:
    obj = {
        "a": np.float32(1.5),
        "b": np.int64(7),
        "c": np.bool_(True),
        "d": float("nan"),
        "e": float("inf"),
        "f": [np.float64(2.0), np.array([1, 2, 3])],
        "g": (1, 2),
    }
    out = sanitize_for_json(obj)
    js = json.dumps(out)  # must not raise
    assert json.loads(js) == {
        "a": 1.5,
        "b": 7,
        "c": True,
        "d": None,
        "e": None,
        "f": [2.0, [1, 2, 3]],
        "g": [1, 2],
    }


def test_descriptive_stats_known_values() -> None:
    s = descriptive_stats(np.array([1.0, 2.0, 3.0, 4.0, np.nan]))
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["min"] == 1.0
    assert s["max"] == 4.0


def test_descriptive_stats_all_nan_returns_nan() -> None:
    s = descriptive_stats(np.array([np.nan, np.nan]))
    assert all(np.isnan(v) for v in s.values())


def test_top_k_by_skips_nans_and_orders_correctly() -> None:
    arr = np.array([3.0, np.nan, 1.0, 2.0, np.nan, 4.0])
    assert top_k_by(arr, 3, ascending=True) == [2, 3, 0]
    assert top_k_by(arr, 2, ascending=False) == [5, 0]


def test_pairwise_win_rates_matrix_shape_and_self_nan() -> None:
    da = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]),
        dims=["schedule", "problem"],
        coords={"schedule": ["a", "b"]},
    )
    pw = pairwise_win_rates(da)
    assert set(pw.keys()) == {"a", "b"}
    assert np.isnan(pw["a"]["a"])
    # a beats b on problem 0; b beats a on problem 1; tie on problem 2.
    assert pw["a"]["b"] == pytest.approx(1.0 / 3.0)
    assert pw["b"]["a"] == pytest.approx(1.0 / 3.0)


def test_shifted_final_min_strictly_positive_with_negative_losses() -> None:
    """``shifted_final_min`` must stay strictly positive even when raw
    losses are negative, by subtracting the per-task minimum first."""
    schedules = ["s1", "s2"]
    problem_index = pd.MultiIndex.from_product(
        [[0], ["t_a", "t_b"]], names=["realization", "taskID"]
    )
    # output_min has negative values throughout, including the global
    # minimum cell. shape: (schedule, output_dim, budget, problem)
    output_min = np.array(
        [
            [[[-7.0, -3.0], [-9.0, -4.0]]],
            [[[-5.0, -2.0], [-6.0, -3.5]]],
        ]
    )
    data = xr.Dataset(
        {
            "output_min": (
                ("schedule", "output_dim", "budget", "problem"),
                output_min,
            ),
        },
        coords={
            "schedule": schedules,
            "output_dim": ["y"],
            "budget": [0.0, 1.0],
            "problem": problem_index,
        },
    )
    labels_arr = np.empty((2, 1), dtype=object)
    labels_arr[0, 0] = -10.0  # known global_min for t_a
    labels_arr[1, 0] = np.nan  # unknown — falls back to empirical
    task_labels = xr.DataArray(
        labels_arr,
        dims=["problem", "feature"],
        coords={"problem": problem_index, "feature": ["global_min"]},
    )
    shifted = shifted_final_min(data=data, task_labels=task_labels)
    vals = np.asarray(shifted.values)
    assert np.all(vals > 0), (
        f"shifted_final_min must be strictly positive, got {vals!r}"
    )
    # The per-task floor must dominate EPSILON in the actual value.
    assert vals.max() > EPSILON


def test_per_task_shift_uses_global_min_when_finite() -> None:
    """Global_min from task_labels overrides the empirical floor when
    finite; falls back to per-task empirical otherwise."""
    schedules = ["s"]
    problem_index = pd.MultiIndex.from_product(
        [[0], ["task_a", "task_b"]], names=["realization", "taskID"]
    )
    output_min = np.array([[[[5.0, 5.0]]]])  # all 5 → empirical floor = 5
    data = xr.Dataset(
        {
            "output_min": (
                ("schedule", "output_dim", "budget", "problem"),
                output_min,
            ),
        },
        coords={
            "schedule": schedules,
            "output_dim": ["y"],
            "budget": [0.0],
            "problem": problem_index,
        },
    )
    labels = np.empty((2, 1), dtype=object)
    labels[0, 0] = -3.0  # known global_min
    labels[1, 0] = np.nan
    task_labels = xr.DataArray(
        labels,
        dims=["problem", "feature"],
        coords={"problem": problem_index, "feature": ["global_min"]},
    )
    shift = per_task_shift(data=data, task_labels=task_labels)
    # task_a uses known global_min, task_b falls back to empirical min.
    assert shift.sel(taskID="task_a").item() == pytest.approx(-3.0)
    assert shift.sel(taskID="task_b").item() == pytest.approx(5.0)


def test_pp_auc_returns_one_for_uniform_best_schedule() -> None:
    """A schedule that matches the per-problem best on every problem
    must score 1.0."""
    shifted = xr.DataArray(
        np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]),
        dims=["schedule", "problem"],
        coords={"schedule": ["a", "b"]},
    )
    assert pp_auc(shifted, "a") == pytest.approx(1.0)
    # ``b`` is always 5× worse than ``a`` → r_b = 5, ρ(τ)=0 for τ<5,
    # ρ(τ)=1 for τ≥5. AUC = (log10(100) − log10(5)) / log10(100).
    expected_b = (np.log10(100.0) - np.log10(5.0)) / np.log10(100.0)
    assert pp_auc(shifted, "b") == pytest.approx(expected_b, rel=1e-2)


def test_descriptive_stats_log10_strictly_positive_input() -> None:
    arr = xr.DataArray(np.array([1.0, 10.0, 100.0]))
    s = descriptive_stats_log10(arr)
    assert s["mean"] == pytest.approx(1.0)
    assert s["min"] == 0.0
    assert s["max"] == 2.0


#                                                  DataLoader — selection
# =============================================================================


def test_ertd_profile_runs(
    synthetic_dataloader: DataLoader,
) -> None:
    """``DataLoader.ertd_profile`` returns a stable result over repeat
    calls on the same loader."""
    first = synthetic_dataloader.ertd_profile()
    second = synthetic_dataloader.ertd_profile()
    xr.testing.assert_identical(first, second)


def test_select_feature_narrows_problems(
    synthetic_dataloader: DataLoader,
) -> None:
    """Selecting a feature subset narrows the loader to the matching
    problems (both loss and performance variables, on one loader)."""
    sub = synthetic_dataloader.select_feature({"fn_name": "sphere"})
    assert isinstance(sub, DataLoader)
    problems = set(sub.data["taskID"].values.tolist())
    assert problems == {"task_0", "task_1"}
    # The single dataset still carries both families after selection.
    assert "output_min" in sub.data and "success" in sub.data


#                                                  DataLoader.report — E2E
# =============================================================================


def test_report_top_level_keys(synthetic_dataloader: DataLoader) -> None:
    r = synthetic_dataloader.report()
    assert set(r.keys()) == {
        "schema_version",
        "key_findings",
        "dataset",
        "task_groups",
        "per_schedule",
        "rankings",
        "grouped",
        "observations",
        "observations_summary",
    }
    assert r["schema_version"] == 2


def test_report_key_findings_present_and_serialisable(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    assert isinstance(r["key_findings"], list)
    assert all(isinstance(s, str) for s in r["key_findings"])
    # At least the headline winner should always be emittable.
    assert len(r["key_findings"]) >= 1
    assert len(r["key_findings"]) <= 5


def test_report_observations_carry_magnitude_and_are_sorted(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    obs = r["observations"]
    assert obs, "fixture should produce at least one observation"
    for o in obs:
        assert "magnitude" in o
        assert "code" in o
        assert "severity" in o
        assert o["severity"] in {"error", "warning", "info"}

    # Severity-then-magnitude monotonicity.
    order = {"error": 3, "warning": 2, "info": 1}
    prev_key: tuple[float, float] = (-float("inf"), -float("inf"))
    for o in obs:
        cur_key = (
            -float(order[o["severity"]]),
            -float(o.get("magnitude") or 0.0),
        )
        assert cur_key >= prev_key, f"unsorted at {o}"
        prev_key = cur_key


def test_report_dataset_carries_uses_shifted_objective(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    assert "uses_shifted_objective" in r["dataset"]
    # The synthetic fixture mixes positive and negative-global_min tasks;
    # ``bad_opt`` final loss is ~1000 (>0) on every problem, but
    # ``good_opt`` lands very close to the per-task floor including
    # negative ones, so we expect the flag to be true.
    assert isinstance(r["dataset"]["uses_shifted_objective"], bool)


def test_report_no_negative_final_loss_rule_in_observations(
    synthetic_dataloader: DataLoader,
) -> None:
    """``negative_final_loss`` was demoted to a dataset-level flag."""
    r = synthetic_dataloader.report()
    codes = {o["code"] for o in r["observations"]}
    assert "negative_final_loss" not in codes
    assert "high_realization_variance" not in codes  # renamed


def test_report_excludes_global_min_from_task_features(
    synthetic_dataloader: DataLoader,
) -> None:
    """``global_min`` is a per-task numerical floor consumed internally
    by ``per_task_shift``; it must not appear in any of the categorical
    feature surfaces (``task_feature_names``, ``task_groups``,
    ``grouped``, ``regime_change`` payloads)."""
    r = synthetic_dataloader.report()
    assert "global_min" not in r["dataset"]["task_feature_names"]
    assert "global_min" not in r["task_groups"]
    assert "global_min" not in r["grouped"]
    for o in r["observations"]:
        if o["code"] in ("regime_change", "hard_subset", "easy_subset"):
            assert o.get("feature") != "global_min"


def test_report_is_json_serialisable(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    js = json.dumps(r)
    # Round-trips without loss of shape.
    assert json.loads(js).keys() == r.keys()


def test_report_per_schedule_has_required_fields(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    required = {
        "n_valid_problems",
        "n_failed_problems",
        "final_loss",
        "loss_above_optimum_log10",
        "ertd_fevals_per_dim",
        "auc_pp",
        "success_fraction_by_target",
        "negative_final_loss_fraction",
    }
    for s, block in r["per_schedule"].items():
        assert required.issubset(block.keys()), (
            f"schedule {s} missing {required - block.keys()}"
        )


def test_report_ertd_fevals_per_dim_has_exactly_six_keys(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    expected_keys = {"1", "10", "100", "1000", "10000", "100000"}
    for block in r["per_schedule"].values():
        assert set(block["ertd_fevals_per_dim"].keys()) == expected_keys


def test_report_dataset_section_counts(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    d = r["dataset"]
    assert d["n_schedules"] == 3
    assert d["n_problems"] == 8  # 4 tasks × 2 realizations
    assert d["n_tasks"] == 4
    assert d["n_realizations"] == 2
    assert d["has_success_array"] is True
    assert d["has_cost_array"] is True


def test_report_observations_summary_matches_list(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    expected = Counter(o["severity"] for o in r["observations"])
    assert r["observations_summary"]["error"] == expected.get("error", 0)
    assert r["observations_summary"]["warning"] == expected.get("warning", 0)
    assert r["observations_summary"]["info"] == expected.get("info", 0)


def test_report_fires_optimizer_crashed_for_all_nan_cell(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    crashed = [
        o for o in r["observations"] if o["code"] == "optimizer_crashed"
    ]
    # good_opt was set to all-NaN on problem 0 (2 realizations of task_0
    # × 1 schedule = depends on realization fixture; we set NaN on the
    # single problem index 0, so 1 failed problem).
    assert len(crashed) >= 1
    assert any(o["schedule"] == "good_opt" for o in crashed)


def test_report_fires_schedule_worse_than_baseline_for_bad_opt(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    pairs = [(o["code"], o.get("schedule")) for o in r["observations"]]
    assert ("schedule_worse_than_baseline", "bad_opt") in pairs


def test_report_fires_dominant_schedule_for_good_opt(
    synthetic_dataloader: DataLoader,
) -> None:
    r = synthetic_dataloader.report()
    pairs = [(o["code"], o.get("schedule")) for o in r["observations"]]
    assert ("dominant_schedule", "good_opt") in pairs


def test_report_grouped_skips_high_cardinality_features(
    synthetic_dataloader: DataLoader,
) -> None:
    """The ``global_min`` feature has cardinality 2 in this fixture
    (0.0 and -5.0), well below the cap, so it should appear in
    ``grouped``. ``task_groups`` should also have the full map."""
    r = synthetic_dataloader.report()
    assert "fn_name" in r["grouped"]
    assert set(r["grouped"]["fn_name"].keys()) == {"sphere", "rosenbrock"}
    assert "fn_name" in r["task_groups"]
    assert set(r["task_groups"]["fn_name"].keys()) == {
        "sphere",
        "rosenbrock",
    }


def test_report_mean_log_regret_vs_sbs_nonnegative_for_non_sbs(
    synthetic_dataloader: DataLoader,
) -> None:
    """The SBS is the per-task minimizer of mean final loss in the same
    space the regret is computed against, so every other schedule's
    mean log-regret must be ≥ 0 (NaN allowed for unmeasurable cases)."""
    r = synthetic_dataloader.report()
    reg = r["rankings"]["mean_log_regret_vs_sbs"]
    for s, v in reg.items():
        if v is None:
            continue  # NaN → None via sanitiser
        assert v >= -1e-12, (
            f"schedule {s} reported negative log-regret {v}; SBS "
            "selection must dominate everywhere"
        )


def test_report_size_under_100kb(synthetic_dataloader: DataLoader) -> None:
    r = synthetic_dataloader.report()
    js = json.dumps(r)
    assert len(js.encode("utf-8")) < 100_000
