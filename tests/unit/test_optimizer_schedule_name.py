"""Tests for how l2co names its meta-optimizer schedules.

Learned meta-optimizers carry path-valued hyperparameters; the generic
``{optimizer}_{k=v}`` join would put absolute checkpoint paths into every
SCHEDULE label, plot legend, and databank key. ``OptimizationStep.name``
lives in ``l2co_optimizers`` and knows no meta-optimizer, so l2co
registers a namer for ``"l2co"`` (``register_schedule_namer``, l2co ADR
0016): both stage checkpoint stems plus the stage-0 fraction ``t``, so
every (stage-0, stage-1) pair in a model-selection grid gets a distinct,
legible label. rl2co registers and tests its own ``"rl2co"`` namer.
"""

from l2co import ALIAS_HYPERPARAMETER, OptimizationStep


def test_alias_overrides_derived_name():
    # An explicit alias short-circuits the stem-derived name for a learned
    # optimizer (the eval pipeline keys the run by its training run id).
    step = OptimizationStep(
        optimizer="rl2co",
        hyperparameters={
            "model": "/x/experiment_data/model/7",
            ALIAS_HYPERPARAMETER: "rl2co_2026-07-22T10-00-00Z",
        },
    )
    assert step.name == "rl2co_2026-07-22T10-00-00Z"


def test_empty_alias_falls_back_to_derivation():
    step = OptimizationStep(
        optimizer="l2co",
        hyperparameters={
            "stage0_model": "/x/experiment_data/model/7",
            ALIAS_HYPERPARAMETER: "",
        },
    )
    assert step.name == "l2co_s0-7"


def test_alias_does_not_change_policy_identity():
    # Two evaluations of the SAME weights under different aliases must share
    # one policy identity (identity is weights-based, alias is a label).
    from l2co.databank import policy_identity_hash

    record = {"training": {"weights_hash": "deadbeef"}}
    base_hp = {"model": "/x/model/7", "n_steps": 100}
    a = OptimizationStep(
        optimizer="rl2co", hyperparameters={**base_hp, "alias": "rl2co_A"}
    )
    b = OptimizationStep(
        optimizer="rl2co", hyperparameters={**base_hp, "alias": "rl2co_B"}
    )
    assert policy_identity_hash(record, a) == policy_identity_hash(record, b)


def test_l2co_name_from_stems_and_t():
    step = OptimizationStep(
        optimizer="l2co",
        hyperparameters={
            "stage0_model": "/scratch/l2co_train_stage0/17/experiment_data/model/10",
            "stage1_model": "/scratch/l2co_train_stage1/18/experiment_data/model/25",
            "t": 0.05,
            "swap_optimizers": None,
        },
    )
    assert step.name == "l2co_s0-10_s1-25_t-0.05"


def test_l2co_name_distinct_across_pairs():
    def make(s0: int, s1: int) -> OptimizationStep:
        return OptimizationStep(
            optimizer="l2co",
            hyperparameters={
                "stage0_model": f"/run0/experiment_data/model/{s0}",
                "stage1_model": f"/run1/experiment_data/model/{s1}",
                "t": 0.01,
            },
        )

    names = {make(s0, s1).name for s0 in range(3) for s1 in range(3)}
    assert len(names) == 9


def test_l2co_name_omits_missing_parts():
    assert OptimizationStep(optimizer="l2co").name == "l2co"
    assert (
        OptimizationStep(
            optimizer="l2co",
            hyperparameters={"stage0_model": "/x/model/3"},
        ).name
        == "l2co_s0-3"
    )


def test_generic_names_unchanged():
    adam = OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 0.03}
    )
    assert adam.name == "adam_learning_rate=0.03"

    assert (
        OptimizationStep(
            optimizer="randomsearch", hyperparameters={"sampler": "random"}
        ).name
        == "randomsearch"
    )
