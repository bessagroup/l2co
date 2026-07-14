"""Tests for the ``OptimizationStep.name`` special branches.

Learned meta-optimizers carry path-valued hyperparameters; the generic
``{optimizer}_{k=v}`` join would put absolute checkpoint paths into every
SCHEDULE label, plot legend, and databank key. ``rl2co`` and ``l2co``
therefore name themselves from the checkpoint stems instead — ``l2co``
from both stage checkpoints plus the stage-0 fraction ``t``, so every
(stage-0, stage-1) pair in a model-selection grid gets a distinct,
legible label.
"""

from l2co import OptimizationStep


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


def test_rl2co_and_generic_names_unchanged():
    rl2co = OptimizationStep(
        optimizer="rl2co",
        hyperparameters={"model": "/x/experiment_data/model/7"},
    )
    assert rl2co.name == "rl2co_7"

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
