"""Filling in task-tag features at the model level.

``tests/unit/test_feature_impute.py`` covers ``FeatureProcessor.with_imputation``
itself. This file covers the layer above it: ``Stage0Model``,
``Stage1Model`` and ``L2COModel`` all hold ``feature_processor_task`` as a
plain field, and each gets a ``with_feature_impute`` that rebuilds only
that field via ``eqx.tree_at`` -- same idiom as the existing
``swap_optimizers``, so the trained weights and the optimizer menu are
carried over untouched.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co import (
    EquinoxStage0Model,
    EquinoxStage1Model,
    FeatureProcessor,
    L2COModel,
    MixedEmbedding,
    OptimizationStep,
)
from l2co.models import MLP, LSTM_Encoder

MENU = (OptimizationStep(optimizer="lbfgs"), OptimizationStep(optimizer="pso"))

# Trained against a task tag that carries `separable`; a foreign suite's
# tag (mirroring cec2005 / quadratic in l2co_experiments) does not.
SCHEMA = {"dimensionality": "continuous", "separable": "boolean"}
SPARSE_TAG = {"dimensionality": 10}
FILL = {"separable": False}


def _stage0() -> EquinoxStage0Model:
    fp_task = FeatureProcessor.init(SCHEMA)
    mlp = MLP.init(
        in_size=2, out_size=len(MENU), width_size=8, depth=1, key=jr.key(0)
    )
    return EquinoxStage0Model.init(
        feature_processor_task=fp_task, model=mlp, optimizers=list(MENU)
    )


def _stage1() -> EquinoxStage1Model:
    key = jr.key(1)
    k_enc, k_task, k_opt, k_mlp = jr.split(key, 4)
    fp_task = FeatureProcessor.init(SCHEMA)
    fp_opt = FeatureProcessor.init(
        schema={"optimizer": [step.name for step in MENU]}
    )
    me_task = MixedEmbedding.from_feature_processor(
        feature_processor=fp_task, embed_dim=4, output_dim=8, key=k_task
    )
    me_opt = MixedEmbedding.from_feature_processor(
        feature_processor=fp_opt, embed_dim=4, output_dim=8, key=k_opt
    )
    history_encoder = LSTM_Encoder(
        in_size=2, hidden_size=8, out_size=8, key=k_enc
    )
    mlp = MLP.init(
        in_size=8 * 2 + 8,
        out_size=len(MENU),
        width_size=8,
        depth=1,
        key=k_mlp,
    )
    return EquinoxStage1Model(
        feature_processor_task=fp_task,
        feature_processor_optimizer=fp_opt,
        mixed_embedding_task=me_task,
        mixed_embedding_optimizer=me_opt,
        history_encoder=history_encoder,
        model=mlp,
        optimizers=list(MENU),
    )


#                                                                    Stage0Model
# =============================================================================


def test_stage0_predict_raises_without_impute():
    with pytest.raises(KeyError, match="separable"):
        _stage0().predict(x_feature_task=SPARSE_TAG)


def test_stage0_with_feature_impute_predicts():
    model = _stage0().with_feature_impute(FILL)
    assert model.predict(x_feature_task=SPARSE_TAG) in MENU


def test_stage0_with_feature_impute_preserves_shape_and_weights():
    plain = _stage0()
    imputed = plain.with_feature_impute(FILL)

    assert list(plain.feature_processor_task.vocabs) == list(
        imputed.feature_processor_task.vocabs
    )
    assert bool(eqx.tree_equal(imputed.model, plain.model))
    assert imputed.optimizers == plain.optimizers


def test_stage0_with_feature_impute_does_not_mutate_the_original():
    plain = _stage0()
    plain.with_feature_impute(FILL)
    assert plain.feature_processor_task.impute == {}


#                                                                    Stage1Model
# =============================================================================


def _predict_stage1(model, x_feature_task):
    return model.predict(
        x_feature_task=x_feature_task,
        x_feature_optimizer={"optimizer": MENU[0].name},
        x_output_mean=jnp.array([0.0, 0.0]),
        x_output_min=jnp.array([0.0, 0.0]),
        global_min=0.0,
    )


def test_stage1_predict_raises_without_impute():
    with pytest.raises(KeyError, match="separable"):
        _predict_stage1(_stage1(), SPARSE_TAG)


def test_stage1_with_feature_impute_predicts():
    model = _stage1().with_feature_impute(FILL)
    assert _predict_stage1(model, SPARSE_TAG) in MENU


def test_stage1_with_feature_impute_leaves_the_optimizer_processor_alone():
    """Only the *task* processor is imputed -- the optimizer processor
    encodes the menu name, which every task set supplies alike."""
    plain = _stage1()
    imputed = plain.with_feature_impute(FILL)
    assert (
        imputed.feature_processor_optimizer
        == plain.feature_processor_optimizer
    )


#                                                                     L2COModel
# =============================================================================


def _model() -> L2COModel:
    return L2COModel(stage0_model=_stage0(), stage1_model=_stage1())


def test_l2co_model_predict_stage0_raises_without_impute():
    with pytest.raises(KeyError, match="separable"):
        _model().predict_stage0(task_features=SPARSE_TAG)


def test_l2co_model_with_feature_impute_fixes_both_stages():
    model = _model().with_feature_impute(FILL)

    assert model.predict_stage0(task_features=SPARSE_TAG) in MENU
    assert _predict_stage1(model.stage1_model, SPARSE_TAG) in MENU


def test_l2co_model_with_feature_impute_does_not_touch_the_original():
    plain = _model()
    plain.with_feature_impute(FILL)

    with pytest.raises(KeyError, match="separable"):
        plain.predict_stage0(task_features=SPARSE_TAG)
