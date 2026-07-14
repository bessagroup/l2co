"""Integration test: the L2CO strategy through the registry path.

End-to-end over the canonical execution path (``docs/adr/0002``):
stage-0 (scikit) and stage-1 (Equinox) models stored to disk, an
``OptimizationStep(optimizer="l2co")`` resolved through the optimizer
registry by ``RolloutWrapper.init``, and a jitted, vmapped
``batch_evaluate`` rollout. Both fires are forced (constant-output
stage-0 regressor, one-hot-biased stage-1 head) so the run
deterministically switches from sepcmaes to adam at the switch
generation, which the stored history must show via the dynamic
per-generation ``update_step`` hash.
"""

from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from l2co_tasks import create_bbob_task
from sklearn.linear_model import LinearRegression

from l2co import (
    EquinoxStage1Model,
    FeatureProcessor,
    MixedEmbedding,
    OptimizationStep,
    RolloutWrapper,
    ScikitStage0Model,
)
from l2co.budget import fixed_budget
from l2co.models import MLP, LSTM_Encoder, Stage0Loader, Stage1Loader
from l2co.sampling import normal_sampling

N_ITERATIONS = 8
T = 0.5  # switch generation = 4
N_REALIZATIONS = 2

SEPCMAES = OptimizationStep(optimizer="sepcmaes")
ADAM = OptimizationStep(
    optimizer="adam", hyperparameters={"learning_rate": 0.03}
)
MENU = (SEPCMAES, ADAM)


def _store_models(tmp_path):
    """Store a forced scikit stage-0 (-> sepcmaes) and a forced
    Equinox stage-1 (-> adam) the way the training pipelines do."""
    fp_task = FeatureProcessor.init(schema={"dimensionality": "continuous"})
    fp_opt = FeatureProcessor.init(
        schema={"optimizer": [step.name for step in MENU]}
    )

    # Stage 0: constant-output regressor -> argmax always sepcmaes.
    stage0 = ScikitStage0Model.init(
        feature_processor_task=fp_task,
        model=LinearRegression(),
        optimizers=list(MENU),
    )
    stage0.model.fit(
        np.array([[2.0], [5.0], [10.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    )
    Stage0Loader.store(stage0, str(tmp_path / "stage0"))

    # Stage 1: zero final weights + one-hot bias -> argmax always adam.
    k_enc, k_task, k_opt, k_mlp = jr.split(jr.key(7), 4)
    mlp = MLP.init(
        in_size=8 * 2 + 8,
        out_size=len(MENU),
        width_size=8,
        depth=1,
        key=k_mlp,
    )
    last = mlp.model.layers[-1]
    mlp = eqx.tree_at(
        lambda m: (m.model.layers[-1].weight, m.model.layers[-1].bias),
        mlp,
        (
            jnp.zeros_like(last.weight),
            jnp.zeros_like(last.bias).at[1].set(10.0),
        ),
    )
    stage1 = EquinoxStage1Model(
        feature_processor_task=fp_task,
        feature_processor_optimizer=fp_opt,
        mixed_embedding_task=MixedEmbedding.from_feature_processor(
            feature_processor=fp_task, embed_dim=4, output_dim=8, key=k_task
        ),
        mixed_embedding_optimizer=MixedEmbedding.from_feature_processor(
            feature_processor=fp_opt, embed_dim=4, output_dim=8, key=k_opt
        ),
        history_encoder=LSTM_Encoder(
            in_size=2, hidden_size=8, out_size=8, key=k_enc
        ),
        model=mlp,
        optimizers=list(MENU),
    )
    Stage1Loader.store(stage1, str(tmp_path / "stage1"))


@pytest.mark.slow
def test_l2co_registry_rollout_switches_at_switch_generation(tmp_path):
    _store_models(tmp_path)
    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)

    step = OptimizationStep(
        optimizer="l2co",
        hyperparameters={
            "stage0_model": str(tmp_path / "stage0"),
            "stage1_model": str(tmp_path / "stage1"),
            "t": T,
            "sampler": normal_sampling,
            "budget_fn": partial(fixed_budget, budget=N_ITERATIONS),
            "swap_optimizers": None,
        },
    )

    rollout_wrapper = RolloutWrapper.init(
        optimizer=step,
        task=task,
        sampler=normal_sampling,
        key=jr.key(0),
    )
    _, _, history_state = rollout_wrapper.batch_evaluate(
        key=jr.key(1),
        n_iterations=N_ITERATIONS,
        n_realizations=N_REALIZATIONS,
    )

    switch_generation = round(T * N_ITERATIONS)
    sep_hash, adam_hash = int(SEPCMAES.hash), int(ADAM.hash)

    assert history_state.update_step.shape == (
        N_REALIZATIONS,
        N_ITERATIONS,
    )
    for r in range(N_REALIZATIONS):
        hashes = [int(h) for h in history_state.update_step[r]]
        # run_ scans n+1 steps and drops the first entry, so the
        # stored history covers generations 2..n+1: the stage-0
        # optimizer owns generations 1..switch_generation (of which
        # switch_generation - 1 survive the drop) and the switched-to
        # adam owns the rest.
        expected = [sep_hash] * (switch_generation - 1) + [adam_hash] * (
            N_ITERATIONS - switch_generation + 1
        )
        assert hashes == expected

        # Feval billing follows the active optimizer's popsize.
        fevals = [int(f) for f in history_state.fevals[r]]
        sep_popsize = fevals[0]
        assert sep_popsize > 1
        assert fevals == [sep_popsize] * (switch_generation - 1) + [1] * (
            N_ITERATIONS - switch_generation + 1
        )

        # The recorded losses are real evaluations, above the optimum.
        mins = history_state.output_min[r]
        assert bool(jnp.isfinite(mins).all())
        assert bool((mins >= task.global_min).all())
