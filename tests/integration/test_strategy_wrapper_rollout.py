"""Integration test: the L2CO strategy through the registry path.

End-to-end over the canonical execution path (``docs/adr/0002``):
stage-0 (scikit) and stage-1 (Equinox) models stored to disk, an
``OptimizationStep(optimizer="l2co")`` resolved through the optimizer
registry by ``RolloutWrapper.init``, and a jitted, vmapped
``batch_evaluate`` rollout. Both fires are forced (constant-output
stage-0 regressor, one-hot-biased stage-1 head) so the run
deterministically switches from sepcmaes to adam once ``t`` of the
evaluation budget B is spent, which the stored history must show via
the dynamic per-generation ``update_step`` hash, and stops after the
first generation that reaches B (``docs/adr/0029``).
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
from l2co.budget import evaluation_budget
from l2co.models import MLP, LSTM_Encoder, Stage0Loader, Stage1Loader
from l2co.sampling import normal_sampling

EVALS_PER_DIM = 150  # B = 300 on the 2-D sphere
T = 0.5  # stage-1 fires once 150 evaluations are spent
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


def _rollout(tmp_path, stopping_criteria=None):
    """Run the stored l2co models through the registry under B = 300.

    Returns ``(task, budget, history_state)``. The run's scan length is
    what dataset generation gives it,
    ``budget_fn(task=task, optimizer=update_class)``.
    """
    _store_models(tmp_path)
    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)
    budget_fn = partial(
        evaluation_budget, evaluations_per_dimension=EVALS_PER_DIM
    )
    step = OptimizationStep(
        optimizer="l2co",
        hyperparameters={
            "stage0_model": str(tmp_path / "stage0"),
            "stage1_model": str(tmp_path / "stage1"),
            "t": T,
            "sampler": normal_sampling,
            "budget_fn": budget_fn,
            "swap_optimizers": None,
        },
        stopping_criteria=stopping_criteria or {},
    )
    rollout_wrapper = RolloutWrapper.init(
        optimizer=step,
        task=task,
        sampler=normal_sampling,
        key=jr.key(0),
    )
    n_iterations = budget_fn(
        task=task, optimizer=rollout_wrapper.run_state.update_class
    )
    _, _, history_state = rollout_wrapper.batch_evaluate(
        key=jr.key(1),
        n_iterations=n_iterations,
        n_realizations=N_REALIZATIONS,
    )
    return task, budget_fn(task=task), history_state


@pytest.mark.slow
def test_l2co_registry_rollout_switches_and_stops_by_evaluations(tmp_path):
    task, budget, history_state = _rollout(tmp_path)
    assert budget == 300
    # The run's scan is B long, the most generations B could buy.
    assert history_state.update_step.shape == (N_REALIZATIONS, budget)

    sep_hash, adam_hash = int(SEPCMAES.hash), int(ADAM.hash)
    switch_fevals = round(T * budget)
    for r in range(N_REALIZATIONS):
        fevals = np.asarray(history_state.fevals[r])
        hashes = np.asarray(history_state.update_step[r])
        ran = fevals > 0
        n_ran = int(ran.sum())
        # The generations that ran come first; every later one is a
        # skipped NaN entry that bills nothing.
        assert ran[:n_ran].all()
        assert not ran[n_ran:].any()
        assert n_ran < budget
        assert bool(jnp.isnan(history_state.output_min[r, n_ran:]).all())

        # Stage-0 (sepcmaes) runs until it has spent t * B evaluations;
        # the next generation is adam's.
        sep_popsize = int(fevals[0])
        assert sep_popsize > 1
        n_sep = -(-switch_fevals // sep_popsize)
        assert (hashes[:n_sep] == sep_hash).all()
        assert (fevals[:n_sep] == sep_popsize).all()
        assert (hashes[n_sep:n_ran] == adam_hash).all()

        # The run stops after the first generation that reaches B:
        # adam costs 1, so it overshoots by less than one generation.
        total = int(fevals.sum())
        assert budget <= total < budget + 1
        assert int(fevals[: n_ran - 1].sum()) < budget

        mins = history_state.output_min[r, :n_ran]
        assert bool(jnp.isfinite(mins).all())
        assert bool((mins >= task.global_min).all())


@pytest.mark.slow
def test_l2co_registry_rollout_user_stopping_criteria_still_stop(tmp_path):
    # A loss tolerance no run can miss stops the run once the recent
    # history window is full, long before B.
    _, budget, history_state = _rollout(
        tmp_path,
        stopping_criteria={"type": "loss_based", "loss_tol": 1e30},
    )
    for r in range(N_REALIZATIONS):
        fevals = np.asarray(history_state.fevals[r])
        assert 0 < int(fevals.sum()) < budget // 2
