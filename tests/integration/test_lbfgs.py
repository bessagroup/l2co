"""Behavioral integration tests for the ``"lbfgs"`` registry entry.

Complements the shape-level coverage that ``test_all_optimizers`` gives
``lbfgs`` (which, on its deterministic task, exercises the stock
``optax.lbfgs`` path) with the stochastic path: a full
``RolloutWrapper`` rollout on a noisy BBOB task, driving the
per-evaluation key splitting through the scan/batching machinery.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest
from l2co_tasks import create_bbob_task

from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout import RolloutWrapper
from l2co.sampling import normal_sampling

# =============================================================================


@pytest.mark.slow
def test_rollout_on_noisy_bbob():
    task = create_bbob_task(
        fn_name="sphere",
        seed=0,
        dimensionality=2,
        noise=0.01,
    )
    assert task.pass_rng

    key = jr.key(0)
    rollout_wrapper = RolloutWrapper.init(
        optimizer=OptimizationStep(optimizer="lbfgs"),
        task=task,
        sampler=normal_sampling,
        key=key,
        bounded=(None, None),
    )
    _, _, history_state = rollout_wrapper.batch_evaluate(
        key=key,
        n_iterations=5,
        n_realizations=2,
    )
    data = history_state.to_xarray_realizations()
    assert bool(jnp.isfinite(jnp.asarray(data["output_min"].values)).all())
