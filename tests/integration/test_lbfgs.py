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

from l2co_optimizers import OptimizationStep
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


@pytest.mark.slow
def test_converged_lbfgs_keeps_billing_the_linesearch():
    """A converged run keeps spending its full linesearch budget.

    On a real BBOB landscape L-BFGS drives the gradient down to a
    small but *non-zero* residual. The Wolfe curvature criterion
    compares ``|slope_step|`` against ``0.9 * |slope_init|`` with
    ``tol=0.0``, so at that residual it can no longer be satisfied
    exactly: the search runs to ``max_linesearch_steps`` and fails out,
    every step, for the rest of the budget. Those evaluations are real,
    and the flat ``1`` of ``docs/adr/0006`` hid them -- a converged
    optimizer looked free (``docs/adr/0010``).

    This needs a real task: a textbook quadratic converges to an
    *exactly* zero gradient, where both Wolfe errors are exactly
    ``0.0`` and the search legitimately succeeds in one trial. The
    saturation is a property of realistic landscapes, not of L-BFGS in
    the abstract.
    """
    import equinox as eqx

    from l2co_optimizers import optimizer_mapping
    from l2co_optimizers import random_sampling

    cap = 5
    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=10)
    assert not task.pass_rng

    update_class = optimizer_mapping("lbfgs")(
        task=task,
        opt_hash=1,
        bounded=(None, None),
        stop_fn=None,
        max_linesearch_steps=cap,
    )
    assert update_class.fevals_bound == cap + 1

    key, sample_key = jr.split(jr.key(0))
    template, _ = eqx.partition(task.model, eqx.is_inexact_array)
    params = random_sampling(sample_key, template, 1)
    carry = (params, update_class.init_fn(params, key=key), key)

    fevals = []
    for _ in range(12):
        carry, history = update_class.step_fn(carry, {})
        fevals.append(int(history.fevals))

    # Cheap while descending; dominated by exhausted searches after.
    # (The post-convergence tail is not uniformly at the cap -- at
    # small caps it alternates -- so assert the balance, not a pin.)
    descent, converged = fevals[1:3], fevals[3:]
    assert max(descent) == 1, fevals
    assert converged.count(cap) >= len(converged) // 2, fevals
    assert max(fevals) <= update_class.fevals_bound, fevals
