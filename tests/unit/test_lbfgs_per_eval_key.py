"""Unit tests for the per-eval-key L-BFGS optimizer.

Covers the key-flow contract (every linesearch evaluation receives its
own freshly split PRNG key — the property the stock optax wiring cannot
provide), the deterministic-task fallback (bit-identical trajectories
to the stock ``lbfgs`` registry entry), and the registry wiring.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from l2co_tasks import Task

from l2co._src.optimizers.mapping import optimizer_mapping, optimizers

# =============================================================================


def _noisy_quadratic(beta: float):
    """Multiplicative log-normal noise on a quadratic bowl."""

    def loss_fn(model, key):
        return jnp.sum(model**2) * jnp.exp(beta * jr.normal(key, ()))

    return loss_fn


def _quadratic(model):
    return jnp.sum(model**2)


def _build(task: Task):
    return optimizer_mapping("lbfgs_per_eval_key")(
        task=task,
        opt_hash=1,
        bounded=(-5.0, 5.0),
        stop_fn=None,
    )


def _run_steps(update_class, n_steps: int, dim: int = 4, jit: bool = True):
    params = jnp.full((1, dim), 2.0)
    state = update_class.init_fn(params, jr.key(0))
    carry = (params, state, jr.key(0))
    step = (
        eqx.filter_jit(update_class.step_fn) if jit else (update_class.step_fn)
    )
    losses = []
    for _ in range(n_steps):
        carry, history = step(carry, {})
        losses.append(history.loss)
    return carry[0], jnp.stack(losses)


# =============================================================================
#                                                                      Registry


def test_registry_contains_lbfgs_per_eval_key():
    assert "lbfgs_per_eval_key" in optimizers
    assert callable(optimizer_mapping("lbfgs_per_eval_key"))


# =============================================================================
#                                                                      Key flow


def test_every_linesearch_evaluation_gets_a_fresh_key():
    """Within one outer step, no two evaluations share a realization.

    A callback inside the loss function records the key of every
    single evaluation. Severe multiplicative noise forces the zoom
    linesearch through several trial evaluations per step; each must
    arrive with a distinct key (the stock optax wiring would show one
    key repeated for the entire step).
    """
    received = []

    def logging_loss(model, key):
        jax.debug.callback(
            lambda kd: received.append(np.asarray(kd).copy()),
            jr.key_data(key),
        )
        return jnp.sum(model**2) * jnp.exp(1.0 * jr.normal(key, ()))

    task = Task(
        model=jnp.zeros(4),
        loss_fn=logging_loss,
        global_min=0.0,
        pass_rng=True,
    )
    update_class = _build(task)

    params = jnp.full((1, 4), 2.0)
    state = update_class.init_fn(params, jr.key(0))
    carry = (params, state, jr.key(0))

    per_step_keys = []
    for _ in range(3):
        received.clear()
        carry, _ = update_class.step_fn(carry, {})
        jax.block_until_ready(carry[0])
        per_step_keys.append([tuple(k.ravel().tolist()) for k in received])

    all_keys = [k for step_keys in per_step_keys for k in step_keys]
    # Several evaluations happened, and every single one used its own
    # key — within a step and across steps.
    assert len(all_keys) >= 3
    assert len(set(all_keys)) == len(all_keys)
    # At least one step exercised multiple linesearch evaluations.
    assert max(len(step_keys) for step_keys in per_step_keys) >= 2


# =============================================================================
#                                                        Deterministic fallback


def test_deterministic_task_matches_stock_lbfgs():
    """Without ``pass_rng`` the factory degrades to plain L-BFGS.

    Fresh keys per evaluation of a deterministic loss are a
    mathematical no-op, so the trajectories must be bit-identical to
    the stock ``lbfgs`` registry entry under the same seeds.
    """
    task = Task(
        model=jnp.zeros(4),
        loss_fn=_quadratic,
        global_min=0.0,
        pass_rng=False,
    )
    per_eval = _build(task)
    stock = optimizer_mapping("lbfgs")(
        task=task,
        opt_hash=1,
        bounded=(-5.0, 5.0),
        stop_fn=None,
    )
    params_per_eval, losses_per_eval = _run_steps(per_eval, n_steps=5)
    params_stock, losses_stock = _run_steps(stock, n_steps=5)
    assert jnp.array_equal(params_per_eval, params_stock)
    assert jnp.array_equal(losses_per_eval, losses_stock)


# =============================================================================
#                                                                   Convergence


def test_converges_on_mildly_noisy_quadratic():
    """Sanity: the machinery still optimizes when noise is small."""
    task = Task(
        model=jnp.zeros(4),
        loss_fn=_noisy_quadratic(beta=1e-3),
        global_min=0.0,
        pass_rng=True,
    )
    update_class = _build(task)
    final_params, losses = _run_steps(update_class, n_steps=15)
    assert bool(jnp.isfinite(losses).all())
    assert float(jnp.linalg.norm(final_params)) < 1e-3


def test_stays_finite_under_severe_noise():
    """Severe noise may stall progress but must never produce NaNs."""
    task = Task(
        model=jnp.zeros(4),
        loss_fn=_noisy_quadratic(beta=1.0),
        global_min=0.0,
        pass_rng=True,
    )
    update_class = _build(task)
    final_params, losses = _run_steps(update_class, n_steps=15)
    assert bool(jnp.isfinite(losses).all())
    assert bool(jnp.isfinite(final_params).all())
