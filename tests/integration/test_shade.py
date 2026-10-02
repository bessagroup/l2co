"""Behavioral integration tests for the SHADE optimizer.

Complements the shape-level coverage that ``test_all_optimizers`` gives
`shade` with an algorithmic-quality check: SHADE's success-history
adaptation must not lose to plain DE at an equal fitness-evaluation
budget. A silent breakage of the memory / archive machinery (e.g. the
memories never updating) degrades SHADE towards random-parameter DE,
which this catches.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from l2co_tasks import create_bbob_task

from l2co_optimizers import optimizer_mapping

# =============================================================================

_POPSIZE = 10
_N_GENERATIONS = 150
_SEEDS = range(6)


def _best_error(name: str, seed: int, **hyperparameters) -> float:
    """Best-so-far error after a fixed-budget run on rastrigin 5D."""
    task = create_bbob_task(fn_name="rastrigin", seed=0, dimensionality=5)
    update_class = optimizer_mapping(name)(
        task=task,
        opt_hash=1,
        bounded=(0.0, 1.0),
        stop_fn=None,
        **hyperparameters,
    )
    params, _ = eqx.partition(task.model, eqx.is_inexact_array)
    key_init, key_run = jr.split(jr.key(seed))
    population = jax.tree.map(
        lambda x: jr.uniform(key_init, (update_class.popsize, *x.shape)),
        params,
    )
    state = update_class.init_fn(population, key_run)
    carry = (population, state, key_run)
    step = eqx.filter_jit(update_class.step_fn)
    best = jnp.inf
    for _ in range(_N_GENERATIONS):
        carry, history = step(carry, {})
        best = jnp.minimum(best, jnp.min(history.loss))
    return float(best - task.global_min)


@pytest.mark.slow
def test_shade_not_worse_than_de_on_rastrigin():
    shade_errors = [
        _best_error("shade", seed, popsize=_POPSIZE) for seed in _SEEDS
    ]
    de_errors = [
        _best_error("differentialevolution", seed, popsize=_POPSIZE)
        for seed in _SEEDS
    ]
    median_shade = float(jnp.median(jnp.array(shade_errors)))
    median_de = float(jnp.median(jnp.array(de_errors)))
    assert all(jnp.isfinite(jnp.array(shade_errors)))
    # Generous margin: SHADE must be at least on par with plain DE.
    assert median_shade <= median_de * 1.10 + 1e-6, (
        f"SHADE median error {median_shade} vs DE {median_de}"
    )


@pytest.mark.slow
def test_turning_shade_runs_and_converges():
    error = _best_error(
        "shade",
        seed=0,
        popsize=_POPSIZE,
        use_turning=True,
        max_fevals=float(_POPSIZE * _N_GENERATIONS),
    )
    assert jnp.isfinite(error)
    # Sanity: far better than the ~1e4-scale loss of a random population.
    assert error < 1e3
