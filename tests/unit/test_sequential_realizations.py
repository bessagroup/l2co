"""Guards for the one-realization-at-a-time rollout driver.

A meta-optimizer's step picks a sub-optimizer from a menu and runs only
that one, written as a ``jax.lax.switch`` on the chosen index. ``vmap``
has no per-lane control flow, so vectorizing realizations together forces
every branch to run on every generation and throws all but one result
away -- which is why ``UpdateClass.batch_run_sequential`` exists and why
``UpdateClass.batch_run`` routes the meta-optimizers to it.

These tests pin, on real tasks built through l2co's ``RolloutWrapper``:

1. The sequential driver is a drop-in for the fused one -- same output
   structure, same shapes, same numbers -- on an optimizer where both are
   valid (no switch to collapse, so the choice is purely about speed).
2. The marker is on for l2co's meta-optimizer bridge.

The routing rule itself, and the marker being off by default, are
``l2co_optimizers``' to test (``tests/test_run_loop.py`` there), since
that is where ``batch_run`` lives now (ADR 0017).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.random as jr
import numpy as np
import pytest
from l2co_tasks import create_bbob_task

from l2co._src.meta_optimizer import L2COUpdateClass
from l2co._src.rollout import RolloutWrapper
from l2co._src.run_state import batch_reset
from l2co.optimization import OptimizationStep
from l2co.sampling import normal_sampling

N_ITERATIONS = 6
N_REALIZATIONS = 3

# float32 reduction-order tolerance; see the comparison comment below.
_RTOL = 1e-4
_ATOL = 1e-5

# One gradient optimizer and one population optimizer: the two shapes of
# `(popsize, dim)` bookkeeping the drivers have to agree about.
_PLAIN_STEPS = [
    OptimizationStep("adam", hyperparameters={"learning_rate": 0.05}),
    OptimizationStep("cmaes", hyperparameters={"popsize": 4}),
]
_STEP_IDS = ["adam", "cmaes"]


@pytest.fixture
def sampler():
    return partial(normal_sampling, mean=0.0, std=1.0)


def _task():
    """A cheap 4-D BBOB task; the drivers do not care which function."""
    return create_bbob_task(fn_name="sphere", seed=0, dimensionality=4)


def _driver_inputs(step: OptimizationStep, sampler, key):
    """Assemble the ``UpdateClass.batch_run_*`` arguments, plus the optimizer.

    Mirrors what the module-level ``batch_evaluate`` does before handing
    off to ``batch_run``: split the key per realization, then reset to
    get an ``opt_state`` / ``params`` carrying the realization axis.
    """
    wrapper = RolloutWrapper.init(
        optimizer=step,
        task=_task(),
        sampler=sampler,
        key=key,
        bounded=(None, None),
    )
    keys = jr.split(key, N_REALIZATIONS)
    run_state = batch_reset(
        wrapper.run_state,
        wrapper.batch_state,
        wrapper.static,
        wrapper.dataset,
        wrapper.loss_fn,
        wrapper.pass_rng,
        sampler,
        keys,
    )
    return run_state.update_class, dict(
        opt_state=run_state.opt_state,
        params=run_state.params,
        batch_state=wrapper.batch_state,
        dataset=wrapper.dataset,
        key=keys,
        n_iterations=N_ITERATIONS,
        verbose=False,
    )


#                                                    drop-in equivalence
# =============================================================================


@pytest.mark.parametrize("step", _PLAIN_STEPS, ids=_STEP_IDS)
def test_sequential_driver_matches_fused_driver(step, sampler, rng):
    """Both drivers return the same thing for the same inputs.

    Run on plain optimizers on purpose: with no ``lax.switch`` to
    collapse, the two lowerings compute the same arithmetic, so any
    disagreement here is a bug in the sequential driver's plumbing --
    a mis-sliced key, a lost realization axis, a different reduction --
    rather than the branch-selection difference the driver's docstring
    allows for the meta-optimizers.
    """
    update_class, inputs = _driver_inputs(step, sampler, rng)

    fused = update_class.batch_run_fused(**inputs)
    sequential = update_class.batch_run_sequential(**inputs)

    fused_leaves = jax.tree.leaves(fused)
    sequential_leaves = jax.tree.leaves(sequential)

    assert jax.tree.structure(fused) == jax.tree.structure(sequential)
    assert len(fused_leaves) == len(sequential_leaves)

    for a, b in zip(fused_leaves, sequential_leaves, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        assert a.shape[0] == N_REALIZATIONS
        # NaN placement is exact -- it encodes padding and skipped
        # steps, which is structure, not arithmetic. Finite values get
        # float32 tolerance: the fused driver vmaps the batch, XLA
        # retiles the reductions, and CMA-ES compounds that difference
        # through its covariance update over `N_ITERATIONS` generations.
        # `_ATOL` is what that compounding costs on a small-magnitude
        # coordinate; a genuine plumbing bug (mis-sliced key, dropped
        # realization axis) moves values by O(1), not by 1e-5.
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a, b, rtol=_RTOL, atol=_ATOL, equal_nan=True)


@pytest.mark.parametrize("step", _PLAIN_STEPS, ids=_STEP_IDS)
def test_sequential_driver_realizations_are_independent(step, sampler, rng):
    """Realization ``r`` of the batch equals realization ``r`` alone.

    The point of the sequential driver is that each realization is its
    own scalar-index run, so slicing one out of a 3-realization call must
    match running it by itself.
    """
    update_class, inputs = _driver_inputs(step, sampler, rng)
    batched = update_class.batch_run_sequential(**inputs)

    for r in range(N_REALIZATIONS):
        single = update_class.batch_run_sequential(
            **{
                **inputs,
                "opt_state": jax.tree.map(
                    lambda x, r=r: x[r : r + 1], inputs["opt_state"]
                ),
                "params": jax.tree.map(
                    lambda x, r=r: x[r : r + 1], inputs["params"]
                ),
                "key": inputs["key"][r : r + 1],
            }
        )
        for a, b in zip(
            jax.tree.leaves(batched), jax.tree.leaves(single), strict=True
        ):
            a = np.asarray(a)[r : r + 1]
            b = np.asarray(b)
            assert np.array_equal(np.isnan(a), np.isnan(b))
            assert np.allclose(a, b, rtol=_RTOL, atol=_ATOL, equal_nan=True)


#                                                          the routing marker
# =============================================================================


def test_marker_is_on_for_the_meta_optimizer_bridge():
    # l2co's own two-stage strategy dispatches on a per-realization
    # sub-optimizer index; rl2co asserts the same for its bridge in
    # its own suite (l2co may not import rl2co).
    assert L2COUpdateClass.sequential_realizations is True
