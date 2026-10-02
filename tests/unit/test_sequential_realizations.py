"""Guards for the one-realization-at-a-time rollout driver.

A meta-optimizer's step picks a sub-optimizer from a menu and runs only
that one, written as a ``jax.lax.switch`` on the chosen index. ``vmap``
has no per-lane control flow, so vectorizing realizations together forces
every branch to run on every generation and throws all but one result
away -- which is why
:func:`l2co._src.update_class.batch_run_sequential_` exists and why
:func:`l2co._src.run_state.batch_run` routes the meta-optimizers to it.

These tests pin the two things that has to be true:

1. The sequential driver is a drop-in for the fused one -- same output
   structure, same shapes, same numbers -- on an optimizer where both are
   valid (no switch to collapse, so the choice is purely about speed).
2. ``batch_run`` actually routes on
   :attr:`l2co.UpdateClass.sequential_realizations`, and the marker is
   off for plain optimizers and on for the meta-optimizer bridge.
"""

from __future__ import annotations

import dataclasses
import inspect
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from l2co_tasks import create_bbob_task

from l2co._src.meta_optimizer import L2COUpdateClass
from l2co_optimizers import RandomSearchUpdateClass
from l2co_optimizers import UpdateClass
from l2co._src.rollout import RolloutWrapper
from l2co._src.run_state import batch_reset, batch_run
from l2co._src.update_class import (
    batch_run_,
    batch_run_randomsearch,
    batch_run_sequential_,
)
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
    """Assemble the ``batch_run_*`` arguments both drivers share.

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
    return dict(
        update_class=run_state.update_class,
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
    inputs = _driver_inputs(step, sampler, rng)

    fused = batch_run_(**inputs)
    sequential = batch_run_sequential_(**inputs)

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
    inputs = _driver_inputs(step, sampler, rng)
    batched = batch_run_sequential_(**inputs)

    for r in range(N_REALIZATIONS):
        single = batch_run_sequential_(
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


def test_marker_is_off_for_plain_optimizers():
    assert UpdateClass.sequential_realizations is False


def test_marker_is_on_for_the_meta_optimizer_bridge():
    # l2co's own two-stage strategy dispatches on a per-realization
    # sub-optimizer index; rl2co asserts the same for its bridge in
    # its own suite (l2co may not import rl2co).
    assert L2COUpdateClass.sequential_realizations is True


def test_marker_is_not_a_dataclass_field():
    """A ``ClassVar`` marker must not widen any constructor signature.

    ``dataclasses.fields`` is the check that matters:
    ``__dataclass_fields__`` also carries ``ClassVar`` pseudo-entries, so
    it would not distinguish a marker from a real field.
    """
    names = {f.name for f in dataclasses.fields(UpdateClass)}
    assert "sequential_realizations" not in names
    assert (
        "sequential_realizations"
        not in inspect.signature(UpdateClass.__init__).parameters
    )


@pytest.mark.parametrize(
    ("marker", "is_random_search", "expected"),
    [
        (False, False, batch_run_),
        (True, False, batch_run_sequential_),
        (False, True, batch_run_randomsearch),
        # Random search wins over the marker: it has its own driver for a
        # different reason (peak memory), and that one is also sequential.
        (True, True, batch_run_randomsearch),
    ],
    ids=["plain", "meta", "randomsearch", "randomsearch-and-marker"],
)
def test_batch_run_routes_on_the_marker(
    monkeypatch, marker, is_random_search, expected
):
    """``batch_run`` picks the driver, and this is the rule it uses."""
    called = {}

    def _spy(name):
        def _fn(**kwargs):
            called["driver"] = name
            n = jax.tree.leaves(kwargs["params"])[0].shape[0]
            zeros = jnp.zeros((n,))
            return (None, None, zeros, None, None, None)

        return _fn

    for fn in (batch_run_, batch_run_sequential_, batch_run_randomsearch):
        monkeypatch.setattr(
            "l2co._src.run_state." + fn.__name__, _spy(fn), raising=True
        )

    # ``RandomSearchUpdateClass`` carries its own constructor, so the
    # probe's kwargs follow whichever base it subclasses.
    if is_random_search:

        class _Probe(RandomSearchUpdateClass):
            sequential_realizations = marker

        probe = _Probe(
            sampling_fn=lambda *a, **k: None,
            static_model=None,
            loss_fn=lambda *a, **k: None,
            pass_rng=False,
            bounded=(None, None),
            popsize=1,
            hash=0,
        )
    else:

        class _Probe(UpdateClass):
            sequential_realizations = marker

        probe = _Probe(
            init_fn=lambda *a, **k: None,
            step_fn=lambda *a, **k: None,
            popsize=1,
            hash=0,
        )

    class _RunState:
        update_class = probe
        opt_state = jnp.zeros((N_REALIZATIONS,))
        params = jnp.zeros((N_REALIZATIONS, 1, 4))
        best_params = params
        best_loss = jnp.zeros((N_REALIZATIONS,))

    batch_run(
        run_state=_RunState(),
        batch_state=None,
        dataset={},
        n_iterations=1,
        key=jr.split(jr.key(0), N_REALIZATIONS),
        verbose=False,
    )
    assert called["driver"] == expected
