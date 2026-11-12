"""``switch_handshake``: the three steps a switch applies to one ``SubOpt``.

Reset-or-continue, fold the carried scale in, then produce the
population -- the order rl2co's ``HandshakeClass.handshake`` and its
deployment wrapper apply, pulled out of both into the switching layer
(l2co#120, ADR 0019). Expected values are built from the optimizer's
own ports (``init_fn``, ``transfer_write_fn``, ``step_fn``,
``handshake_population``) composed in the specified order.
"""

#                                                                       Modules
# =============================================================================

# Standard
import dataclasses

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from l2co_tasks import create_bbob_task

# Local
from l2co import OptimizationStep
from l2co.optimizers import (
    CONF_ABSENT,
    CONF_ESTIMATED,
    TransferBundle,
    optstep_to_subopt,
    switch_handshake,
)
from l2co.sampling import normal_sampling

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

KEY_RESET = jr.key(11)
KEY_ASK = jr.key(12)


@pytest.fixture(scope="module")
def task():
    return create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)


def _sub(task, name):
    return optstep_to_subopt(
        OptimizationStep(optimizer=name),
        model=task.model,
        loss_fn=task.loss_fn,
        pass_rng=task.pass_rng,
    )


def _x0(task):
    return jax.tree.map(jnp.asarray, task.model)


def _bundle(task, sigma=0.37, conf=CONF_ESTIMATED, moves=True):
    """A bundle whose ``x`` is displaced from where the state was built."""
    return TransferBundle(
        x=jax.tree.map(lambda a: a + 3.0, _x0(task)),
        f=jnp.asarray(1.0),
        sigma=jnp.asarray(sigma),
        conf_sigma=jnp.asarray(conf),
        moves_iterate=jnp.asarray(moves),
    )


def _equal(a, b) -> bool:
    try:
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    except AssertionError:
        return False
    return True


def _add_sigma(bundle, sub_state):
    """A writer whose effect a later reset would visibly undo."""
    return jax.tree.map(
        lambda a: a + bundle.sigma
        if jnp.issubdtype(a.dtype, jnp.floating)
        else a,
        sub_state,
    )


def _init_at(x, key):
    """An init port whose state is the point it was built at."""
    return x


def test_reset_happens_around_bundle_x_before_the_scale_write(task):
    # lbfgs resets, but its real state hides ``x`` and it writes nothing
    # (``write_none``); swap in ports that make both steps visible.
    sub = dataclasses.replace(
        _sub(task, "lbfgs"), init_fn=_init_at, transfer_write_fn=_add_sigma
    )
    assert sub.opt_state_handshake == "reset"
    bundle = _bundle(task)

    new_state, _ = switch_handshake(
        sub, _x0(task), bundle, normal_sampling, KEY_RESET, KEY_ASK
    )

    # Reset to bundle.x (= x0 + 3), then + sigma: a write before the
    # reset would leave bundle.x, no reset would leave x0 + sigma.
    expected = jax.tree.map(lambda a: a + 3.0 + 0.37, _x0(task))
    np.testing.assert_allclose(
        np.asarray(jax.tree.leaves(new_state)[0]),
        np.asarray(jax.tree.leaves(expected)[0]),
        rtol=1e-6,
    )


def test_own_ask_draws_from_the_written_state(task):
    sub = _sub(task, "sepcmaes")
    assert sub.own_ask and sub.opt_state_handshake == "continue"
    state = sub.init_fn(_x0(task), jr.key(0))
    bundle = _bundle(task)

    new_state, rows = switch_handshake(
        sub, state, bundle, normal_sampling, KEY_RESET, KEY_ASK
    )

    written = sub.transfer_write_fn(bundle, state)
    expected_rows, expected_state = sub.step_fn(None, written, KEY_ASK)
    assert jax.tree.leaves(rows)[0].shape[0] == sub.popsize
    assert _equal(rows, expected_rows)
    assert _equal(new_state, expected_state)
    # Asking before the write would draw from the unscaled state.
    assert not _equal(rows, sub.step_fn(None, state, KEY_ASK)[0])


def test_without_own_ask_the_population_is_the_layout_at_bundle_x(task):
    sub = _sub(task, "simple_ga")
    assert not sub.own_ask and sub.population_handshake == "best_one"
    state = sub.init_fn(_x0(task), jr.key(0))
    bundle = _bundle(task)

    _, rows = switch_handshake(
        sub, state, bundle, normal_sampling, KEY_RESET, KEY_ASK
    )

    leaf = jax.tree.leaves(rows)[0]
    assert leaf.shape[0] == sub.popsize
    # Canonical ``best_one``: sampler draws, bundle.x in the last slot.
    np.testing.assert_array_equal(
        np.asarray(leaf[-1]), np.asarray(jax.tree.leaves(bundle.x)[0])
    )
    assert _equal(
        rows, sub.handshake_population(bundle.x, normal_sampling, KEY_ASK)
    )


def test_a_switch_that_does_not_move_writes_the_scale_only(task):
    # sepcmaes asks for itself; force a reset policy so all three steps
    # are live and only ``moves_iterate`` can switch two of them off.
    sub = dataclasses.replace(
        _sub(task, "sepcmaes"), opt_state_handshake="reset"
    )
    state = sub.init_fn(_x0(task), jr.key(0))
    held = jax.tree.map(
        lambda a: jnp.broadcast_to(a, (sub.popsize, *a.shape)) - 1.0,
        _x0(task),
    )
    bundle = _bundle(task, moves=False)

    new_state, rows = switch_handshake(
        sub,
        state,
        bundle,
        normal_sampling,
        KEY_RESET,
        KEY_ASK,
        held_population=held,
    )

    # No reset to bundle.x, no own ask: only the scale write lands, and
    # the population is the one the optimizer already holds.
    assert _equal(new_state, sub.transfer_write_fn(bundle, state))
    assert not _equal(
        new_state,
        sub.transfer_write_fn(bundle, sub.init_fn(bundle.x, KEY_RESET)),
    )
    assert _equal(rows, held)


def test_an_absent_scale_leaves_a_continue_state_untouched(task):
    sub = _sub(task, "simple_ga")
    assert sub.opt_state_handshake == "continue" and not sub.own_ask
    state = sub.init_fn(_x0(task), jr.key(0))
    bundle = _bundle(task, sigma=0.0, conf=CONF_ABSENT)

    new_state, _ = switch_handshake(
        sub, state, bundle, normal_sampling, KEY_RESET, KEY_ASK
    )

    assert _equal(new_state, state)
