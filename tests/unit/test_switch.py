"""The shared switch sequence (``docs/adr/0022``).

``handshake_incoming`` is the one place that decides what runs when
control passes to a different optimizer. These tests drive it with toy
primitives that record what they were handed, so they pin the
*sequence* -- what is gated on news, what order things run in, what the
writer is told -- independently of any real optimizer.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from l2co_optimizers import (
    CONF_ABSENT,
    CONF_EXACT,
    OptimizationStep,
    Shape,
    TransferBundle,
)

# Local
from l2co.optimizers import (
    OwnState,
    handshake_incoming,
    has_news,
    is_stale,
    optstep_to_subopt,
    own_distance,
    read_own_state,
)

from .toy_problems import sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Markers each primitive leaves in the toy state, so a test can read
#: back which ran and in which order.
RESET, WRITE, ASK, LAYOUT, RESUME = 1.0, 10.0, 100.0, 1000.0, 10000.0
RESTART, NEUTRAL = 1e5, 1e6


def _bundle() -> TransferBundle:
    return TransferBundle(
        x=jnp.zeros(3),
        f=jnp.asarray(1.0),
        sigma=jnp.asarray(0.5),
        conf_sigma=jnp.asarray(CONF_EXACT),
    )


def _primitives(seen: dict):
    """Toy primitives over a state ``{"log": scalar, "order": scalar}``.

    ``log`` sums the markers of what ran. ``order`` is shifted left one
    decimal place per step, so it records the sequence. ``seen`` captures
    the bundle the writer was handed.
    """

    def mark(state, marker):
        return {
            "log": state["log"] + marker,
            "order": state["order"] * 10 + jnp.log10(marker) + 1,
        }

    def reset(state):
        return mark(state, RESET)

    def write(bundle, state):
        seen["bundle"] = bundle
        return mark(state, WRITE)

    def own_ask(state):
        state = mark(state, ASK)
        return state["log"], state

    def layout(state):
        state = mark(state, LAYOUT)
        return state["log"], state

    def resume(state):
        state = mark(state, RESUME)
        return state["log"], state

    return reset, write, own_ask, layout, resume


def _state():
    return {"log": jnp.asarray(0.0), "order": jnp.asarray(0.0)}


class TestHasNews:
    @pytest.mark.parametrize(
        ("bundle_f", "own_best", "has_run", "expected"),
        [
            (1.0, 2.0, True, True),  # a better point
            (2.0, 2.0, True, False),  # it holds the best already
            (2.0, 2.0, False, True),  # it has never run
            (1.0, 2.0, False, True),
        ],
    )
    def test_truth_table(self, bundle_f, own_best, has_run, expected):
        got = has_news(
            jnp.asarray(bundle_f), jnp.asarray(own_best), jnp.asarray(has_run)
        )
        assert bool(got) is expected


class TestSequence:
    def test_news_runs_reset_then_write_then_layout(self):
        seen = {}
        reset, write, _, layout, resume = _primitives(seen)
        _, state = handshake_incoming(
            _state(),
            _bundle(),
            news=True,
            reset=reset,
            write=write,
            layout=layout,
            resume=resume,
        )
        assert float(state["log"]) == RESET + WRITE + LAYOUT
        # reset (1), write (2), layout (4), read left to right.
        assert float(state["order"]) == pytest.approx(124)

    def test_own_ask_replaces_the_layout_and_sees_the_write(self):
        seen = {}
        reset, write, own_ask, layout, resume = _primitives(seen)
        population, state = handshake_incoming(
            _state(),
            _bundle(),
            news=True,
            reset=reset,
            write=write,
            own_ask=own_ask,
            layout=layout,
            resume=resume,
        )
        assert float(state["log"]) == RESET + WRITE + ASK
        # The ask draws from the state the write produced.
        assert float(population) == RESET + WRITE + ASK

    def test_no_news_skips_reset_and_resumes(self):
        seen = {}
        reset, write, own_ask, layout, resume = _primitives(seen)
        _, state = handshake_incoming(
            _state(),
            _bundle(),
            news=False,
            reset=reset,
            write=write,
            own_ask=own_ask,
            layout=layout,
            resume=resume,
        )
        # The write still runs; the writer is told nothing moved and
        # skips its location half itself.
        assert float(state["log"]) == WRITE + RESUME
        assert not bool(seen["bundle"].moves_iterate)

    def test_traced_news_selects_the_same_branches(self):
        seen = {}
        reset, write, _, layout, resume = _primitives(seen)
        for news, expected in (
            (jnp.asarray(True), RESET + WRITE + LAYOUT),
            (jnp.asarray(False), WRITE + RESUME),
        ):
            _, state = handshake_incoming(
                _state(),
                _bundle(),
                news=news,
                reset=reset,
                write=write,
                layout=layout,
                resume=resume,
            )
            assert float(state["log"]) == expected

    def test_absent_primitives_trace_away(self):
        _, _, _, layout, resume = _primitives({})
        _, state = handshake_incoming(
            _state(), _bundle(), news=True, layout=layout, resume=resume
        )
        assert float(state["log"]) == LAYOUT


class TestScale:
    def test_scale_is_withheld_by_default(self):
        seen = {}
        _, write, _, layout, resume = _primitives(seen)
        handshake_incoming(
            _state(),
            _bundle(),
            news=True,
            write=write,
            layout=layout,
            resume=resume,
        )
        assert float(seen["bundle"].conf_sigma) == CONF_ABSENT
        # The location still crosses.
        assert float(seen["bundle"].f) == 1.0
        assert bool(seen["bundle"].moves_iterate)

    def test_scale_crosses_when_switched_on(self):
        seen = {}
        _, write, _, layout, resume = _primitives(seen)
        handshake_incoming(
            _state(),
            _bundle(),
            news=True,
            write=write,
            layout=layout,
            resume=resume,
            transfer_scale=True,
        )
        assert float(seen["bundle"].conf_sigma) == CONF_EXACT
        assert float(seen["bundle"].sigma) == 0.5


DIM = 3


def _own(
    *,
    has_run=True,
    readable=True,
    has_shape=False,
    evidence=(5.0, 5.0),
    broken=False,
):
    """An optimizer at the origin with scale 1 and a neutral shape."""
    return OwnState(
        point=jnp.zeros(DIM),
        shape=Shape.neutral(DIM, jnp.asarray(1.0)),
        has_shape=has_shape,
        scale_readable=jnp.asarray(readable),
        evidence=jnp.asarray(evidence),
        has_run=jnp.asarray(has_run),
        broken=jnp.asarray(broken),
    )


NEAR = jnp.full(DIM, 0.5)  # half a width away
FAR = jnp.full(DIM, 10.0)  # ten widths away


class TestIsStale:
    def test_near_and_known_is_not_stale(self):
        assert not bool(is_stale(_own(), NEAR))

    def test_far_is_stale(self):
        assert bool(is_stale(_own(), FAR))

    def test_the_threshold_is_in_own_widths(self):
        np.testing.assert_allclose(float(own_distance(_own(), FAR)), 10.0)
        assert not bool(is_stale(_own(), FAR, distance_threshold=11.0))

    def test_never_run_is_stale(self):
        assert bool(is_stale(_own(has_run=False), NEAR))

    def test_no_readable_scale_is_stale(self):
        own = _own(readable=False)
        assert bool(is_stale(own, NEAR))
        assert float(own_distance(own, NEAR)) == float("inf")

    def test_a_shape_needs_one_adaptation_time(self):
        assert bool(is_stale(_own(has_shape=True, evidence=(0.5, 0)), NEAR))
        assert not bool(
            is_stale(_own(has_shape=True, evidence=(1.0, 0)), NEAR)
        )

    def test_no_shape_ignores_evidence(self):
        assert not bool(is_stale(_own(evidence=(0.0, 0.0)), NEAR))


def _gated(**kwargs):
    seen = {}
    reset, write, own_ask, layout, resume = _primitives(seen)

    def mark(state, marker):
        return {
            "log": state["log"] + marker,
            "order": state["order"] * 10 + jnp.log10(marker) + 1,
        }

    defaults = dict(
        news=True,
        write=write,
        layout=layout,
        resume=resume,
        restart=lambda s: mark(s, RESTART),
        neutralize=lambda s: mark(s, NEUTRAL),
        gate_own_state=True,
    )
    defaults.update(kwargs)
    bundle = _bundle_at(defaults.pop("destination", FAR))
    _, state = handshake_incoming(_state(), bundle, **defaults)
    return float(state["log"])


def _bundle_at(x):
    return TransferBundle(
        x=x,
        f=jnp.asarray(1.0),
        sigma=jnp.asarray(0.5),
        conf_sigma=jnp.asarray(CONF_EXACT),
    )


class TestGatedSequence:
    def test_stale_restarts_under_arm_3b(self):
        assert _gated(own=_own()) == RESTART + WRITE + LAYOUT

    def test_stale_forgets_the_shape_under_arm_3a(self):
        log = _gated(own=_own(), restart_stale=False)
        assert log == NEUTRAL + WRITE + LAYOUT

    def test_not_stale_keeps_its_state(self):
        assert _gated(own=_own(), destination=NEAR) == WRITE + LAYOUT

    def test_no_news_checks_nothing(self):
        assert _gated(own=_own(), news=False) == WRITE + RESUME

    def test_gate_off_is_step_2(self):
        assert _gated(own=_own(), gate_own_state=False) == WRITE + LAYOUT

    def test_a_reset_row_resets_regardless(self):
        seen = {}
        reset = _primitives(seen)[0]
        assert _gated(own=_own(), reset=reset) == RESET + WRITE + LAYOUT

    def test_arm_3a_leaves_a_shapeless_optimizer_alone(self):
        log = _gated(own=_own(), restart_stale=False, neutralize=None)
        assert log == WRITE + LAYOUT

    def test_the_gate_needs_its_inputs(self):
        with pytest.raises(ValueError, match="own="):
            _gated()
        with pytest.raises(ValueError, match="restart"):
            _gated(own=_own(), restart=None)


class TestBrokenRestart:
    """``restart_broken``: a broken optimizer restarts on every switch."""

    def _run(self, **kwargs):
        defaults = dict(
            gate_own_state=False, restart_broken=True, destination=NEAR
        )
        defaults.update(kwargs)
        return _gated(**defaults)

    def test_broken_restarts_and_arrives(self):
        assert self._run(own=_own(broken=True)) == RESTART + WRITE + LAYOUT

    def test_broken_restarts_without_news_too(self):
        # Resuming a NaN state only produces NaN, so a switch with no
        # news restarts it and lets it arrive.
        log = self._run(own=_own(broken=True), news=False)
        assert log == RESTART + WRITE + LAYOUT

    def test_a_sound_optimizer_is_step_2(self):
        assert self._run(own=_own()) == WRITE + LAYOUT
        assert self._run(own=_own(), news=False) == WRITE + RESUME

    def test_off_keeps_a_broken_state(self):
        log = self._run(own=_own(broken=True), restart_broken=False)
        assert log == WRITE + LAYOUT

    def test_a_reset_row_resets_a_broken_state_without_news(self):
        reset = _primitives({})[0]
        log = self._run(own=_own(broken=True), news=False, reset=reset)
        assert log == RESET + WRITE + LAYOUT

    def test_with_the_gate_it_restarts_once(self):
        # Broken and far: restarted for being broken, not neutralised
        # on top by arm 3a.
        log = self._run(
            own=_own(broken=True),
            gate_own_state=True,
            restart_stale=False,
            destination=FAR,
        )
        assert log == RESTART + WRITE + LAYOUT

    def test_it_needs_its_inputs(self):
        with pytest.raises(ValueError, match="own="):
            self._run()
        with pytest.raises(ValueError, match="restart"):
            self._run(own=_own(), restart=None)


class TestNoHandOver:
    def test_only_resumes(self):
        # News, a broken state and the gate all on: with hand_over off
        # nothing of the sequence runs.
        reset = _primitives({})[0]
        for kwargs in (
            {},
            {"reset": reset},
            {"restart_broken": True, "own": _own(broken=True)},
            {"own": _own()},
        ):
            assert _gated(hand_over=False, **kwargs) == RESUME, kwargs


class TestReadOwnState:
    @pytest.fixture(scope="class")
    def problem(self):
        return sphere_problem(DIM)

    def _sub(self, problem, name, **hp):
        return optstep_to_subopt(
            OptimizationStep(optimizer=name, hyperparameters=hp),
            model=problem["model"],
            loss_fn=problem["loss_fn"],
            pass_rng=problem["pass_rng"],
        )

    def test_an_es_reads_its_point_and_shape(self, problem):
        sub = self._sub(problem, "sepcmaes")
        state = sub.init_fn(jnp.full(DIM, 0.25), jr.key(0))
        own = read_own_state(
            sub.switch_ports,
            state,
            read_scale=sub.transfer_read_fn,
            evidence=jnp.zeros(2),
            has_run=jnp.asarray(True),
        )
        assert own.has_shape
        np.testing.assert_allclose(np.asarray(own.point), 0.25)
        assert bool(own.scale_readable)
        # Too little evidence: a fresh ES is stale even right where it is.
        assert bool(is_stale(own, jnp.full(DIM, 0.25)))

    def test_a_gradient_optimizer_takes_the_callers_iterate(self, problem):
        sub = self._sub(problem, "adam", learning_rate=1e-2)
        state = sub.init_fn(jnp.zeros(DIM), jr.key(0))
        own = read_own_state(
            sub.switch_ports,
            state,
            read_scale=sub.transfer_read_fn,
            evidence=jnp.zeros(2),
            has_run=jnp.asarray(True),
            point=jnp.full(DIM, 0.75),
        )
        assert not own.has_shape
        np.testing.assert_allclose(np.asarray(own.point), 0.75)
        # Before its first step Adam has no step to report: stale.
        assert not bool(own.scale_readable)

    def test_no_point_and_no_reader_is_an_error(self, problem):
        sub = self._sub(problem, "adam", learning_rate=1e-2)
        state = sub.init_fn(jnp.zeros(DIM), jr.key(0))
        with pytest.raises(ValueError, match="iterate"):
            read_own_state(
                sub.switch_ports,
                state,
                read_scale=sub.transfer_read_fn,
                evidence=jnp.zeros(2),
                has_run=jnp.asarray(True),
            )

    def _own_of(self, sub, state, point=None):
        return read_own_state(
            sub.switch_ports,
            state,
            read_scale=sub.transfer_read_fn,
            evidence=jnp.zeros(2),
            has_run=jnp.asarray(True),
            point=point,
        )

    def test_a_nan_width_is_broken(self, problem):
        sub = self._sub(problem, "crfmnes")
        state = sub.init_fn(jnp.full(DIM, 0.25), jr.key(0))
        assert not bool(self._own_of(sub, state).broken)
        nan = state.replace(std=jnp.asarray(jnp.nan))
        assert bool(self._own_of(sub, nan).broken)
        inf = state.replace(std=jnp.asarray(jnp.inf))
        assert bool(self._own_of(sub, inf).broken)

    def test_a_nan_point_is_broken(self, problem):
        sub = self._sub(problem, "sepcmaes")
        state = sub.init_fn(jnp.full(DIM, 0.25), jr.key(0))
        nan = state.replace(mean=state.mean.at[0].set(jnp.nan))
        assert bool(self._own_of(sub, nan).broken)

    def test_a_gradient_optimizer_is_broken_by_its_iterate(self, problem):
        sub = self._sub(problem, "adam", learning_rate=1e-2)
        state = sub.init_fn(jnp.zeros(DIM), jr.key(0))
        # Before its first step Adam has no step to report. That is not
        # broken: it declares no width yet.
        own = self._own_of(sub, state, point=jnp.zeros(DIM))
        assert not bool(own.scale_readable)
        assert not bool(own.broken)
        nan = jnp.zeros(DIM).at[1].set(jnp.nan)
        assert bool(self._own_of(sub, state, point=nan).broken)
