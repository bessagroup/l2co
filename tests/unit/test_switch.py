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
import pytest
from l2co_optimizers import CONF_ABSENT, CONF_EXACT, TransferBundle

# Local
from l2co.optimizers import handshake_incoming, has_news

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Markers each primitive leaves in the toy state, so a test can read
#: back which ran and in which order.
RESET, WRITE, ASK, LAYOUT, RESUME = 1.0, 10.0, 100.0, 1000.0, 10000.0


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
