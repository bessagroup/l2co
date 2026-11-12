"""The three steps a switch applies to the incoming sub-optimizer.

rl2co's training environment (``HandshakeClass.handshake``) and its
deployment wrapper (``_RL2COPopulationStrategy``) both hand control to an
incoming optimizer in the same three steps; l2co's strategy wrapper
applies them too under ``opt_state_handshake="bundle"`` (l2co#120). This
module holds them once, against a :class:`SubOpt` rather than any
caller's run state, so the callers agree by construction (ADR 0019):

1. **Reset or continue** the internal state, as the optimizer's
   :data:`HANDSHAKE_POLICY` entry says. A reset re-initialises around
   ``bundle.x``.
2. **Fold the carried scale in** with the incoming optimizer's
   ``transfer_write_fn``. After the reset, so a fresh state still gets
   the scale.
3. **Produce the population**: the incoming optimizer's own ``ask``
   against the state just written where :attr:`SubOpt.own_ask` is set,
   the canonical layout (:meth:`SubOpt.handshake_population`) otherwise.

Deciding *whether* a switch happens, reading the outgoing optimizer into
the bundle and padding the population to a generation width stay with
the caller.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import jax
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import TransferBundle

# Local
from l2co._src.switching.sub_optimizer import SubOpt

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def switch_handshake(
    sub: SubOpt,
    sub_state: PyTree,
    bundle: TransferBundle,
    sampler: Callable,
    key_reset: PRNGKeyArray,
    key_ask: PRNGKeyArray,
    held_population: PyTree | None = None,
) -> tuple[PyTree, PyTree]:
    """Hand control to ``sub`` across a switch.

    Applies the three steps of the module docstring, in order, to one
    sub-optimizer's state. ``bundle.moves_iterate`` gates the two halves
    that exist because the iterate jumps -- the reset and the
    population -- while the scale write always runs (the writers gate
    their own place-justified parts on the same flag).

    Parameters
    ----------
    sub : SubOpt
        The incoming sub-optimizer. Its static
        :attr:`~SubOpt.opt_state_handshake`,
        :attr:`~SubOpt.transfer_write_fn` and :attr:`~SubOpt.own_ask`
        pick the steps at trace time.
    sub_state : PyTree
        ``sub``'s internal state before the switch.
    bundle : TransferBundle
        What the outgoing optimizer handed over: ``x`` is the point a
        reset re-initialises around and the layout is built at;
        ``sigma`` / ``conf_sigma`` the scale the writer folds in. A
        :data:`~l2co_optimizers.CONF_ABSENT` bundle writes nothing.
    sampler : Callable
        ``l2co.sampling`` function for the layout, called as
        ``sampler(key, bundle.x, popsize)``. Unused where ``own_ask``.
    key_reset : PRNGKeyArray
        Key for ``sub.init_fn`` on a reset.
    key_ask : PRNGKeyArray
        Key for the own ``ask`` or the layout's draws.
    held_population : PyTree or None, optional
        The ``popsize`` rows ``sub`` already holds, returned unchanged
        when ``bundle.moves_iterate`` is ``False``. ``None`` (default)
        is for a caller that only hands over on a move, such as l2co's
        strategy wrapper: the population step then runs unconditionally.

    Returns
    -------
    tuple[PyTree, PyTree]
        ``(sub_state, population)``: the incoming state after the
        handshake and a population with a leading axis of
        ``sub.popsize``. Padding it to a wider generation is the
        caller's concern.
    """
    moves = bundle.moves_iterate
    if sub.opt_state_handshake == "reset":
        sub_state = jax.lax.cond(
            moves,
            lambda s: sub.init_fn(bundle.x, key_reset),
            lambda s: s,
            sub_state,
        )
    if sub.transfer_write_fn is not None:
        sub_state = sub.transfer_write_fn(bundle, sub_state)

    def _arrive(s):
        if sub.own_ask:
            rows, s = sub.step_fn(None, s, key_ask)
            return s, rows
        return s, sub.handshake_population(bundle.x, sampler, key_ask)

    if held_population is None:
        return _arrive(sub_state)
    return jax.lax.cond(
        moves, _arrive, lambda s: (s, held_population), sub_state
    )
