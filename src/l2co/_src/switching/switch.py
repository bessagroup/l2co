"""The one switch sequence every consumer runs (ADR 0022).

When control passes to a different optimizer, a handshake decides what
the incoming optimizer keeps, what it is told, and which population it
starts from. rl2co's training environment and its two deployment
wrappers each used to write that sequence out themselves, and the copies
drifted: the environment skipped most of it when the switch told the
incoming optimizer nothing new, the wrappers never did, so every policy
was trained on one set of dynamics and evaluated on another (rl2co ADR
0010).

This module holds the sequence once. Its callers still own their data
models -- a ``RunState`` per optimizer in the environment, raw
sub-optimizer states inside a wrapper ``State`` -- so they hand in the
per-optimizer primitives (reset, write, own ask, population layout,
resume) as closures over their own state, and :func:`handshake_incoming`
decides which of them run, in which order.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from l2co_optimizers import TransferBundle

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = ["handshake_incoming", "has_news"]

S = TypeVar("S")
P = TypeVar("P")


def has_news(
    bundle_f: Float[Array, ""],
    own_best: Float[Array, ""],
    has_run: Bool[Array, ""],
) -> Bool[Array, ""]:
    """Whether a switch tells the incoming optimizer anything new.

    It does when the handed-over point is better than anything the
    incoming optimizer has found itself, or when it has never run. The
    second half is not a detail: every member of a menu is handed the
    same initial sample, so the wide-population members hold the global
    best before they have earned it, and holding the best point in a
    population is not the same as being centred on it (rl2co ADR 0009).

    The comparison is exact rather than approximate. ``bundle_f`` is the
    run's best loss, a ``nanmin`` over every member's own best, so it
    equals ``own_best`` exactly when this member holds the run's best.

    Parameters
    ----------
    bundle_f : Float[Array, ""]
        Loss at the handed-over point (``TransferBundle.f``).
    own_best : Float[Array, ""]
        The incoming optimizer's own best loss so far.
    has_run : Bool[Array, ""]
        Whether the incoming optimizer has produced a generation since
        the episode (or run) began.

    Returns
    -------
    Bool[Array, ""]
        ``True`` when the switch carries news.
    """
    return jnp.logical_or(bundle_f < own_best, jnp.logical_not(has_run))


def handshake_incoming(
    state: S,
    bundle: TransferBundle,
    *,
    news: Bool[Array, ""] | bool,
    layout: Callable[[S], tuple[P, S]],
    resume: Callable[[S], tuple[P, S]],
    reset: Callable[[S], S] | None = None,
    write: Callable[[TransferBundle, S], S] | None = None,
    own_ask: Callable[[S], tuple[P, S]] | None = None,
    transfer_scale: bool = False,
) -> tuple[P, S]:
    """Hand control to the incoming optimizer.

    Call this only on a switch: the optimizer about to run differs from
    the one that produced the previous generation, and something has run
    (``docs`` of rl2co ADR 0001). The caller resolves every callable for
    *one* optimizer, so ``reset``, ``write`` and ``own_ask`` are known at
    trace time and a ``None`` traces away.

    The sequence:

    1. **Tag the bundle.** ``bundle.moves_iterate`` is set to ``news``,
       so the writer applies the half of its work that repairs a jump
       (re-centring a mean, zeroing a cumulation path, rebasing an elite)
       only when there is a jump to repair. Unless ``transfer_scale`` is
       on, the bundle's scale is also dropped: the outgoing optimizer's
       scale describes what *it* was looking at, and handing it over made
       runs worse (l2co ADR 0022).
    2. **Reset**, on news only. ``reset`` is ``None`` for an optimizer
       tabulated ``continue``.
    3. **Write** the tagged bundle into the incoming optimizer.
    4. **Population.** With news, the optimizer arrives: it draws its own
       population from the state just written where it has an
       ``own_ask``, and is handed ``layout``'s population otherwise.
       Without news it ``resume``s exactly as if no switch had happened.

    Parameters
    ----------
    state : S
        The caller's state, holding the incoming optimizer's state in
        whatever form the caller keeps it.
    bundle : TransferBundle
        What the outgoing side hands over: the best point so far and its
        loss, plus the outgoing optimizer's scale.
    news : Bool[Array, ""] or bool
        Whether the switch carries news (:func:`has_news`). A consumer
        whose incoming optimizer always moves, because every member shares
        one iterate, passes ``True``.
    layout : Callable[[S], tuple[P, S]]
        The population handed to an optimizer that does not draw its own
        (``best`` / ``best_one``), with the state.
    resume : Callable[[S], tuple[P, S]]
        What the optimizer does when the switch carries no news: its
        ordinary next generation, with the state.
    reset : Callable[[S], S] or None, optional
        Re-initialise the incoming optimizer at the handed-over point.
        ``None`` for an optimizer whose state carries over.
    write : Callable[[TransferBundle, S], S] or None, optional
        Fold the bundle into the incoming optimizer's state (its transfer
        writer). ``None`` when it has none.
    own_ask : Callable[[S], tuple[P, S]] or None, optional
        The incoming optimizer's own ``ask``, for an optimizer that draws
        its own post-switch population (l2co ADR 0014). Takes precedence
        over ``layout``.
    transfer_scale : bool, optional
        Keep the outgoing optimizer's scale in the bundle. Defaults to
        ``False``; the switch exists to measure what the scale write did,
        and goes once that is settled (l2co ADR 0022).

    Returns
    -------
    tuple[P, S]
        The incoming optimizer's population, and the updated state.
    """
    news = jnp.asarray(news)
    if not transfer_scale:
        bundle = TransferBundle.absent(bundle.x, bundle.f)
    bundle = eqx.tree_at(lambda b: b.moves_iterate, bundle, news)

    if reset is not None:
        state = jax.lax.cond(news, reset, lambda s: s, state)

    if write is not None:
        state = write(bundle, state)

    arrive = own_ask if own_ask is not None else layout
    return jax.lax.cond(news, arrive, resume, state)
