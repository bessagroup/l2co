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
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, PyTree
from l2co_optimizers import CONF_ABSENT, Shape, SwitchPorts, TransferBundle

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "DISTANCE_THRESHOLD",
    "OwnState",
    "handshake_incoming",
    "has_news",
    "is_stale",
    "own_distance",
    "read_own_state",
]

#: How many of its own widths a destination may lie from where an
#: optimizer is before its own state counts as stale (the distance gate,
#: ADR 0022's ``c``). A placeholder: step 3's A/B picks it from
#: ``{1, 2, 3, 5, 10}``.
DISTANCE_THRESHOLD = 3.0

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


class OwnState(eqx.Module):
    """What the incoming optimizer holds now, for the staleness check.

    Read by the caller from its own data model, normally through
    :func:`read_own_state`, because only the caller knows where the
    optimizer's state and iterate live.

    Attributes
    ----------
    point : Float[Array, " d"]
        Where the optimizer is, flattened.
    shape : Shape
        Its own shape, at the width it pairs with; the neutral shape at
        its scale when it holds no shape of its own.
    has_shape : bool
        Static. Whether :attr:`shape` is the optimizer's own. The
        evidence check applies only then (ADR 0022).
    scale_readable : Bool[Array, ""]
        Whether the optimizer could report its scale at all.
    evidence : Float[Array, " 2"]
        Adaptation times behind its diagonal and its direction, counted
        by the caller (updates times the rates the optimizer declares).
    has_run : Bool[Array, ""]
        Whether it has produced a generation since the run began.
    broken : Bool[Array, ""]
        Whether its state has stopped being a number: its point, or a
        width it declares it can report, is NaN or infinite. Such an
        optimizer only produces NaN until it is restarted.
    """

    point: Float[Array, " d"]
    shape: Shape
    has_shape: bool = eqx.field(static=True)
    scale_readable: Bool[Array, ""]
    evidence: Float[Array, " 2"]
    has_run: Bool[Array, ""]
    broken: Bool[Array, ""]


def read_own_state(
    switch_ports: SwitchPorts | None,
    opt_state: PyTree,
    *,
    read_scale,
    evidence: Float[Array, " 2"],
    has_run: Bool[Array, ""],
    point: PyTree | None = None,
) -> OwnState:
    """Read what an optimizer holds, through what it declares.

    Parameters
    ----------
    switch_ports : SwitchPorts or None
        The optimizer's switch ports (``SubOpt.switch_ports``,
        ``UpdateClass.switch_ports``).
    opt_state : PyTree
        Its own state.
    read_scale : Callable
        Its scale reader, ``read_scale(opt_state) -> (sigma, conf)``.
        Used when it holds no shape of its own.
    evidence : Float[Array, " 2"]
        The caller's evidence counters for it.
    has_run : Bool[Array, ""]
        Whether it has run.
    point : PyTree or None, optional
        Its point, for an optimizer whose point the caller holds (a
        gradient optimizer's iterate). Ignored when it declares a point
        reader.

    Returns
    -------
    OwnState
        The view :func:`is_stale` and ``restart_broken`` read. It is
        *broken* when its point is not finite, or when it reports a
        width (it has a shape, or a scale reader that is not absent) and
        that width is not finite. An optimizer that has no width to
        report is not broken for lacking one.

    Raises
    ------
    ValueError
        If the optimizer declares no point reader and no ``point`` is
        given.
    """
    if switch_ports is not None and switch_ports.read_point is not None:
        flat_point = switch_ports.read_point(opt_state)
    elif point is not None:
        flat_point = ravel_pytree(point)[0]
    else:
        raise ValueError(
            "the optimizer declares no point; pass the iterate the caller "
            "holds for it"
        )
    flat_point = jnp.asarray(flat_point, dtype=float)

    if switch_ports is not None and switch_ports.read_shape is not None:
        shape = switch_ports.read_shape(opt_state)
        readable = jnp.isfinite(shape.scale) & (shape.scale > 0.0)
        width_broken = jnp.logical_not(jnp.isfinite(shape.scale))
        has_shape = True
    else:
        sigma, conf = read_scale(opt_state)
        shape = Shape.neutral(flat_point.shape[0], sigma)
        readable = (conf > CONF_ABSENT) & jnp.isfinite(sigma) & (sigma > 0)
        width_broken = (conf > CONF_ABSENT) & jnp.logical_not(
            jnp.isfinite(sigma)
        )
        has_shape = False
    point_broken = jnp.logical_not(jnp.all(jnp.isfinite(flat_point)))
    return OwnState(
        point=flat_point,
        shape=shape,
        has_shape=has_shape,
        scale_readable=readable,
        evidence=jnp.asarray(evidence, dtype=float),
        has_run=jnp.asarray(has_run),
        broken=point_broken | width_broken,
    )


def own_distance(own: OwnState, destination: PyTree) -> Float[Array, ""]:
    """How far ``destination`` lies, in the optimizer's own widths.

    The distance ADR 0022's distance gate reads, and the one rl2co's
    observation reports per menu member. ``inf`` when the optimizer
    could not report a scale.

    Parameters
    ----------
    own : OwnState
        What the optimizer holds.
    destination : PyTree
        Where the switch sends it (``TransferBundle.x``).

    Returns
    -------
    Float[Array, ""]
        Distance in widths.
    """
    offset = ravel_pytree(destination)[0] - own.point
    return jnp.where(own.scale_readable, own.shape.distance(offset), jnp.inf)


def is_stale(
    own: OwnState,
    destination: PyTree,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> Bool[Array, ""]:
    """Whether the optimizer's own state no longer fits at ``destination``.

    Stale when any of these holds (ADR 0022):

    * it has never run, so it has no adaptation to keep;
    * it cannot report a scale, so it cannot show its state still fits;
    * the destination lies more than ``distance_threshold`` of its own
      widths away;
    * it holds a shape of its own with less than one adaptation time of
      evidence behind its diagonal.

    Parameters
    ----------
    own : OwnState
        What the optimizer holds.
    destination : PyTree
        Where the switch sends it.
    distance_threshold : float, optional
        The distance gate, in widths.

    Returns
    -------
    Bool[Array, ""]
        ``True`` when its own state is stale there.
    """
    stale = jnp.logical_not(own.has_run) | jnp.logical_not(own.scale_readable)
    stale = stale | (own_distance(own, destination) > distance_threshold)
    if own.has_shape:
        stale = stale | (own.evidence[0] < 1.0)
    return stale


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
    own: OwnState | None = None,
    restart: Callable[[S], S] | None = None,
    neutralize: Callable[[S], S] | None = None,
    gate_own_state: bool = False,
    restart_stale: bool = True,
    distance_threshold: float = DISTANCE_THRESHOLD,
    restart_broken: bool = False,
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
       tabulated ``continue``. With ``restart_broken`` on, a broken
       optimizer (:attr:`OwnState.broken`) is restarted on *every*
       switch, news or not, and then arrives as on a switch with news:
       a broken state has nothing to keep, and resuming it only
       produces NaN. With ``gate_own_state`` on, a ``continue``
       optimizer whose own state is stale at the destination
       (:func:`is_stale`) is restarted instead (``restart``, arm 3b of
       ADR 0022) or, with ``restart_stale`` off, keeps its scale and has
       its shape made neutral (``neutralize``, arm 3a).
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
    own : OwnState or None, optional
        What the incoming optimizer holds now (:func:`read_own_state`).
        Required with ``gate_own_state``.
    restart : Callable[[S], S] or None, optional
        Restart the incoming optimizer at the handed-over point: its own
        ``init`` there, keeping what its switch ports declare, with the
        caller's evidence counters for it zeroed.
    neutralize : Callable[[S], S] or None, optional
        Forget the incoming optimizer's shape and keep its width
        (``SwitchPorts.neutralize``), zeroing its evidence. ``None`` for
        an optimizer without a shape of its own, which arm 3a leaves
        alone.
    gate_own_state : bool, optional
        Check, on a switch with news, whether the incoming optimizer's
        own state is stale at the destination. Defaults to ``False``
        until step 3's A/B decides; the switch then goes (ADR 0022).
    restart_stale : bool, optional
        What happens to a stale optimizer: restart it (``True``, arm 3b)
        or make its shape neutral and keep its scale (``False``, arm
        3a).
    distance_threshold : float, optional
        The distance gate, in the optimizer's own widths.
    restart_broken : bool, optional
        Restart a broken incoming optimizer (``own.broken``) at the
        handed-over point, with or without news. Independent of
        ``gate_own_state``. Without it the switch keeps a broken
        CR-FM-NES, and every later visit to it is NaN; its A/B found
        that it removes every such visit and changes no result. Off
        here only because it needs ``own`` and a restart, which a caller
        that keeps no per-optimizer state cannot give; rl2co's
        environment and deployment bridge turn it on by default
        (decided 2026-10-16).

    Returns
    -------
    tuple[P, S]
        The incoming optimizer's population, and the updated state.
    """
    news = jnp.asarray(news)
    broken = jnp.asarray(False)
    if restart_broken:
        if own is None:
            raise ValueError(
                "restart_broken needs the incoming optimizer's "
                "own state (own=)"
            )
        if reset is None and restart is None:
            raise ValueError("restart_broken needs a restart")
        broken = own.broken
        # A restarted optimizer arrives: write, then its population.
        news = news | broken
    if not transfer_scale:
        bundle = TransferBundle.absent(bundle.x, bundle.f)
    bundle = eqx.tree_at(lambda b: b.moves_iterate, bundle, news)

    if reset is not None:
        state = jax.lax.cond(news, reset, lambda s: s, state)
    else:
        if restart_broken:
            state = jax.lax.cond(broken, restart, lambda s: s, state)
        if gate_own_state:
            if own is None:
                raise ValueError(
                    "gate_own_state needs the incoming optimizer's "
                    "own state (own=)"
                )
            if restart_stale and restart is None:
                raise ValueError("restart_stale needs a restart")
            action = restart if restart_stale else neutralize
            if action is not None:
                stale = (
                    news
                    & jnp.logical_not(broken)
                    & is_stale(own, bundle.x, distance_threshold)
                )
                state = jax.lax.cond(stale, action, lambda s: s, state)

    if write is not None:
        state = write(bundle, state)

    arrive = own_ask if own_ask is not None else layout
    return jax.lax.cond(news, arrive, resume, state)
