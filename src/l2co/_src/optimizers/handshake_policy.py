"""Which handshake an optimizer needs when control passes to it.

A *handshake* is the state transfer applied when the active optimizer
changes: which population the incoming optimizer sees, and whether its
internal state is reset or continued. Which variant a given optimizer
needs is a property of the **optimizer** -- can its accumulated internal
state survive an iterate jump? -- not of the loop doing the switching.
This module holds that table, and only that table.

The concrete transfer functions are *not* here, deliberately. They come
in two incompatible data models: the rl2co environment's operate on
``RunState`` / ``UpdateClass``, the deployment wrappers' on
:class:`~l2co._src.optimizers.sub_optimizer.SubOpt` and their own
strategy ``State``, and the two population halves have divergent slot
layouts. Only the *decision* is shared, so the two implementations
cannot disagree about which optimizer needs what. See
``docs/adr/0009-shared-handshake-policy-table.md``.
"""

from __future__ import annotations

import equinox as eqx

from ..optimizer_schedule import OptimizationStep

# =============================================================================

__all__ = [
    "DEFAULT_HANDSHAKE_POLICY",
    "HANDSHAKE_POLICY",
    "OPT_STATE_VARIANTS",
    "POPULATION_VARIANTS",
    "HandshakePolicy",
    "handshake_policy_for",
]


class HandshakePolicy(eqx.Module):
    """Which handshake variants one optimizer needs on a switch.

    Two orthogonal halves. Both fields are static: a
    ``HandshakePolicy`` contributes no array leaves to a surrounding
    pytree.

    Attributes
    ----------
    population : str
        ``"best"`` -- the incoming optimizer sees the best parameters
        so far, expanded to its (unit) population. ``"best_one"`` -- it
        sees a freshly sampled population with the best injected.
    opt_state : str
        ``"continue"`` -- the optimizer's internal state carries over,
        so switching back resumes its own adaptation. ``"reset"`` -- the
        internal state is re-initialised around the best parameters,
        for optimizers whose accumulated state cannot survive an
        iterate jump.
    """

    population: str = eqx.field(static=True)
    opt_state: str = eqx.field(static=True)


POPULATION_VARIANTS = ("best", "best_one")
OPT_STATE_VARIANTS = ("reset", "continue")

DEFAULT_HANDSHAKE_POLICY = HandshakePolicy(
    population="best_one", opt_state="continue"
)
"""Fallback for optimizers absent from :data:`HANDSHAKE_POLICY`.

The optimizer registry is extensible (``register_optimizer``), so any
consumer may be handed an optimizer this table has no entry for. Rather
than raising, those fall back to ``best_one`` + ``continue``.

This pair is the conservative choice: ``best_one`` degenerates to
``best`` whenever ``popsize == 1`` (the default for gradient-based
optimizers), so the fallback matches the tabulated behaviour for both
gradient-based and population-based entries. It differs only for
optimizers that need their internal state *reset* on a switch --
``lbfgs`` is the one known case, and it is tabulated explicitly. Add an
entry for any new optimizer whose internal state cannot survive an
iterate jump.
"""

HANDSHAKE_POLICY: dict[str, HandshakePolicy] = {
    "adam": HandshakePolicy("best", "continue"),
    "adam_variable_updates": HandshakePolicy("best", "continue"),
    "sgd": HandshakePolicy("best", "continue"),
    "sepcmaes": HandshakePolicy("best_one", "continue"),
    "cmaes": HandshakePolicy("best_one", "continue"),
    "pso": HandshakePolicy("best_one", "continue"),
    "shade": HandshakePolicy("best_one", "continue"),
    # L-BFGS carries the best candidate but must wipe its curvature
    # memory and line-search state: both describe the neighbourhood of
    # the iterate it was building them at, and the handshake jumps the
    # iterate somewhere else.
    "lbfgs": HandshakePolicy("best", "reset"),
}
"""Handshake policy per optimizer name, keyed as in the registry."""


def handshake_policy_for(
    optimizer: OptimizationStep | str,
) -> HandshakePolicy:
    """Resolve the handshake policy for an optimizer.

    Parameters
    ----------
    optimizer : OptimizationStep | str
        Optimizer spec, or its registry name directly.

    Returns
    -------
    HandshakePolicy
        The tabulated policy, or :data:`DEFAULT_HANDSHAKE_POLICY` for
        names absent from :data:`HANDSHAKE_POLICY`.
    """
    name = (
        optimizer.optimizer
        if isinstance(optimizer, OptimizationStep)
        else optimizer
    )
    return HANDSHAKE_POLICY.get(name, DEFAULT_HANDSHAKE_POLICY)


# =============================================================================
