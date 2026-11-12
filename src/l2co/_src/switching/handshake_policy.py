"""Which handshake an optimizer needs when control passes to it.

A *handshake* is the state transfer applied when the active optimizer
changes: which population the incoming optimizer sees, and whether its
internal state is reset or continued. Which variant a given optimizer
needs is a property of the **optimizer** -- can its accumulated internal
state survive an iterate jump? -- not of the loop doing the switching.
This module holds that table, and only that table.

It still lives with the switching layer rather than with the optimizers
(ADR 0019): a handshake only exists where something switches, and an
optimizer absent from the table gets :data:`DEFAULT_HANDSHAKE_POLICY`,
so a new bare optimizer works without an entry. What the optimizer
itself contributes to a switch -- its transfer ports and whether it
draws its own post-switch population -- stays in ``l2co_optimizers``.

The **population** half's transfer functions live here too, and are the
single canonical implementation. They used to be duplicated per consumer
with divergent slot layouts -- one placing the best point in the last
slot and drawing ``popsize`` samples, the other in slot 0 drawing
``popsize - 1``. That is not cosmetic. CR-FM-NES draws
``concat([z, -z])`` (``cr_fm_nes.py:176-177``), so its antithetic
partners sit at ``(i, i + popsize / 2)``, and -- more to the point -- it
*caches* those draws in state and consumes them on the next ``tell``,
pairing ``fitness`` with ``state.z`` index for index (``:201``,
``:207``, ``:235``). Which slot the best point occupies therefore
decides which cached search direction its fitness is credited to, and
the two layouts produce *different* updates from the same handshake. A
policy trained against one layout and deployed against the other
diverges at every switch. (An earlier version of this note attributed
the divergence to adjacent pairs at ``(2i, 2i + 1)`` and to population
geometry; the conclusion held but the mechanism did not.) The deeper
problem -- that substituting *any* foreign population invalidates those
cached draws -- is what ``ADR 0012`` addresses, by regenerating the
population through the incoming optimizer's own ``ask`` instead.

The canonical layout is the environment's (best point
last), chosen because it is what existing trained checkpoints and stored
trajectories were produced with. This amends
``ADR 0009-shared-handshake-policy-table.md``, which recorded the
divergence as accepted.

The **opt_state** half's transfer functions are still per-consumer,
because they genuinely operate on incompatible data models: the rl2co
environment's on ``RunState`` / ``UpdateClass``, the deployment
wrappers' on :class:`~l2co._src.switching.sub_optimizer.SubOpt` and
their own strategy ``State``. Only the *decision* is shared there.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import OptimizationStep, SamplerFunction

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "DEFAULT_HANDSHAKE_POLICY",
    "DEFAULT_TRANSFER_MODE",
    "HANDSHAKE_POLICY",
    "POPULATION_HANDSHAKES",
    "population_best",
    "population_best_one",
    "population_handshake_for",
    "OPT_STATE_VARIANTS",
    "POPULATION_VARIANTS",
    "TRANSFER_MODES",
    "HandshakePolicy",
    "handshake_policy_for",
    "transfer_mode_index",
    "validate_transfer_mode",
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
        internal state is re-initialised, for optimizers whose
        accumulated state cannot survive an iterate jump. Whether the
        point handed to ``init`` matters depends on the optimizer.
        L-BFGS's ``init`` reads its argument's shape and dtype only
        (``optax`` ``transform.py:1719`` stores ``zeros_like(params)``),
        so the values passed do not reach the state. The solvers that
        evaluate the objective themselves -- ``bfgs``, ``dfp``,
        ``nonlinearcg``, ``neldermead``, ``lbfgsb``, ``slsqp`` -- keep
        their iterate *in* the state, so for them the point is
        load-bearing: ``init`` at the wrong point makes the solver's next
        evaluation somewhere the loop did not put it. Every caller
        therefore re-initialises at the point the incoming optimizer
        will step from: the best parameters where the handshake jumps
        the iterate (the rl2co environment, this package's strategy
        wrapper, rl2co's ``evosax_wrapper``), the current iterate where
        it does not (rl2co's ``optax_wrapper``).
    """

    population: str = eqx.field(static=True)
    opt_state: str = eqx.field(static=True)


POPULATION_VARIANTS = ("best", "best_one")
OPT_STATE_VARIANTS = ("reset", "continue")

TRANSFER_MODES = ("full", "none")
"""Switch-level transfer modes (``docs/adr/0028``).

Orthogonal to the per-optimizer :data:`HANDSHAKE_POLICY` table, which
only says *how* a transfer is done. ``"full"`` -- today's handshake:
the table's population and opt-state variants plus the transfer-port
write. ``"none"`` -- a cold start: the incoming optimizer is
initialised exactly as at the start of a run, from the run's sampler
with a fresh key, and nothing of the outgoing run reaches it. A tuple
of string constants, like :data:`POPULATION_VARIANTS`, so a later mode
is an addition rather than a breaking change.

The order is append-only: a mode's position is its integer code
(:func:`transfer_mode_index`), which a traced per-switch argument
carries. New modes go at the end so existing codes never change.
"""

DEFAULT_TRANSFER_MODE = "full"
"""The transfer mode every caller that accepts one defaults to.

``"full"`` is the handshake as it was before transfer modes existed, so
a caller that does not ask for a mode reproduces its old trajectories.
"""

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
optimizers that need their internal state *reset* on a switch -- those
are tabulated explicitly. Add an entry for any new optimizer whose
internal state cannot survive an iterate jump: falling back to
``continue`` is not merely suboptimal for one that keeps its iterate in
its state, it feeds the solver a loss and gradient measured somewhere
other than where it believes it is.
"""

HANDSHAKE_POLICY: dict[str, HandshakePolicy] = {
    "adam": HandshakePolicy("best", "continue"),
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
    # The solvers that evaluate the objective themselves (l2co-optimizers
    # ADR 0011). Each keeps its iterate in its state -- optimistix's
    # ``y`` / ``y_eval``, scipy's Fortran workspace, the Nelder-Mead
    # simplex -- next to state that describes that iterate's
    # neighbourhood: a Hessian estimate, a conjugate direction, a
    # simplex and its losses. A jump leaves both wrong, and only
    # ``init`` at the new point makes them agree with it again.
    "bfgs": HandshakePolicy("best", "reset"),
    "dfp": HandshakePolicy("best", "reset"),
    "nonlinearcg": HandshakePolicy("best", "reset"),
    "neldermead": HandshakePolicy("best", "reset"),
    "lbfgsb": HandshakePolicy("best", "reset"),
    "slsqp": HandshakePolicy("best", "reset"),
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


def validate_transfer_mode(mode: str) -> str:
    """Check that ``mode`` names a switch-level transfer mode.

    Parameters
    ----------
    mode : str
        Candidate mode, spelled exactly as in :data:`TRANSFER_MODES`.

    Returns
    -------
    str
        ``mode`` unchanged, so a caller can validate inline.

    Raises
    ------
    TypeError
        If ``mode`` is not a ``str`` (for example a traced or array
        code); convert a name with :func:`transfer_mode_index` instead.
    ValueError
        If ``mode`` is not one of :data:`TRANSFER_MODES`.
    """
    if not isinstance(mode, str):
        raise TypeError(
            "transfer mode must be a str naming one of TRANSFER_MODES; got "
            f"{type(mode).__name__}. A traced per-switch mode is an int "
            "code: build it from a name with transfer_mode_index."
        )
    if mode not in TRANSFER_MODES:
        valid = ", ".join(repr(m) for m in TRANSFER_MODES)
        raise ValueError(f"transfer mode must be one of {valid}; got {mode!r}")
    return mode


def transfer_mode_index(mode: str) -> int:
    """Integer code of a transfer mode, for a traced per-switch argument.

    The code is the mode's position in :data:`TRANSFER_MODES`, whose
    order is append-only, so codes stay stable when modes are added.

    Parameters
    ----------
    mode : str
        Mode name, spelled exactly as in :data:`TRANSFER_MODES`.

    Returns
    -------
    int
        ``TRANSFER_MODES.index(mode)``.

    Raises
    ------
    TypeError, ValueError
        As :func:`validate_transfer_mode`.
    """
    return TRANSFER_MODES.index(validate_transfer_mode(mode))


# =============================================================================


#                                          Population handshake implementations
# =============================================================================


def population_best(
    best_params: PyTree,
    popsize: int,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> PyTree:
    """Hand the incoming optimizer the best parameters in every slot.

    The ``"best"`` variant. For the unit populations this variant is
    tabulated for (``adam``, ``sgd``, ``lbfgs``) that is a single row,
    identical to what a ``"best_one"`` handshake would produce -- the two
    only diverge above ``popsize == 1``.

    Parameters
    ----------
    best_params : PyTree
        Best parameters seen so far, shaped as one solution (no leading
        population axis).
    popsize : int
        Population size of the incoming optimizer.
    sampler : SamplerFunction
        Unused; accepted so both variants share one signature.
    key : PRNGKeyArray
        Unused; accepted so both variants share one signature.

    Returns
    -------
    PyTree
        ``best_params`` with a leading axis of ``popsize``.
    """
    del sampler, key
    return jax.tree.map(
        lambda b: jnp.broadcast_to(b, (popsize, *jnp.shape(b))), best_params
    )


def population_best_one(
    best_params: PyTree,
    popsize: int,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> PyTree:
    """Hand over a fresh population with the best parameters injected.

    The ``"best_one"`` variant: ``popsize`` draws from ``sampler``, with
    ``best_params`` written into the **last** slot. Note the draws are
    *not* centred on ``best_params`` -- every ``l2co_optimizers``
    sampler uses ``params`` for shape and dtype only (``_sample``), so
    this is a fresh draw from the sampler's own distribution with the
    best point injected, not a local perturbation of it.

    That slot is load-bearing, which is why this implementation is shared
    rather than reproduced per consumer: CR-FM-NES caches its own draws
    in state and credits each candidate's fitness to the cached
    direction at the same index, so moving the best point yields a
    different update from the same handshake. Last is canonical because
    it is what existing trained checkpoints and stored trajectories were
    produced with.

    Parameters
    ----------
    best_params : PyTree
        Best parameters seen so far, shaped as one solution (no leading
        population axis).
    popsize : int
        Population size of the incoming optimizer; also the number of
        draws requested, one of which the injected best point replaces.
    sampler : SamplerFunction
        ``l2co.sampling`` function, called as
        ``sampler(key, best_params, popsize)``.
    key : PRNGKeyArray
        Key for the draws.

    Returns
    -------
    PyTree
        Population with a leading axis of ``popsize``.
    """
    samples = sampler(key, best_params, popsize)
    return jax.tree.map(lambda s, b: s.at[-1].set(b), samples, best_params)


POPULATION_HANDSHAKES: dict[str, Callable[..., PyTree]] = {
    "best": population_best,
    "best_one": population_best_one,
}
"""The ``population`` half's implementations, keyed as in the table.

Keys match :data:`POPULATION_VARIANTS`. All share the signature
``(best_params, popsize, sampler, key) -> PyTree``, returning a
population with a leading axis of ``popsize``. A consumer whose
generation width exceeds ``popsize`` (the strategy wrappers, which
allocate ``max(popsize)`` across the menu) pads the result itself --
padding is the caller's layout concern, not the handshake's.
"""


def population_handshake_for(
    optimizer: OptimizationStep | str,
) -> Callable[..., PyTree]:
    """Resolve the population-handshake implementation for an optimizer.

    Parameters
    ----------
    optimizer : OptimizationStep | str
        Optimizer spec, or its registry name directly.

    Returns
    -------
    Callable[..., PyTree]
        The tabulated variant's implementation, or the default's for
        names absent from :data:`HANDSHAKE_POLICY`.
    """
    return POPULATION_HANDSHAKES[handshake_policy_for(optimizer).population]
