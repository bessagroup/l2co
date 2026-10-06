"""Sub-optimizer adapters shared by the strategy wrappers.

A :class:`SubOpt` is a thin static adapter that hides whether a
sub-optimizer is gradient-based (an optax ``GradientTransformation``)
or population-based (an evosax-style ``(init, ask, tell)`` triple).
Wrappers that expose a selection model as a duck-typed evosax
strategy — l2co's own :mod:`l2co._src.strategy_wrapper` and rl2co's
policy-deployment wrappers (``rl2co._src.optax_wrapper`` /
``rl2co._src.evosax_wrapper``, which import this module through a
shim) — dispatch across a tuple of ``SubOpt`` adapters via
:func:`jax.lax.switch`, reading the static ``is_pop`` flag at Python
trace time to pick which branch implementation to build for each
slot.

Two factory functions build ``SubOpt`` instances directly
(:func:`grad_sub_optimizer`, :func:`pop_sub_optimizer`); a third,
:func:`optstep_to_subopt`, builds one from an
:class:`~l2co_optimizers.OptimizationStep`. A ``SubOpt`` is an
optimizer's *parts* plus the switching decision about it: the parts --
which library object the name resolves to, its popsize, family,
transfer ports and feval bound -- come from
:func:`l2co_optimizers.optimizer_parts`; the handshake it gets comes
from :mod:`l2co._src.switching.handshake_policy` (ADR 0019).

The module was authored in rl2co and promoted to l2co because both
packages' wrappers need it (ADR 0003); it spent ADRs 0016--0018 in
``l2co-optimizers`` and came back with the rest of the switching layer
in ADR 0019.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Third-party
import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree
from l2co_optimizers import (
    FAMILY_GRADIENT,
    GradientParts,
    LossFunction,
    OptimizationStep,
    optimizer_parts,
    step_fevals,
)

# Local
from l2co._src.switching.handshake_policy import (
    POPULATION_HANDSHAKES,
    handshake_policy_for,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "SubOpt",
    "grad_sub_optimizer",
    "optstep_to_subopt",
    "pop_sub_optimizer",
]


#                                                       Sub-optimizer adapter
# =============================================================================


class SubOpt(eqx.Module):
    """Internal adapter for one sub-optimizer.

    Every entry of a wrapper's sub-optimizer tuple is a ``SubOpt``;
    the wrapper dispatches across them via :func:`jax.lax.switch` and
    reads the static ``is_pop`` flag at Python trace time to decide
    which branch implementation to build for each slot. All fields
    are static — a ``SubOpt`` contributes no array leaves to the
    surrounding pytree.

    Build via :func:`grad_sub_optimizer` or :func:`pop_sub_optimizer`,
    or have the wrapper factories build them automatically from
    :class:`~l2co_optimizers.OptimizationStep`
    entries via :func:`optstep_to_subopt`.

    Attributes
    ----------
    is_pop : bool
        ``True`` for population-based (evosax-style) adapters,
        ``False`` for gradient-based (optax-style) adapters. Read at
        Python trace time to pick the ``lax.switch`` branch
        implementation for each sub-optimizer slot.
    popsize : int
        Population size. ``1`` for gradient-based.
    name_hash : int
        Identifier written into ``HistoryState.update_step`` so the
        selection model can tell which optimizer produced each entry.
    init_fn : Callable
        ``init_fn(params, key) -> sub_state``.
    step_fn : Callable
        Gradient-based: ``(grads, params, sub_state, key, **extra_args)
        -> (update, new_sub_state)`` — returns the optax-style negated
        update. ``**extra_args`` are forwarded verbatim to the optax
        transform's ``update``; line-search transforms (e.g.
        ``optax.lbfgs``) require ``value`` / ``grad`` / ``value_fn``
        there, while plain transforms (Adam, SGD, …) ignore them.
        Population-based: ``(params, sub_state, key) -> (population,
        new_sub_state)`` — i.e. an ``ask`` call returning a fresh
        population of shape ``(popsize, *params_shape)``.
    tell_fn : Callable | None
        Population-based only: ``(sub_state, population, fitnesses,
        key) -> new_sub_state``. ``None`` for gradient-based.
    opt_state_handshake : str
        ``"continue"`` (default) or ``"reset"`` — whether this
        sub-optimizer's internal state survives a handshake, or must be
        re-initialised around the best parameters because it cannot.
        Resolved from :data:`~l2co._src.switching.handshake_policy.
        HANDSHAKE_POLICY` by :func:`optstep_to_subopt`; read at Python
        trace time by the strategy wrappers, so a ``'continue'``
        sub-optimizer's reset branch is never traced. See
        ``ADR 0009-shared-handshake-policy-table.md``.
    population_handshake : str
        ``"best_one"`` (default) or ``"best"`` -- which population the
        incoming optimizer sees. Resolved from the same shared table as
        :attr:`opt_state_handshake` by :func:`optstep_to_subopt`. Apply it
        through :meth:`handshake_population`, never by reimplementing the
        layout: the slot the best point lands in changes the update a
        mirrored-sampling algorithm computes.
    family : str
        Which optimizer family this is, one of
        :data:`~l2co_optimizers.FAMILIES`. Says what
        kind of search state this sub-optimizer holds, and nothing more
        -- the population half of a switch is decided by
        :attr:`own_ask`.
    own_ask : bool
        Whether this sub-optimizer regenerates its own post-switch
        population. Read at Python trace time by the strategy wrappers:
        where it is set they call :attr:`step_fn` (which *is* the
        ``ask`` on the population path) against the state a handshake
        just wrote, rather than handing over a layout. That is what
        keeps an optimizer's next update consistent with the population
        it is told about -- the ask/tell-coupled evosax algorithms cache
        their draws in state (``ADR 0012``), and the mutation GAs
        score the next generation against a baseline the writer just
        repaired (``ADR 0014``). Resolved from the same
        ``l2co_optimizers`` transfer-spec
        table the ``UpdateClass`` path reads, so the environment and
        both deployment wrappers agree.
    transfer_read_fn : Callable | None
        ``transfer_read_fn(sub_state) -> (sigma, conf)`` -- projects this
        sub-optimizer's own state onto the scale a switch carries.
        Resolved from the same
        :func:`~l2co_optimizers.build_transfer_fns`
        the ``UpdateClass`` path uses, so the environment and both
        deployment wrappers agree (``ADR 0009``'s constraint,
        applied to ``ADR 0013``'s channel).
    transfer_write_fn : Callable | None
        ``transfer_write_fn(bundle, sub_state) -> sub_state`` -- folds a
        carried scale into this sub-optimizer's own state.
    fevals_fn : Callable | None
        ``fevals_fn(sub_state_before, sub_state_after) -> Array`` — the
        function evaluations one step of this sub-optimizer actually
        spent. ``None`` (the default, and the case for every
        population-based adapter) means the step costs exactly
        :attr:`popsize`. :func:`grad_sub_optimizer` wires
        :func:`~l2co_optimizers.step_fevals`
        in here, which is what makes a *line-search* transform bill its true
        trial count instead of a flat ``1``. Read it through
        :meth:`bill_fevals`.
    """

    is_pop: bool = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    name_hash: int = eqx.field(static=True)
    init_fn: Callable = eqx.field(static=True)
    step_fn: Callable = eqx.field(static=True)
    tell_fn: Callable | None = eqx.field(static=True)
    opt_state_handshake: str = eqx.field(static=True, default="continue")
    fevals_fn: Callable | None = eqx.field(static=True, default=None)
    population_handshake: str = eqx.field(static=True, default="best_one")
    max_fevals_per_step: int | None = eqx.field(static=True, default=None)
    family: str = eqx.field(static=True, default=FAMILY_GRADIENT)
    transfer_read_fn: Callable | None = eqx.field(static=True, default=None)
    transfer_write_fn: Callable | None = eqx.field(static=True, default=None)
    own_ask: bool = eqx.field(static=True, default=False)

    @property
    def fevals_bound(self) -> int:
        """Worst-case function evaluations one step of this can spend.

        :attr:`max_fevals_per_step` when set, else :attr:`popsize`. The two
        differ for line-search transforms, which evaluate the objective
        once per trial against a population of one.

        This is the *bound*, for sizing and normalising; the actual cost of
        a given step comes from :meth:`bill_fevals`. Consumers that
        normalise a per-evaluation quantity (rl2co's ``AUCEnv`` reward and
        its feval-progress observation channel) must use this rather than
        ``popsize``, or they under-estimate the worst case and let a
        normalised quantity leave its intended range.

        Returns
        -------
        int
            Positive upper bound on one step's evaluations.
        """
        if self.max_fevals_per_step is None:
            return self.popsize
        return self.max_fevals_per_step

    def handshake_population(
        self,
        best_params: PyTree,
        sampler: Callable,
        key: PRNGKeyArray,
    ) -> PyTree:
        """Build the population this sub-optimizer receives on a switch.

        Dispatches on :attr:`population_handshake` into
        :data:`~l2co._src.switching.handshake_policy.
        POPULATION_HANDSHAKES`, the single canonical implementation the
        rl2co environment also uses -- so a deployed policy sees the same
        post-switch population it was trained against. The returned
        leading axis is :attr:`popsize`; a caller whose generation width
        is wider pads it.

        Parameters
        ----------
        best_params : PyTree
            Best parameters seen so far, as one solution.
        sampler : Callable
            ``l2co.sampling`` function, called as
            ``sampler(key, best_params, popsize)``.
        key : PRNGKeyArray
            Key for the draws.

        Returns
        -------
        PyTree
            Population with a leading axis of :attr:`popsize`.
        """
        return POPULATION_HANDSHAKES[self.population_handshake](
            best_params, self.popsize, sampler, key
        )

    def bill_fevals(
        self, sub_state_before: PyTree, sub_state_after: PyTree
    ) -> Array:
        """Function evaluations one step of this sub-optimizer spent.

        Consumers that record ``OptHistory.fevals`` /
        ``HistoryState.fevals`` from an ask/tell wrapper must bill
        through here rather than assuming :attr:`popsize`. The two
        differ for line-search optimizers: ``lbfgs`` has a population
        of one but evaluates the objective once per line-search trial,
        up to ``max_linesearch_steps + 1`` in a single step (see
        ``ADR 0010-lbfgs-bills-one-feval-per-linesearch-trial.md``).
        Billing ``popsize`` there under-reports its cost by up to 21x
        and makes a wrapper-driven run non-comparable with the
        registry-driven runs it is measured against.

        Because the count is only known *after* the step, a caller that
        records history before dispatching the step has to reorder.

        Parameters
        ----------
        sub_state_before : PyTree
            This sub-optimizer's internal state before the step.
        sub_state_after : PyTree
            Its state after the step, holding the line-search info.

        Returns
        -------
        Array
            Scalar integer feval count for this step.
        """
        if self.fevals_fn is None:
            return jnp.asarray(self.popsize, dtype=int)
        return jnp.asarray(
            self.fevals_fn(sub_state_before, sub_state_after), dtype=int
        )


def grad_sub_optimizer(
    transform: optax.GradientTransformation,
    name_hash: int,
    opt_state_handshake: str = "continue",
    population_handshake: str = "best_one",
    max_fevals_per_step: int | None = None,
    family: str = FAMILY_GRADIENT,
    transfer_read_fn: Callable | None = None,
    transfer_write_fn: Callable | None = None,
) -> SubOpt:
    """Wrap an optax gradient transformation as a ``SubOpt``.

    Hyperparameters live inside ``transform`` (closed over per the
    optax convention; see rl2co's
    ``agent-reviews/optax_gradient_transformation_api.md`` §3).

    Parameters
    ----------
    transform : optax.GradientTransformation
        Any optax transformation. Its ``update`` is called once per
        outer step.
    name_hash : int
        Stable identifier written into ``HistoryState.update_step``;
        must be unique across the sub-optimizer tuple. Typically the
        hash of the source :class:`OptimizationStep`.
    opt_state_handshake : str, optional
        ``"continue"`` (default) or ``"reset"``; see
        :class:`SubOpt`. Defaulted so hand-built adapters keep the
        universal behaviour without opting in.

    Returns
    -------
    SubOpt
        Adapter ready to slot into a wrapper factory.
    """

    def init_fn(params: PyTree, key: PRNGKeyArray) -> PyTree:
        """Initialise the optax transform's internal state.

        Parameters
        ----------
        params : PyTree
            Parameters used to shape Adam moments, etc.
        key : PRNGKeyArray
            Unused; accepted for interface uniformity with
            population-based ``init_fn``.

        Returns
        -------
        PyTree
            optax optimizer state.
        """
        del key
        return transform.init(params)

    def step_fn(
        grads: PyTree,
        params: PyTree,
        sub_state: PyTree,
        key: PRNGKeyArray,
        **extra_args: Any,
    ) -> tuple[PyTree, PyTree]:
        """Apply one optax step.

        Parameters
        ----------
        grads : PyTree
            Outer-loop gradients for ``params``.
        params : PyTree
            Current parameters; forwarded to transforms that need
            them (e.g. weight decay).
        sub_state : PyTree
            Previous optax state.
        key : PRNGKeyArray
            Unused; accepted for interface uniformity.
        **extra_args
            Forwarded verbatim to ``transform.update`` as keyword
            arguments. Line-search transforms (e.g. ``optax.lbfgs``,
            which chains ``scale_by_zoom_linesearch``) require
            ``value`` (the current loss), ``grad`` (the gradient at
            ``params``), and ``value_fn`` (a callable ``params ->
            loss`` the line search probes at trial points). Plain
            transforms (Adam, SGD, …) accept and ignore these via
            optax's ``GradientTransformationExtraArgs`` support, so
            passing them is always safe; omitting them leaves the
            stock single-argument behaviour unchanged.

        Returns
        -------
        tuple[PyTree, PyTree]
            ``(updates, new_sub_state)`` — already optax-sign-correct.
        """
        del key
        return transform.update(grads, sub_state, params, **extra_args)

    return SubOpt(
        is_pop=False,
        popsize=1,
        name_hash=int(name_hash),
        init_fn=init_fn,
        step_fn=step_fn,
        tell_fn=None,
        opt_state_handshake=opt_state_handshake,
        population_handshake=population_handshake,
        max_fevals_per_step=max_fevals_per_step,
        # A plain transform bills 1 (== popsize) and a line-search
        # transform its true trial count, from the same helper the
        # registry path uses -- so an ask/tell wrapper's fevals are
        # comparable with a registry-driven run's. See
        # ``SubOpt.bill_fevals``.
        fevals_fn=step_fevals,
        family=family,
        transfer_read_fn=transfer_read_fn,
        transfer_write_fn=transfer_write_fn,
    )


def pop_sub_optimizer(
    init_fn: Callable[[PyTree, PRNGKeyArray], PyTree],
    ask_fn: Callable[[PyTree, PyTree, PRNGKeyArray], tuple[PyTree, PyTree]],
    tell_fn: Callable[[PyTree, PyTree, Array, PRNGKeyArray], PyTree],
    popsize: int,
    name_hash: int,
    opt_state_handshake: str = "continue",
    population_handshake: str = "best_one",
    max_fevals_per_step: int | None = None,
    family: str = FAMILY_GRADIENT,
    transfer_read_fn: Callable | None = None,
    transfer_write_fn: Callable | None = None,
    own_ask: bool = False,
) -> SubOpt:
    """Wrap an evosax-style ``(init, ask, tell)`` triple as a ``SubOpt``.

    The ``init/ask/tell`` callables are supplied by the user so this
    layer does not import evosax directly; see
    :func:`optstep_to_subopt` for the canonical wiring.

    Parameters
    ----------
    init_fn : Callable
        ``(params, key) -> sub_state``. Builds the ES internal state.
    ask_fn : Callable
        ``(params, sub_state, key) -> (population, new_sub_state)``,
        with ``population`` of shape ``(popsize, *params_shape)``.
    tell_fn : Callable
        ``(sub_state, population, fitnesses, key) -> new_sub_state``.
    popsize : int
        Number of candidates emitted per ``ask``. Static.
    name_hash : int
        Stable identifier; see :func:`grad_sub_optimizer`.
    opt_state_handshake : str, optional
        ``"continue"`` (default) or ``"reset"``; see :class:`SubOpt`.
    own_ask : bool, optional
        Whether a switch should draw this sub-optimizer's post-switch
        population from ``ask_fn`` instead of handing it a layout; see
        :class:`SubOpt`. Defaults to False.

    Returns
    -------
    SubOpt
        Adapter ready to slot into a wrapper factory.
    """
    return SubOpt(
        is_pop=True,
        popsize=int(popsize),
        name_hash=int(name_hash),
        init_fn=init_fn,
        step_fn=ask_fn,
        tell_fn=tell_fn,
        opt_state_handshake=opt_state_handshake,
        population_handshake=population_handshake,
        max_fevals_per_step=max_fevals_per_step,
        family=family,
        transfer_read_fn=transfer_read_fn,
        transfer_write_fn=transfer_write_fn,
        own_ask=own_ask,
    )


#                                       OptimizationStep -> SubOpt dispatch
# =============================================================================


def optstep_to_subopt(
    opt_step: OptimizationStep,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
) -> SubOpt:
    """Convert one :class:`~l2co_optimizers.OptimizationStep` to a ``SubOpt``.

    Unpacks the optimizer with :func:`l2co_optimizers.optimizer_parts`
    -- which resolves the name, popsize, family, transfer ports and the
    line-search feval bound, exactly as the ``UpdateClass`` factory
    would -- and attaches the handshake :func:`handshake_policy_for`
    assigns it.

    The parts path has no ``bounded`` channel and no per-evaluation-key
    channel, so ``'lbfgs'`` runs the stock line search here on every
    problem (ADR 0008); see :func:`l2co_optimizers.optimizer_parts`.

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec.
    model : PyTree
        Model whose trainable arrays size the population template.
    loss_fn : LossFunction
        The objective; needed to resolve the popsize.
    pass_rng : bool
        Whether ``loss_fn`` takes a key.

    Returns
    -------
    SubOpt
        Adapter ready to slot into a wrapper factory.

    Raises
    ------
    ValueError
        If ``opt_step.optimizer`` cannot be unpacked into parts.
    """
    parts = optimizer_parts(
        opt_step, model=model, loss_fn=loss_fn, pass_rng=pass_rng
    )
    handshake = handshake_policy_for(opt_step)
    if isinstance(parts, GradientParts):
        return grad_sub_optimizer(
            parts.transform,
            name_hash=parts.name_hash,
            opt_state_handshake=handshake.opt_state,
            population_handshake=handshake.population,
            max_fevals_per_step=parts.max_fevals_per_step,
            family=parts.family,
            transfer_read_fn=parts.transfer_read_fn,
            transfer_write_fn=parts.transfer_write_fn,
        )
    return pop_sub_optimizer(
        init_fn=parts.init_fn,
        ask_fn=parts.ask_fn,
        tell_fn=parts.tell_fn,
        popsize=parts.popsize,
        name_hash=parts.name_hash,
        opt_state_handshake=handshake.opt_state,
        population_handshake=handshake.population,
        family=parts.family,
        transfer_read_fn=parts.transfer_read_fn,
        transfer_write_fn=parts.transfer_write_fn,
        own_ask=parts.own_ask,
    )
