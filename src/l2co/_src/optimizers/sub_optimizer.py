"""Sub-optimizer adapters shared by the strategy wrappers.

A :class:`_SubOpt` is a thin static adapter that hides whether a
sub-optimizer is gradient-based (an optax ``GradientTransformation``)
or population-based (an evosax-style ``(init, ask, tell)`` triple).
Wrappers that expose a selection model as a duck-typed evosax
strategy — l2co's own :mod:`l2co._src.strategy_wrapper` and rl2co's
policy-deployment wrappers (``rl2co._src.optax_wrapper`` /
``rl2co._src.evosax_wrapper``, which import this module through a
shim) — dispatch across a tuple of ``_SubOpt`` adapters via
:func:`jax.lax.switch`, reading the static ``is_pop`` flag at Python
trace time to pick which branch implementation to build for each
slot.

Two factory functions build ``_SubOpt`` instances directly
(:func:`grad_sub_optimizer`, :func:`pop_sub_optimizer`); a third,
:func:`_optstep_to_subopt`, dispatches an
:class:`~l2co._src.optimizer_schedule.OptimizationStep` to the
appropriate factory by looking the optimizer name up in the canonical
l2co registries.

The module was authored in rl2co and promoted here because it imports
only l2co internals and both packages' wrappers need it — see
``docs/adr/0003-promote-subopt-adapter-from-rl2co.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from l2co_tasks import Task

from ..optimizer_schedule import OptimizationStep
from .evosax_implementations import normalized_evosax
from .mapping import optimizer_mapping
from .optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
)
from .shade import SHADE
from .utils import normalize_key

# =============================================================================

__all__ = [
    "_SubOpt",
    "_optstep_to_subopt",
    "_resolve_popsize",
    "grad_sub_optimizer",
    "l2co_native_evosax",
    "pop_sub_optimizer",
]


# l2co-native evosax-style algorithms: absent from evosax's own
# registries but subclassing its base API, so the generic evosax branch
# of ``_optstep_to_subopt`` drives them unchanged.
l2co_native_evosax: dict[str, type] = {"shade": SHADE}

# Shape-affecting hyperparameter names consumed by an algorithm's
# *constructor* rather than its flax-struct ``Params`` (which only
# holds value-like leaves). These are stripped from the hyperparameters
# before the ``default_params.replace(...)`` overrides.
_CONSTRUCTOR_HYPERPARAMETERS: dict[str, tuple[str, ...]] = {
    "shade": ("memory_size", "archive_size"),
}


#                                                       Sub-optimizer adapter
# =============================================================================


class _SubOpt(eqx.Module):
    """Internal adapter for one sub-optimizer.

    Every entry of a wrapper's sub-optimizer tuple is a ``_SubOpt``;
    the wrapper dispatches across them via :func:`jax.lax.switch` and
    reads the static ``is_pop`` flag at Python trace time to decide
    which branch implementation to build for each slot. All fields
    are static — a ``_SubOpt`` contributes no array leaves to the
    surrounding pytree.

    Build via :func:`grad_sub_optimizer` or :func:`pop_sub_optimizer`,
    or have the wrapper factories build them automatically from
    :class:`~l2co._src.optimizer_schedule.OptimizationStep` entries
    via :func:`_optstep_to_subopt`.

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
    """

    is_pop: bool = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    name_hash: int = eqx.field(static=True)
    init_fn: Callable = eqx.field(static=True)
    step_fn: Callable = eqx.field(static=True)
    tell_fn: Callable | None = eqx.field(static=True)


def grad_sub_optimizer(
    transform: optax.GradientTransformation,
    name_hash: int,
) -> _SubOpt:
    """Wrap an optax gradient transformation as a ``_SubOpt``.

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

    Returns
    -------
    _SubOpt
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

    return _SubOpt(
        is_pop=False,
        popsize=1,
        name_hash=int(name_hash),
        init_fn=init_fn,
        step_fn=step_fn,
        tell_fn=None,
    )


def pop_sub_optimizer(
    init_fn: Callable[[PyTree, PRNGKeyArray], PyTree],
    ask_fn: Callable[[PyTree, PyTree, PRNGKeyArray], tuple[PyTree, PyTree]],
    tell_fn: Callable[[PyTree, PyTree, Array, PRNGKeyArray], PyTree],
    popsize: int,
    name_hash: int,
) -> _SubOpt:
    """Wrap an evosax-style ``(init, ask, tell)`` triple as a ``_SubOpt``.

    The ``init/ask/tell`` callables are supplied by the user so this
    layer does not import evosax directly; see
    :func:`_optstep_to_subopt` for the canonical wiring.

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

    Returns
    -------
    _SubOpt
        Adapter ready to slot into a wrapper factory.
    """
    return _SubOpt(
        is_pop=True,
        popsize=int(popsize),
        name_hash=int(name_hash),
        init_fn=init_fn,
        step_fn=ask_fn,
        tell_fn=tell_fn,
    )


#                                       OptimizationStep -> _SubOpt dispatch
# =============================================================================


def _resolve_popsize(opt_step: OptimizationStep, task: Task) -> int:
    """Resolve the canonical popsize for ``opt_step`` via l2co's rules.

    Builds an :class:`~l2co._src.optimizers.update_class.UpdateClass`
    purely to read its ``popsize`` attribute — ``1`` for optax-style
    optimizers, an explicit value or ``int(4 + 3 * log(d))`` for
    evosax — so the wrapper's population sizing matches the call site
    in ``l2co._src.run_state.RunState.init``. The ``UpdateClass``
    instance itself is discarded; the wrapper builds its own
    optax/evosax adapter against the same hyperparameters and the
    correctly-resolved popsize.

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec; ``opt_step.hyperparameters`` (including any
        explicit ``popsize`` kwarg) is forwarded to the factory.
    task : Task
        Forwarded to :func:`~l2co._src.optimizers.mapping.
        optimizer_mapping`; needed because some evosax algorithms
        read the model dimensionality from ``task`` when computing
        their default popsize.

    Returns
    -------
    int
        Resolved popsize.
    """
    update_class = optimizer_mapping(normalize_key(opt_step.optimizer))(
        task=task,
        opt_hash=opt_step.hash,
        bounded=(None, None),
        stop_fn=opt_step.stopping_fn,
        **dict(opt_step.hyperparameters),
    )
    return int(update_class.popsize)


def _optstep_to_subopt(
    opt_step: OptimizationStep,
    task: Task,
) -> _SubOpt:
    """Convert one :class:`OptimizationStep` to a :class:`_SubOpt`.

    Dispatch is by ``opt_step.optimizer`` against
    :data:`~l2co._src.optimizers.optax_implementations.
    normalized_optax_normal`, :data:`...normalized_optax_from_state`,
    :data:`~l2co._src.optimizers.evosax_implementations.
    normalized_evosax`, and :data:`l2co_native_evosax` (l2co's own
    evosax-style algorithms, e.g. ``'shade'``). Population size is
    resolved by :func:`_resolve_popsize`.

    Note that this path has no ``bounded`` channel: algorithms whose
    internals use box bounds (SHADE's midpoint boundary repair and
    turning-based mutation) receive them as ordinary ``x_min`` /
    ``x_max`` hyperparameters instead; without them SHADE runs
    unbounded (its repair degrades to a no-op).

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec; ``opt_step.optimizer`` is the name (normalised
        by lowercase/alphanumeric, matching ``l2co``'s registry),
        and ``opt_step.hyperparameters`` is forwarded to the
        underlying optax/evosax constructor (after stripping
        ``popsize`` itself and any :data:`_CONSTRUCTOR_HYPERPARAMETERS`
        entries, which go to the algorithm constructor rather than
        its ``Params``).
    task : Task
        Task containing the model whose trainable arrays size the
        evosax ``solution=`` template, and which is forwarded to
        :func:`_resolve_popsize` for popsize resolution.

    Returns
    -------
    _SubOpt
        Adapter ready to slot into a wrapper factory.

    Raises
    ------
    ValueError
        If ``opt_step.optimizer`` is not found in either registry.
    """
    name = normalize_key(opt_step.optimizer)
    hyperparams = dict(opt_step.hyperparameters)
    popsize = _resolve_popsize(opt_step, task)

    # ``popsize`` has now been consumed by ``_resolve_popsize``;
    # strip it before forwarding ``hyperparams`` to the optax /
    # evosax constructors below, neither of which accepts it as a
    # kwarg.
    hyperparams.pop("popsize", None)

    # --- Optax / gradient-based ----------------------------------------
    if name in normalized_optax_normal or name in normalized_optax_from_state:
        if name in normalized_optax_normal:
            optax_cls = normalized_optax_normal[name]
        else:
            optax_cls = normalized_optax_from_state[name]
        transform = optax_cls(**hyperparams)
        return grad_sub_optimizer(transform, name_hash=opt_step.hash)

    # --- Evosax / population-based -------------------------------------
    if name in normalized_evosax or name in l2co_native_evosax:
        algo_cls = normalized_evosax.get(name) or l2co_native_evosax[name]
        # Shape-affecting hyperparameters go to the constructor, not to
        # ``default_params.replace`` (e.g. SHADE's memory/archive size).
        constructor_kwargs = {
            kwarg: hyperparams.pop(kwarg)
            for kwarg in _CONSTRUCTOR_HYPERPARAMETERS.get(name, ())
            if kwarg in hyperparams
        }
        # evosax wants ``solution`` to infer per-candidate shape;
        # ``task.model``'s trainable arrays are the standard template
        # (matching ``RunState.init``).
        solution_template = eqx.filter(task.model, eqx.is_inexact_array)
        algo = algo_cls(
            population_size=popsize,
            solution=solution_template,
            **constructor_kwargs,
        )
        es_params = (
            algo.default_params.replace(**hyperparams)
            if hyperparams
            else algo.default_params
        )
        is_distribution_based = issubclass(
            algo_cls, DistributionBasedAlgorithm
        )
        is_population_based = issubclass(algo_cls, PopulationBasedAlgorithm)
        if not (is_distribution_based or is_population_based):
            raise ValueError(
                f"evosax algorithm {algo_cls.__name__} is not registered "
                f"as distribution-based or population-based; cannot "
                f"build a _SubOpt automatically."
            )

        def init_fn(params: PyTree, key: PRNGKeyArray) -> PyTree:
            """Initialise the ES state around ``params``.

            Dispatches on the algorithm family (distribution-based
            vs population-based):

            * **Distribution-based** (CMA-ES, SepCMA-ES, …) — init
              takes the single-instance ``params`` as the mean of
              the initial distribution.
            * **Population-based** (PSO, DE, …) — init takes an
              initial population (broadcast of ``params``) plus
              placeholder fitnesses; the ES updates them on the
              first ``tell``.

            Parameters
            ----------
            params : PyTree
                Single-instance parameter pytree.
            key : PRNGKeyArray
                ES initialisation key.

            Returns
            -------
            PyTree
                evosax algorithm state.
            """
            if is_distribution_based:
                return algo.init(key, params, es_params)
            pop0 = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (popsize, *x.shape)),
                params,
            )
            # Placeholder fitness is +inf, not zero: greedy-selection
            # algorithms (DE, SHADE) only replace a parent when the
            # trial's fitness beats the stored one, so a zero
            # placeholder would freeze the population until a trial
            # reached a negative loss.
            fit0 = jnp.full((popsize,), jnp.inf, dtype=float)
            return algo.init(
                key=key,
                population=pop0,
                fitness=fit0,
                params=es_params,
            )

        def ask_fn(
            params: PyTree,
            sub_state: PyTree,
            key: PRNGKeyArray,
        ) -> tuple[PyTree, PyTree]:
            """Ask the ES for a fresh population.

            Parameters
            ----------
            params : PyTree
                Current parameters (unused — the ES proposes its
                own center).
            sub_state : PyTree
                Current ES state.
            key : PRNGKeyArray
                Sampling key.

            Returns
            -------
            tuple[PyTree, PyTree]
                ``(population, new_sub_state)`` with ``population``
                of shape ``(popsize, *params_shape)``.
            """
            del params
            return algo.ask(key=key, state=sub_state, params=es_params)

        def tell_fn(
            sub_state: PyTree,
            population: PyTree,
            fitnesses: Float[Array, " popsize"],
            key: PRNGKeyArray,
        ) -> PyTree:
            """Tell the ES the previous round's fitnesses.

            Parameters
            ----------
            sub_state : PyTree
                Current ES state.
            population : PyTree
                The population that was evaluated, shape
                ``(popsize, *params_shape)``.
            fitnesses : Float[Array, " popsize"]
                Per-candidate losses (to be minimised).
            key : PRNGKeyArray
                Sampling key.

            Returns
            -------
            PyTree
                Updated ES state.
            """
            new_state, _ = algo.tell(
                key=key,
                population=population,
                fitness=fitnesses,
                state=sub_state,
                params=es_params,
            )
            return new_state

        return pop_sub_optimizer(
            init_fn=init_fn,
            ask_fn=ask_fn,
            tell_fn=tell_fn,
            popsize=popsize,
            name_hash=opt_step.hash,
        )

    raise ValueError(
        f"Optimizer {opt_step.optimizer!r} is not registered in "
        f"l2co's optax_mapping or evosax_mapping; cannot build a "
        f"_SubOpt automatically. Consider passing a hand-built "
        f"_SubOpt via grad_sub_optimizer/pop_sub_optimizer."
    )
