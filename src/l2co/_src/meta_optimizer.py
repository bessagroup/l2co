"""
The L2CO meta-optimizer as a self-registering ``UpdateClass``.

``L2COUpdateClass`` is a thin subclass of
:class:`l2co._src.optimizers.update_class.UpdateClass` whose step runs
the two-stage L2CO strategy (see ``docs/adr/0002``). It lives here --
outside the ``optimizers`` subpackage -- because its constructor imports
:class:`~l2co._src.models.l2co_model.L2COModel` and
:func:`~l2co._src.strategy_wrapper.l2co_population_strategy`, whereas the
optimizers subpackage must not depend on ``models`` / ``strategy_wrapper``
(that back-edge was the circular import ``l2co_update`` used to dodge with
deferred imports -- see ``docs/adr/0007``). Registering ``"l2co"`` from
here, as an import-time side effect, makes l2co's own meta-optimizer wire
itself into the registry exactly the way importing ``rl2co`` registers
``"rl2co"``. The closure factory ``l2co_population_based_fn`` that builds
its ``(init_fn, step_fn)`` lives here for the same reason.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import (
    InitFunction,
    InputParameters,
    LossFunction,
    OptHistory,
    OptimizationStep,
    StepFunction,
    StopFunction,
    UpdateClass,
    register_schedule_namer,
)
from l2co_tasks import Task

# Local
from l2co._src.models.l2co_model import L2COModel
from l2co._src.strategy_wrapper import _l2co_population_strategy
from l2co._src.switching.menu_eval import menu_loss_and_grad
from l2co._src.task_bridge import register_optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                               Closure factory
# =============================================================================


def l2co_population_based_fn(
    strategy,
    static: PyTree,
    loss_fn: LossFunction,
    bounded: tuple[float | None, float | None] | None,
    pass_rng: bool,
) -> tuple[InitFunction, StepFunction]:
    """Build ``(init_fn, step_fn)`` driving the L2CO strategy wrapper.

    The analogue of :func:`evosax_population_based_fn` for
    :func:`l2co._src.strategy_wrapper.l2co_population_strategy`
    (mirroring rl2co's ``rl2co_population_based_fn``). Three
    differences from the stock evosax factory:

    * **Ask-eval-tell rhythm.** The wrapper proposes via ``ask``
      before it can ``tell`` (the stage-1 fire and the handshake
      happen inside ``ask``), so each ``step_fn`` asks, evaluates the
      proposed population, then tells. The stock factory instead
      evaluates the carried population, tells, then asks.
    * **Evaluation is dispatched to the active optimizer's width.**
      The menu mixes gradient-based and population-based
      sub-optimizers, and the wrapper's ``tell`` consumes ``grads=``
      but reads only ``grads[0]``. Rather than evaluating the full
      generation width with gradients every generation, the evaluation
      goes through :func:`~l2co._src.optimizers.menu_eval.
      menu_loss_and_grad`, which evaluates ``popsize`` rows value-only
      for a population generation and the single iterate with
      gradients for a gradient one. That helper also owns the NaN
      contract (padding rows and diverged candidates alike), so the
      sentinel is robust regardless of how ``loss_fn`` treats NaN
      inputs. See ``docs/adr/0011-per-optimizer-evaluation-width-on-
      the-strategy-bridge.md``.
    * **The ``update_step`` hash is dynamic.** Before the stage-1
      fire every entry carries the stage-0 optimizer's hash, after a
      switching fire the stage-1 optimizer's — read from the wrapper
      state after each ``ask`` — so the stored history records which
      optimizer produced each entry (one combined history instead of
      the old path's two concatenated per-stage histories).

    ``init_fn`` does **not** evaluate the initial population: it
    forwards an all-``NaN`` fitness to ``strategy.init``, matching
    the stock machinery (where the initial samples are only evaluated
    by ``reset``'s hidden bookkeeping pass, outside any history) and
    the old eager path's init point (``samples[0]``).

    Parameters
    ----------
    strategy : l2co._src.strategy_wrapper._L2COPopulationStrategy
        The wrapper returned by ``l2co_population_strategy``. Closed
        over; its stage-1 weights become compile-time constants under
        JIT. (Typed loosely to keep this module importable without
        the strategy wrapper, which is imported instead by
        :meth:`l2co._src.meta_optimizer.L2COUpdateClass.
        l2co_population_update`, the sole caller.)
    static : PyTree
        Non-inexact-array partition of ``task.model``; recombined
        with each population via :func:`equinox.combine` before
        evaluation.
    loss_fn : LossFunction
        Task loss function (``task.loss_fn``).
    bounded : tuple[float | None, float | None] or None
        Box bounds clipped onto every asked population before
        evaluation; ``None`` or ``(None, None)`` disables clipping.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword
        (``task.pass_rng``).

    Returns
    -------
    tuple[InitFunction, StepFunction]
        ``(init_fn, step_fn)`` ready to slot into an
        :class:`UpdateClass`.
    """
    if bounded is None:
        bounded = (None, None)

    popsize = strategy.population_size
    default_params = strategy.default_params
    name_hashes = strategy.name_hashes_arr
    popsizes = jnp.array(
        [sub.popsize for sub in strategy.sub_optimizers], dtype=int
    )
    evaluate = menu_loss_and_grad(
        sub_optimizers=strategy.sub_optimizers,
        population_size=popsize,
        static=static,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
    )

    def init_fn(
        params: InputParameters,
        key: PRNGKeyArray,
        **sample,
    ) -> PyTree:
        """Build the wrapper state from the initial population.

        No evaluation happens here (see the factory docstring): the
        fitness forwarded to ``strategy.init`` is all-``NaN``, so
        best-tracking seeds to ``+inf`` and the sub-optimizers
        initialise around ``params[0]``.

        Parameters
        ----------
        params : InputParameters
            Initial population, leading axis
            ``strategy.population_size``.
        key : PRNGKeyArray
            Key for sub-optimizer initialisation.
        **sample
            Current data batch forwarded by
            :meth:`UpdateClass.init_state`; unused.

        Returns
        -------
        PyTree
            The wrapper state, used as the opt_state in the scan
            carry.
        """
        fitness = jnp.full((popsize,), jnp.nan, dtype=float)
        return strategy.init(key, params, fitness, default_params)

    def step_fn(
        carry: tuple[InputParameters, PyTree, PRNGKeyArray],
        sample: dict,
    ) -> tuple[tuple[InputParameters, PyTree, PRNGKeyArray], OptHistory]:
        """Run one ask-eval-tell generation of the L2CO wrapper.

        Parameters
        ----------
        carry : tuple[InputParameters, PyTree, PRNGKeyArray]
            ``(params, wrapper_state, key)``. ``params`` is the
            scan's carried population; the asked population is
            returned in its slot so ``history.params`` stays
            meaningful.
        sample : dict
            Current data batch.

        Returns
        -------
        tuple[tuple[InputParameters, PyTree, PRNGKeyArray], OptHistory]
            ``((population, new_state, new_key), history)`` with
            ``history.update_step`` carrying the dynamic hash of the
            sub-optimizer active this generation.
        """
        _params, state, key = carry
        new_key, ask_key, eval_key, tell_key = jr.split(key, 4)

        # Ask: fires stage-1 at the switch generation and emits either
        # the best_one handshake (on a switch) or the active
        # sub-optimizer's own population.
        population, state = strategy.ask(ask_key, state, default_params)
        population = jax.tree.map(lambda p: jnp.clip(p, *bounded), population)

        # The sub-optimizer active *this* generation; drives both the
        # dynamic history hash and the feval count.
        active_idx = state.active_opt_idx

        # Eval at the active sub-optimizer's evaluation width: a
        # population generation gets ``popsize`` rows value-only, a
        # gradient one gets the single iterate with gradients. Padding
        # rows and diverged candidates come back NaN.
        eval_keys = jr.split(eval_key, popsize)
        loss, grads = evaluate(active_idx, population, sample, eval_keys)

        def value_fn(candidate_params: PyTree) -> jax.Array:
            """Task loss of one candidate params partition.

            Lets line-search grad sub-optimizers (``optax.lbfgs``)
            probe trial points; reuses candidate 0's key for
            stochastic losses so the line search is consistent with
            the ``value``/``grad`` passed alongside it.
            """
            candidate_model = eqx.combine(candidate_params, static)
            if pass_rng:
                return loss_fn(candidate_model, key=eval_keys[0], **sample)
            return loss_fn(candidate_model, **sample)

        state, _ = strategy.tell(
            tell_key,
            population,
            loss,
            state,
            default_params,
            grads=grads,
            value_fn=value_fn,
        )

        history = OptHistory(
            loss=loss,
            params=population,
            grads=grads,
            update_step=name_hashes[active_idx].astype(int),
            fevals=popsizes[active_idx],
            iterations=jnp.array(1, dtype=int),
        )
        return (population, state, new_key), history

    return init_fn, step_fn


#                                                               L2COUpdateClass
# =============================================================================


class L2COUpdateClass(UpdateClass):
    """``UpdateClass`` whose step runs the two-stage L2CO strategy.

    A thin subclass of :class:`l2co.UpdateClass` -- it adds no fields,
    only the :meth:`l2co_population_update` classmethod. The subclass
    lives here (not on ``UpdateClass`` itself) so that the optimizers
    subpackage stays free of any import into ``models`` /
    ``strategy_wrapper``; the classmethod needs both, so it is authored
    on the meta-optimizer side -- mirroring rl2co's ``RL2COUpdateClass``
    (which lives on the rl2co side because l2co may not import rl2co).
    Built instances behave identically to any other ``UpdateClass`` --
    call their inherited ``init_state`` / ``run`` / ``batch_run`` (or
    wrap them in a :class:`l2co.RunState` directly).
    """

    # The two-stage strategy picks a sub-optimizer from its menu and
    # dispatches on that index (``jax.lax.switch`` in
    # :mod:`l2co._src.strategy_wrapper`), and the pick differs per
    # realization -- so realizations must not be vectorized together.
    # See :meth:`UpdateClass.batch_run`, which reads this marker.
    sequential_realizations = True

    @classmethod
    def l2co_population_update(
        cls,
        task: Task,
        opt_hash: int,
        budget_fn: Callable,
        stage0_model: str,
        stage1_model: str,
        t: float,
        sampler: Callable,
        swap_optimizers: list | None = None,
        feature_processor_impute: Mapping[str, Any] | None = None,
        bounded: tuple[float | None, float | None] | None = (None, None),
        stop_fn: StopFunction | None = None,
        **hyperparameters,
    ) -> UpdateClass:
        """Construct an ``UpdateClass`` running the L2CO strategy.

        The canonical execution path for
        ``OptimizationStep(optimizer="l2co")`` (see
        ``docs/adr/0002``): loads the two-stage
        :class:`~l2co._src.models.l2co_model.L2COModel`, wraps it as
        an evosax-style strategy via
        :func:`~l2co._src.strategy_wrapper.l2co_population_strategy`
        (which performs the stage-0 fire eagerly, at construction),
        and bridges it through :func:`l2co_population_based_fn` so
        the ordinary rollout machinery drives it like any plain
        optimizer. The stage-1 fire happens in-graph at
        ``round(t * budget_fn(task=task))`` completed generations.

        This signature matches the meta-optimizer registry contract
        called by :func:`l2co.init_run_state`: ``task``, ``opt_hash``,
        ``bounded`` and ``stop_fn`` are supplied by ``init_run_state``
        directly, and the remaining construction arguments arrive
        through the :class:`~l2co._src.optimizer_schedule.
        OptimizationStep`'s ``hyperparameters`` once the ``"l2co"``
        optimizer is registered (importing ``l2co`` performs that
        registration via l2co's ``register_optimizer``,
        ``docs/adr/0018``). It can also be called directly, e.g. from
        a notebook.

        Parameters
        ----------
        task : Task
            Task providing the model, loss function, features, and
            ``global_min`` (required — see the strategy factory).
        opt_hash : int
            Stable hash stored in the static ``hash`` field. Unlike
            the stock factories it is **not** stamped into the
            per-generation ``OptHistory.update_step``, which is
            dynamic (the active sub-optimizer's hash); it is used
            only for the ``skip_step`` dummy entry and
            ``OptHistory.init``.
        budget_fn : Callable
            Budget function called as ``budget_fn(task=task)``
            (matching ``l2co.budget.fixed_budget`` /
            ``based_on_dimensionality``). Resolves the budget the
            switch generation is computed against — it must equal
            the ``n_iterations`` of the eventual run for ``t`` to
            keep its meaning. An iteration budget: called without an
            optimizer, ``evaluations_based_on_dimensionality`` raises
            (``docs/adr/0026``).
        stage0_model : str
            Path to the stage-0 model file (without extension),
            forwarded to ``L2COModel.load``.
        stage1_model : str
            Path to the stage-1 model file (without extension).
        t : float
            Stage-0 fraction of the budget.
        sampler : Callable
            ``l2co.sampling`` function filling the post-fire
            handshake population beyond slot 0. Configs should name
            the same sampler the runner uses for initial sampling —
            the old path implicitly reused it.
        swap_optimizers : list | None, optional
            Explicit replacement menu: a list of
            ``OptimizationStep``s (or their ``to_dict`` dicts),
            applied to both stages via
            ``L2COModel.swap_optimizers``. ``None`` (default) keeps
            the menu baked into the loaded model. Replaces the old
            hardcoded four-optimizer list (``docs/adr/0002``).
        feature_processor_impute : Mapping[str, Any] | None, optional
            Fill values, as ``{feature: value}``, for schema features
            a task tag does not carry. Without this a checkpoint can
            only be run on task sets that tag everything its training
            schema named -- ``FeatureProcessor.transform`` raises
            ``KeyError`` on the first feature a suite omits, in both
            stages. Applied to both stages' ``feature_processor_task``
            via :meth:`L2COModel.with_feature_impute`, which leaves
            vocabularies and feature count untouched, so the checkpoint
            still loads unchanged; what changes is only that the absent
            feature is asserted to have the given value. Pick fills
            that are true of the task family (or, failing that, the
            training marginal), and report them with any result: the
            encoded vector does not record which entries were
            invented. Same mechanism as rl2co's
            ``feature_processor_impute`` (both draw on
            ``l2co.FeatureProcessor.with_imputation``). Default
            ``None``.
        bounded : tuple[float | None, float | None] or None, optional
            Box bounds clipped onto every asked population before
            evaluation. ``None`` is treated as ``(None, None)``.
        stop_fn : StopFunction | None, optional
            Stopping function.
        **hyperparameters
            Ignored; accepted so unrecognised schedule
            hyperparameters do not break the registry call.

        Returns
        -------
        UpdateClass
            Configured L2CO meta-optimizer wrapper; its ``popsize``
            is the strategy's ``population_size`` (max menu
            popsize), so ``init_run_state`` and ``reset`` sample a
            compatible population.

        Raises
        ------
        ValueError
            Propagated from the strategy factory: empty menu,
            stage-0 choice missing from the menu, missing
            ``task.global_min``, zero switch generation, or a
            non-traceable stage-1 model. Also propagated from
            ``FeatureProcessor.with_imputation`` when
            ``feature_processor_impute`` names a feature the
            processor does not encode, or gives a fill value outside
            that feature's vocabulary.
        """
        model = L2COModel.load(
            stage0_model=stage0_model, stage1_model=stage1_model
        )
        if feature_processor_impute is not None:
            model = model.with_feature_impute(dict(feature_processor_impute))
        if swap_optimizers is not None:
            steps = [
                step
                if isinstance(step, OptimizationStep)
                else OptimizationStep.from_dict(step)
                for step in swap_optimizers
            ]
            model = model.swap_optimizers(optimizers=steps)

        if bounded is None:
            bounded = (None, None)

        max_iterations = budget_fn(task=task)

        strategy = _l2co_population_strategy(
            model=model,
            task=task,
            sampler=sampler,
            t=t,
            max_iterations=max_iterations,
        )

        _, static = eqx.partition(task.model, eqx.is_inexact_array)
        init_fn, step_fn = l2co_population_based_fn(
            strategy=strategy,
            static=static,
            loss_fn=task.loss_fn,
            bounded=bounded,
            pass_rng=task.pass_rng,
        )

        return cls(
            init_fn=init_fn,
            step_fn=step_fn,
            popsize=strategy.population_size,
            hash=opt_hash,
            stop_fn=stop_fn,
        )


#                                                       l2co registry injection
# =============================================================================

# Inject l2co's own two-stage meta-optimizer into l2co's Task-level registry
# (``docs/adr/0018``) so an ``OptimizationStep(optimizer="l2co", ...)``
# resolves to ``L2COUpdateClass`` inside ``l2co.init_run_state``. This is a
# side effect of importing ``l2co`` (see ``l2co/__init__.py``); the bare
# ``optimizers`` mapping never holds ``"l2co"``, so the registry treats
# both meta-optimizers ("l2co" here, "rl2co" from rl2co) the same way
# (``docs/adr/0007``). ``task``, ``opt_hash``, ``bounded`` and ``stop_fn``
# are explicit parameters; the remaining construction arguments
# (``budget_fn``, ``stage0_model``, ``stage1_model``, ``t``, ``sampler``,
# ``swap_optimizers``) flow in through the step's ``**hyperparameters``.
register_optimizer("l2co", L2COUpdateClass.l2co_population_update)


def l2co_schedule_name(step: OptimizationStep) -> str:
    """Schedule name for an ``"l2co"`` step: checkpoint stems plus ``t``.

    ``l2co_s0-<stage0 stem>_s1-<stage1 stem>_t-<t>`` -- the model paths
    are reduced to their stems so they do not leak into SCHEDULE labels,
    plot legends and databank keys. Byte-identical to the branch
    ``OptimizationStep.name`` carried before the l2co-optimizers split
    (l2co ADR 0016); an explicit ``alias`` still wins over it.

    Parameters
    ----------
    step : OptimizationStep
        An ``"l2co"`` step.

    Returns
    -------
    str
        The schedule name.
    """
    parts = ["l2co"]
    for key, prefix in (
        ("stage0_model", "s0"),
        ("stage1_model", "s1"),
    ):
        path = step.hyperparameters.get(key, "")
        if path:
            parts.append(f"{prefix}-{Path(path).stem}")
    t = step.hyperparameters.get("t")
    if t is not None:
        parts.append(f"t-{t}")
    return "_".join(parts)


register_schedule_namer("l2co", l2co_schedule_name)
