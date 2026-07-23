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
``"rl2co"``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
from l2co_tasks import Task

# Local
from l2co._src.models.l2co_model import L2COModel
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.optimizers.mapping import register_optimizer
from l2co._src.optimizers.update_class import (
    UpdateClass,
    l2co_population_based_fn,
)
from l2co._src.strategy_wrapper import l2co_population_strategy
from l2co._src.typing import StopFunction

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


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
    hand them to :func:`l2co._src.update_class.init_` / ``run_`` (or wrap
    them in a :class:`l2co.RunState` directly).
    """

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
        bounded: tuple[float, float] | None = (None, None),
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

        This signature matches the registry contract called by
        :meth:`l2co.RunState.init`: ``task``, ``opt_hash``,
        ``bounded`` and ``stop_fn`` are supplied by ``RunState.init``
        directly, and the remaining construction arguments arrive
        through the :class:`~l2co._src.optimizer_schedule.
        OptimizationStep`'s ``hyperparameters`` once the ``"l2co"``
        optimizer is registered (importing ``l2co`` performs that
        registration via ``register_optimizer``). It can also be
        called directly, e.g. from a notebook.

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
            keep its meaning.
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
        bounded : tuple[float, float] | None, optional
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
            popsize), so ``RunState.init`` and ``reset`` sample a
            compatible population.

        Raises
        ------
        ValueError
            Propagated from the strategy factory: empty menu,
            stage-0 choice missing from the menu, missing
            ``task.global_min``, zero switch generation, or a
            non-traceable stage-1 model.
        """
        model = L2COModel.load(
            stage0_model=stage0_model, stage1_model=stage1_model
        )
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

        strategy = l2co_population_strategy(
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

# Inject l2co's own two-stage meta-optimizer into the optimizer registry so
# an ``OptimizationStep(optimizer="l2co", ...)`` resolves to
# ``L2COUpdateClass`` inside ``l2co.RunState.init``. This is a side effect of
# importing ``l2co`` (see ``l2co/__init__.py``); the base ``optimizers``
# mapping deliberately no longer hardcodes ``"l2co"`` so the registry treats
# both meta-optimizers ("l2co" here, "rl2co" from rl2co) the same way
# (``docs/adr/0007``). ``task``, ``opt_hash``, ``bounded`` and ``stop_fn``
# are explicit parameters; the remaining construction arguments
# (``budget_fn``, ``stage0_model``, ``stage1_model``, ``t``, ``sampler``,
# ``swap_optimizers``) flow in through the step's ``**hyperparameters``.
register_optimizer("l2co", L2COUpdateClass.l2co_population_update)
