"""Public API for optimization rollouts.

Stable import surface for the rollout/dataset-generation engines
(``RolloutWrapper``, ``RunOptimization``), the per-step ``OptimizationStep``
configuration, the ``UpdateClass`` adapter that wraps optimizers (and,
since ADR 0017, carries the run loop as methods), the ``RunState``
helpers (``reset``, ``run``, ``evaluate``; all three re-exported from
``l2co_optimizers``, ADR 0017), ``init_run_state`` -- which resolves an
``OptimizationStep`` against a ``Task`` (ADR 0018) -- and the L2CO
strategy wrapper
(``l2co_population_strategy``) that exposes a trained two-stage model as an
evosax-style strategy.
"""

from l2co_optimizers import (
    ALIAS_HYPERPARAMETER,
    OptimizationStep,
    RunState,
    UpdateClass,
    evaluate,
    reset,
    run,
)

from l2co._src.rollout import RolloutWrapper, RunOptimization
from l2co._src.strategy_wrapper import l2co_population_strategy
from l2co._src.task_bridge import init_run_state

__all__ = [
    "ALIAS_HYPERPARAMETER",
    "OptimizationStep",
    "RolloutWrapper",
    "RunOptimization",
    "RunState",
    "UpdateClass",
    "evaluate",
    "init_run_state",
    "l2co_population_strategy",
    "reset",
    "run",
]
