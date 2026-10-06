"""Public API for optimization rollouts.

Stable import surface for the rollout/dataset-generation engines
(``RolloutWrapper``, ``RunOptimization``), the per-step ``OptimizationStep``
configuration, the ``UpdateClass`` adapter that wraps optimizers (and,
since ADR 0017, carries the run loop as methods), the ``RunState``
helpers (``reset``, ``run``, ``evaluate``), and the L2CO strategy wrapper
(``l2co_population_strategy``) that exposes a trained two-stage model as an
evosax-style strategy.
"""

from l2co_optimizers import ALIAS_HYPERPARAMETER, OptimizationStep, UpdateClass

from l2co._src.model_evaluation import evaluate
from l2co._src.rollout import RolloutWrapper, RunOptimization
from l2co._src.run_state import RunState, reset, run
from l2co._src.strategy_wrapper import l2co_population_strategy

__all__ = [
    "ALIAS_HYPERPARAMETER",
    "OptimizationStep",
    "RolloutWrapper",
    "RunOptimization",
    "RunState",
    "UpdateClass",
    "evaluate",
    "l2co_population_strategy",
    "reset",
    "run",
]
