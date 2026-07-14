"""Public API for optimization rollouts.

Stable import surface for the rollout/dataset-generation engines
(``RolloutWrapper``, ``RunOptimization``), the per-step ``OptimizationStep``
configuration, the ``UpdateClass`` adapter that wraps optimizers, the
``RunState`` helpers (``reset``, ``run``, ``evaluate``), and the L2CO
strategy wrapper (``l2co_population_strategy``) that exposes a trained
two-stage model as an evosax-style strategy.
"""

from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.optimizers.update_class import UpdateClass
from l2co._src.rollout import RolloutWrapper, RunOptimization
from l2co._src.run_state import RunState, reset, run
from l2co._src.strategy_wrapper import l2co_population_strategy
from l2co._src.update_class import evaluate

__all__ = [
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
