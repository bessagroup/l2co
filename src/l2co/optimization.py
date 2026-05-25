"""Public API for optimization rollouts.

Stable import surface for the rollout/dataset-generation engines
(``RolloutWrapper``, ``RunOptimization``), the per-step ``OptimizationStep``
configuration, the ``UpdateClass`` adapter that wraps optimizers, and the
``RunState`` helpers (``reset``, ``run``, ``evaluate``).
"""

from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.optimizers.update_class import UpdateClass
from l2co._src.rollout import RolloutWrapper, RunOptimization
from l2co._src.run_state import RunState, reset, run
from l2co._src.update_class import evaluate

__all__ = [
    "OptimizationStep",
    "RolloutWrapper",
    "RunOptimization",
    "RunState",
    "UpdateClass",
    "evaluate",
    "reset",
    "run",
]
