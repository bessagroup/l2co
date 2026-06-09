"""Rollout machinery.

* ``rollout_wrapper`` - JIT-friendly inference rollout (``RolloutWrapper``).
* ``dataset_generation`` - the ``RunOptimization`` ``DataGenerator`` used by
  the ``experiments/create_optimization_dataset`` Hydra pipeline.

Both names are re-exported here so the qualified path
``l2co._src.rollout.<Class>`` still resolves for checkpoints created before
the package split.
"""

from .dataset_generation import RunOptimization
from .rollout_wrapper import RolloutWrapper

__all__ = [
    "RolloutWrapper",
    "RunOptimization",
]
