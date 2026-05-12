"""Rollout machinery, split from the historical ``run_optimization`` module.

* ``rollout_wrapper`` - JIT-friendly inference rollout (``RolloutWrapper``).
* ``dataset_generation`` - the ``RunOptimization`` ``DataGenerator`` used by
  the ``experiments/create_optimization_dataset`` Hydra pipeline.

Both names are re-exported here so the qualified path
``l2co._src.rollout.<Class>`` continues to resolve for pickled artefacts.
"""

from .dataset_generation import RunOptimization
from .rollout_wrapper import RolloutWrapper

# Pickle stability for pre-restructure checkpoints that pickled the
# qualified ``l2co._src.run_optimization.<Class>`` path.
for _cls in (RolloutWrapper, RunOptimization):
    _cls.__module__ = __name__
del _cls

__all__ = [
    "RolloutWrapper",
    "RunOptimization",
]
