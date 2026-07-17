"""Public API for available optimizer configurations.

Stable import surface for the ``optimizer_mapping`` and ``optimizers`` dicts,
the ``register_optimizer`` extension point, the
``create_schedules_experimentdata`` helper, the built-in ``SHADE``
algorithm (``shade_update`` factory, ``shade_popsize`` default), and the
per-eval-key L-BFGS variant (``lbfgs_per_eval_key_update`` factory,
``lbfgs_per_eval_key`` optimizer,
``scale_by_zoom_linesearch_per_eval_key`` linesearch).
"""

from ._src.optimizers import (
    SHADE,
    create_schedules_experimentdata,
    lbfgs_per_eval_key,
    lbfgs_per_eval_key_update,
    optimizer_mapping,
    optimizers,
    register_optimizer,
    scale_by_zoom_linesearch_per_eval_key,
    shade_popsize,
    shade_update,
)

__all__ = [
    "SHADE",
    "create_schedules_experimentdata",
    "lbfgs_per_eval_key",
    "lbfgs_per_eval_key_update",
    "optimizer_mapping",
    "optimizers",
    "register_optimizer",
    "scale_by_zoom_linesearch_per_eval_key",
    "shade_popsize",
    "shade_update",
]
