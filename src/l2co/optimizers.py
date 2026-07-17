"""Public API for available optimizer configurations.

Stable import surface for the ``optimizer_mapping`` and ``optimizers`` dicts,
the ``register_optimizer`` extension point, the
``create_schedules_experimentdata`` helper, the built-in ``SHADE``
algorithm (``shade_update`` factory, ``shade_popsize`` default), and the
L-BFGS entry (``lbfgs_update`` factory — per-eval-key noise semantics on
stochastic tasks — with the ``lbfgs_per_eval_key`` optimizer and
``scale_by_zoom_linesearch_per_eval_key`` linesearch building blocks).
"""

from ._src.optimizers import (
    SHADE,
    create_schedules_experimentdata,
    lbfgs_per_eval_key,
    lbfgs_update,
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
    "lbfgs_update",
    "optimizer_mapping",
    "optimizers",
    "register_optimizer",
    "scale_by_zoom_linesearch_per_eval_key",
    "shade_popsize",
    "shade_update",
]
