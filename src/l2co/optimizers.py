"""Public API for available optimizer configurations.

Stable import surface for the ``optimizer_mapping`` and ``optimizers`` dicts,
the ``register_optimizer`` extension point, the
``create_schedules_experimentdata`` helper, and the built-in ``SHADE``
algorithm (``shade_update`` factory, ``shade_popsize`` default).
"""

from ._src.optimizers import (
    SHADE,
    create_schedules_experimentdata,
    optimizer_mapping,
    optimizers,
    register_optimizer,
    shade_popsize,
    shade_update,
)

__all__ = [
    "SHADE",
    "create_schedules_experimentdata",
    "optimizer_mapping",
    "optimizers",
    "register_optimizer",
    "shade_popsize",
    "shade_update",
]
