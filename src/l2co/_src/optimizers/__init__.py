"""Optimizer subpackage providing mappings from names to implementations."""

from l2co._src.optimizers.experimentdata import create_schedules_experimentdata
from l2co._src.optimizers.lbfgs import (
    lbfgs_per_eval_key,
    lbfgs_update,
    scale_by_zoom_linesearch_per_eval_key,
)
from l2co._src.optimizers.mapping import (
    optimizer_mapping,
    optimizers,
    register_optimizer,
)
from l2co._src.optimizers.popsize import (
    shade_popsize,
    variable_popsize,
    variable_popsize_even,
)
from l2co._src.optimizers.shade import SHADE, shade_update
from l2co._src.optimizers.update_class import UpdateClass

__all__ = [
    "SHADE",
    "UpdateClass",
    "create_schedules_experimentdata",
    "lbfgs_per_eval_key",
    "lbfgs_update",
    "optimizer_mapping",
    "optimizers",
    "register_optimizer",
    "scale_by_zoom_linesearch_per_eval_key",
    "shade_popsize",
    "shade_update",
    "variable_popsize",
    "variable_popsize_even",
]
