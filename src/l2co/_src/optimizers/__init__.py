"""Optimizer subpackage providing mappings from names to implementations."""

from .experimentdata import create_schedules_experimentdata
from .lbfgs import (
    lbfgs_per_eval_key,
    lbfgs_update,
    scale_by_zoom_linesearch_per_eval_key,
)
from .mapping import optimizer_mapping, optimizers, register_optimizer
from .shade import SHADE, shade_update
from .update_class import UpdateClass
from .utils import shade_popsize, variable_popsize, variable_popsize_even

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
