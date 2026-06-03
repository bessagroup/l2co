"""Optimizer subpackage providing mappings from names to implementations."""

from .experimentdata import create_schedules_experimentdata
from .mapping import optimizer_mapping, optimizers, register_optimizer
from .update_class import UpdateClass
from .utils import variable_popsize, variable_popsize_even

__all__ = [
    "UpdateClass",
    "create_schedules_experimentdata",
    "optimizer_mapping",
    "optimizers",
    "register_optimizer",
    "variable_popsize",
    "variable_popsize_even",
]
