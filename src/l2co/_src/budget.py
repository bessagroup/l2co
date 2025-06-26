"""
Module for determining the budget for optimization tasks.
"""

#                                                                       Modules
# =============================================================================


# Standard
from typing import Optional

# Local
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


# =============================================================================


def fixed_budget(budget: int, **kwargs) -> int:
    return budget


def based_on_dimensionality(
    task: Task, budget_per_dimension: int,
        max_iterations: Optional[int] = None) -> int:

    budget = task.dimensionality * budget_per_dimension

    if max_iterations is not None:
        return min(budget, max_iterations)

    else:
        return budget
