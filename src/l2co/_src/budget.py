from typing import Optional

from .task import Task


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
