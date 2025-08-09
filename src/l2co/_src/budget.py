"""
Module for determining the budget for optimization tasks.
"""

#                                                                       Modules
# =============================================================================


# Standard

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
    """
    Return a fixed budget for optimization tasks.

    Parameters
    ----------
    budget : int
        Fixed budget value.
    **kwargs : dict
        Additional arguments (ignored).

    Returns
    -------
    int
        The fixed budget value.
    """
    return budget


def based_on_dimensionality(
    task: Task, budget_per_dimension: int,
    max_iterations: int | None = None
) -> int:
    """
    Calculate the budget based on the dimensionality of the task.

    Parameters
    ----------
    task : Task
        The optimization task.
    budget_per_dimension : int
        Budget allocated per dimension of the task.
    max_iterations : Optional[int], optional
        Maximum number of iterations allowed, by default None.

    Returns
    -------
    int
        The calculated budget, capped at `max_iterations` if provided.
    """

    budget = task.dimensionality * budget_per_dimension

    if max_iterations is not None:
        return min(budget, max_iterations)

    else:
        return budget
