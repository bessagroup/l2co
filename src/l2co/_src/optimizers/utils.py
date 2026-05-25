"""Utility helpers shared across optimizer implementations."""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import re
from typing import TYPE_CHECKING

# Third-party
import jax.numpy as jnp

# Local
from l2co._src.typing import PopSize

if TYPE_CHECKING:
    from l2co_tasks import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def normalize_key(key: str) -> str:
    """Strip non-alphanumerics and lowercase ``key`` for mapping lookups."""
    return re.sub(r"[^a-zA-Z0-9]", "", key).lower()


def variable_popsize(task: Task) -> int:
    """EvoSax-style default population size, ``int(4 + 3 * log(d))``.

    ``d`` is the problem dimensionality (``task.dimensionality``). This
    is the value that ``evosax`` itself uses when no population size is
    provided to a distribution- or population-based algorithm.

    Parameters
    ----------
    task : Task
        Task whose ``dimensionality`` drives the formula.

    Returns
    -------
    int
        Heuristic population size for the given task.
    """
    return int(4 + 3 * jnp.log(task.dimensionality))


def resolve_popsize(popsize: PopSize, task: Task) -> int:
    """Coerce a :data:`PopSize` (int or callable) to a concrete ``int``.

    Parameters
    ----------
    popsize : int or Callable[[Task], int]
        Either a literal population size or a callable that derives one
        from the ``Task`` (e.g. :func:`variable_popsize`).
    task : Task
        Task passed to ``popsize`` when it is callable.

    Returns
    -------
    int
        The resolved population size.
    """
    if callable(popsize):
        return int(popsize(task))
    return int(popsize)
