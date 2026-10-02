"""Population-size defaults and resolution for optimizer factories.

A factory's ``popsize`` argument is a :data:`~l2co._src.typing.PopSize`:
either a literal ``int`` or a callable deriving one from the ``Task``.
The callables here are the built-in defaults, and
:func:`resolve_popsize` turns either form into a concrete ``int``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

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


def variable_popsize_even(task: Task) -> int:
    """:func:`variable_popsize` rounded up to the nearest even number.

    Several EvoSax distribution-based algorithms require an even
    population size. The antithetic / mirrored-sampling strategies (ARS,
    ASEBO, CR_FM_NES, GuidedES, NoiseReuseES, OpenES, PersistentES, PGPE)
    ``assert population_size % 2 == 0`` at construction, and ESMC silently
    emits ``popsize - 1`` candidates for an odd population size — which
    then mismatches the ``popsize``-shaped buffers downstream. This
    variant is their default so they run for any task dimensionality
    (``variable_popsize`` is odd whenever ``int(4 + 3 * log(d))`` is odd,
    e.g. ``d == 3`` gives ``7``).

    Parameters
    ----------
    task : Task
        Task whose ``dimensionality`` drives the formula.

    Returns
    -------
    int
        :func:`variable_popsize` for ``task``, rounded up to the nearest
        even integer.
    """
    popsize = variable_popsize(task)
    return popsize + (popsize % 2)


def shade_popsize(task: Task) -> int:
    """:func:`variable_popsize` floored at 10, for SHADE.

    SHADE samples the greediness of its current-to-pbest/1 mutation as
    ``p_i ~ U[2 / NP, 0.2]`` (Sun et al. 2020, Eqs. 5-6), which is only
    a well-defined range for ``NP >= 10``. The paper-scale ``NP = 100``
    remains available via the ``popsize`` hyperparameter.

    Parameters
    ----------
    task : Task
        Task whose ``dimensionality`` drives the formula.

    Returns
    -------
    int
        ``max(10, int(4 + 3 * log(d)))`` for the given task.
    """
    return max(10, variable_popsize(task))


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
