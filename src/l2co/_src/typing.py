"""
Module for defining type aliases used throughout the optimization process.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from abc import abstractmethod

# Standard
from collections.abc import Callable
from typing import Any

import equinox as eqx

# Third-party
from evosax.algorithms.base import State as EvoSaxState
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from optax import OptState as OptaxOptState

from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptHistoryType(eqx.Module):
    loss: eqx.AbstractVar[Float[Array, " popsize"]]
    params: eqx.AbstractVar[Float[PyTree, " popsize dim"]]
    grads: eqx.AbstractVar[Float[PyTree, " popsize dim"]]
    update_step: eqx.AbstractVar[Int[Array, " popsize"]]
    fevals: eqx.AbstractVar[Int[Array, " popsize"]]
    iterations: eqx.AbstractVar[Int[Array, " popsize"]]
    """Base class for optimizer history."""

    @classmethod
    @abstractmethod
    def init(
        cls, params: PyTree, popsize: int, last_step: int
    ) -> OptHistoryType:
        """Initialize the history."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_init_sample(
        cls, update_state, task: Task, key: PRNGKeyArray
    ) -> OptHistoryType:
        """Create history from initial sample."""
        raise NotImplementedError

    @abstractmethod
    def update(self, history: OptHistoryType) -> OptHistoryType:
        """Update the recent history with the current history."""
        raise NotImplementedError


# =============================================================================

InputParameters = PyTree
OptState = PyTree

StepFunction = Callable[
    [tuple[InputParameters, PyTree, PRNGKeyArray]],
    tuple[tuple[InputParameters, PyTree, PRNGKeyArray], dict],
]

StopFunction = Callable[[OptHistoryType, PyTree, int], bool]

InitFunction = Callable[[PyTree], OptaxOptState | EvoSaxState]

LossFunction = Any

BudgetFn = Callable[[Any], int]
