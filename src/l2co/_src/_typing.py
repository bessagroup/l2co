"""
Module for defining type aliases used throughout the optimization process.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from abc import abstractmethod

# Standard
from collections.abc import Callable
from typing import Any, Union

import equinox as eqx
import jax

# Third-party
from evosax.algorithms.base import State as EvoSaxState
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState as OptaxOptState

from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptHistoryType(eqx.Module):
    loss: eqx.AbstractVar[jax.Array]
    params: eqx.AbstractVar[PyTree]
    update_step: eqx.AbstractVar[jax.Array]
    fevals: eqx.AbstractVar[jax.Array]
    grads: eqx.AbstractVar[PyTree]
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
OptStates = PyTree[Union[OptaxOptState, EvoSaxState]]

StepFunction = Callable[
    [tuple[InputParameters, OptStates, PRNGKeyArray]],
    tuple[tuple[InputParameters, OptStates, PRNGKeyArray], dict],
]

StopFunction = Callable[[OptHistoryType, OptStates, int], bool]

InitFunction = Callable[[PyTree], OptaxOptState | EvoSaxState]

LossFunction = Any
