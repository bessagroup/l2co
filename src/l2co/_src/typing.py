"""
Module for defining type aliases used throughout the optimization process.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from abc import abstractmethod

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

# Third-party
from evosax.algorithms.base import State as EvoSaxState
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from optax import OptState as OptaxOptState

if TYPE_CHECKING:
    from .batching import BatchState
    from .opt_history import RecentHistory

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptHistoryType(eqx.Module):
    """Abstract base class for optimization history.

    Defines the interface and required fields for storing
    per-step losses, parameters, gradients, and evaluation
    counters during optimization.
    """

    loss: eqx.AbstractVar[Float[Array, " popsize"]]
    params: eqx.AbstractVar[Float[PyTree, " popsize dim"]]
    grads: eqx.AbstractVar[Float[PyTree, " popsize dim"]]
    update_step: eqx.AbstractVar[Int[Array, " popsize"]]
    fevals: eqx.AbstractVar[Int[Array, " popsize"]]
    iterations: eqx.AbstractVar[Int[Array, " popsize"]]

    @classmethod
    @abstractmethod
    def init(
        cls, params: PyTree, popsize: int, last_step: int
    ) -> OptHistoryType:
        """Initialize the history."""
        raise NotImplementedError

    @abstractmethod
    def flatten(self) -> OptHistoryType:
        """
        Flatten the first and second dimension of the loss,
        params and grad.
        """
        raise NotImplementedError


# =============================================================================

InputParameters = Float[PyTree, " popsize dimensionality"]
OptState = PyTree

StepFunction = Callable[
    [tuple[InputParameters, PyTree, PRNGKeyArray]],
    tuple[tuple[InputParameters, PyTree, PRNGKeyArray], dict],
]

StopFunction = Callable[[OptHistoryType, PyTree, int], bool]

InitFunction = Callable[[PyTree], OptaxOptState | EvoSaxState]

LossFunction = Any

BudgetFn = Callable[[Any], int]

SamplerFunction = Callable[[PRNGKeyArray, PyTree, int], InputParameters]

Carry = tuple[
    InputParameters,
    PyTree,
    PRNGKeyArray,
    "BatchState",
    jax.Array,
    "RecentHistory",
]
