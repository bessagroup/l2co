"""
Module for defining type aliases used throughout the optimization process.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable
from typing import Any, Union

# Third-party
from evosax.algorithms.base import State as EvoSaxState
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState as OptaxOptState

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

InputParameters = PyTree
OptStates = PyTree[Union[OptaxOptState, EvoSaxState]]

StepFunction = Callable[
    [tuple[InputParameters, OptStates, PRNGKeyArray]],
    tuple[tuple[InputParameters, OptStates, PRNGKeyArray], dict]]

InitFunction = Callable[[PyTree], OptaxOptState | EvoSaxState]

LossFunction = Any
