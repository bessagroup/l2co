"""
Module for defining type aliases used throughout the optimization process.

The optimizer-contract aliases (``InputParameters``, ``StepFunction``,
``StopFunction``, ...) live in :mod:`l2co_optimizers` and are re-exported
here so existing ``l2co._src.typing`` imports keep working. Only the
aliases describing l2co's own rollout machinery are defined here.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Third-party
import jax
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import (
    AskFunction,
    InitFunction,
    InputParameters,
    LossFunction,
    OptHistoryType,
    OptState,
    PopSize,
    SamplerFunction,
    StepFunction,
    StopFunction,
    TaskLike,
    TransferReadFunction,
    TransferWriteFunction,
)

if TYPE_CHECKING:
    from l2co_optimizers import RecentHistory

    from l2co._src.batching import BatchState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "AskFunction",
    "BudgetFn",
    "Carry",
    "InitFunction",
    "InputParameters",
    "LossFunction",
    "OptHistoryType",
    "OptState",
    "PopSize",
    "SamplerFunction",
    "StepFunction",
    "StopFunction",
    "TaskLike",
    "TransferReadFunction",
    "TransferWriteFunction",
]

BudgetFn = Callable[[Any], int]

#: The scan carry of l2co's rollout: ``(params, opt_state, key,
#: batch_state, done, recent_history)``.
Carry = tuple[
    InputParameters,
    PyTree,
    PRNGKeyArray,
    "BatchState",
    jax.Array,
    "RecentHistory",
]
