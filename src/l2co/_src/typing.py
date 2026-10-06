"""
Module for defining type aliases used throughout the optimization process.

The optimizer-contract aliases (``InputParameters``, ``StepFunction``,
``StopFunction``, ...) live in :mod:`l2co_optimizers` and are re-exported
here so existing ``l2co._src.typing`` imports keep working. Only
``BudgetFn`` is defined here; the run-loop ``Carry`` moved to
``l2co_optimizers`` with the loop (ADR 0017).
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import Any

# Third-party
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

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "AskFunction",
    "BudgetFn",
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
