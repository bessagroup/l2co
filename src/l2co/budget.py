"""
Budget functions: how many iterations a run takes.

``fixed_budget`` and ``based_on_dimensionality`` are iteration budgets.
``evaluations_based_on_dimensionality`` gives every run the same number
of function evaluations and needs the built optimizer to turn them into
iterations (``docs/adr/0026``). ``evaluation_budget`` returns that
number of evaluations, B, and needs no optimizer.
"""

#                                                                       Modules
# =============================================================================

# Local
from l2co._src.budget import (
    based_on_dimensionality,
    evaluation_budget,
    evaluations_based_on_dimensionality,
    fixed_budget,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================
#
# =============================================================================

__all__ = [
    "fixed_budget",
    "based_on_dimensionality",
    "evaluations_based_on_dimensionality",
    "evaluation_budget",
]
