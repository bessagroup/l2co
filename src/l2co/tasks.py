"""
Submodule with built-in optimization tasks
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.tasks.benchmark_task import create_bbob_task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

__all__ = [
    "create_bbob_task",
]
