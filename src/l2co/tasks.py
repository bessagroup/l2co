"""
Submodule with built-in optimization tasks.

This module is a backward-compatibility shim. The task implementations live
in the standalone ``l2co_tasks`` package; prefer importing from there
directly in new code::

    from l2co_tasks import create_bbob_task
"""

#                                                                       Modules
# =============================================================================

# Local
from l2co_tasks import (
    create_bbob_task,
    create_cec2005_task,
    create_gaussian_meta_task,
    create_gaussian_task,
    create_mnist1d_task,
    create_pde_task,
    create_quadratic_task,
    create_spiral_task,
)

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
    "create_cec2005_task",
    "create_gaussian_meta_task",
    "create_gaussian_task",
    "create_mnist1d_task",
    "create_pde_task",
    "create_quadratic_task",
    "create_spiral_task",
]
