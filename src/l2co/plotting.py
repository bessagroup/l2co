"""
Plotting module
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.visualize.colors import get_schedule_styles
from ._src.visualize.plotting import (
    plot_2d_loss_landscape_only,
    plot_2d_loss_landscape_slice_grid,
    plot_3d_loss_landscape_only,
    plot_ertd,
    plot_ertd_profile,
    set_size,
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
    "set_size",
    "plot_ertd_profile",
    "plot_ertd",
    "plot_2d_loss_landscape_only",
    "plot_2d_loss_landscape_slice_grid",
    "plot_3d_loss_landscape_only",
    "get_schedule_styles",
]
