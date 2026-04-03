"""
Plotting module
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.visualize import bbob_plot as bbob
from ._src.visualize import cec2005_plot as cec2005
from ._src.visualize.colors import get_schedule_styles
from ._src.visualize.plotting import (
    ertd_per_group,
    generate_report,
    plot_2d_loss_landscape_only,
    plot_2d_loss_landscape_slice_grid,
    plot_3d_loss_landscape_only,
    plot_ertd,
    plot_ertd_profile,
    plot_metrics,
    plot_r2,
    plot_single_r2,
    plot_update_step,
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
    "plot_metrics",
    "plot_ertd",
    "plot_r2",
    "plot_update_step",
    "plot_single_r2",
    "plot_2d_loss_landscape_only",
    "plot_2d_loss_landscape_slice_grid",
    "plot_3d_loss_landscape_only",
    "get_schedule_styles",
    "ertd_per_group",
    "generate_report",
    "bbob",
    "cec2005",
]
