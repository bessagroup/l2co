"""
Plotting module
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.plotting import (
    OPTIMIZER_COLORS,
    OPTIMIZER_LINESTYLES,
    change_colors_and_style,
    format_plot,
    plot_2d_loss_landscape_only,
    plot_2d_loss_landscape_slice_grid,
    plot_2d_trajectories,
    plot_3d_loss_landscape_only,
    plot_actions,
    plot_ertd,
    plot_ertd_profile,
    plot_perf_profile,
    plot_pp,
    plot_rtd,
    plot_rtd_profile,
    plot_trajectories,
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
    "OPTIMIZER_COLORS",
    "OPTIMIZER_LINESTYLES",
    "change_colors_and_style",
    "format_plot",
    "plot_2d_loss_landscape_only",
    "plot_2d_loss_landscape_slice_grid",
    "plot_2d_trajectories",
    "plot_3d_loss_landscape_only",
    "plot_actions",
    "plot_ertd",
    "plot_ertd_profile",
    "plot_perf_profile",
    "plot_pp",
    "plot_rtd",
    "plot_rtd_profile",
    "plot_trajectories",
    "set_size",
]
