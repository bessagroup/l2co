"""
Plotting module
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.plotting import (OPTIMIZER_COLORS, format_plot,
                            plot_2d_loss_landscape_only, plot_2d_trajectories,
                            plot_3d_loss_landscape_only, plot_perf_profile,
                            plot_rtd_profile, plot_trajectories, set_size)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


__all__ = ["set_size",
           "plot_perf_profile",
           "plot_rtd_profile",
           "plot_trajectories",
           "OPTIMIZER_COLORS",
           "format_plot",
           "plot_2d_loss_landscape_only",
           "plot_2d_trajectories",
           "plot_3d_loss_landscape_only",
           ]
