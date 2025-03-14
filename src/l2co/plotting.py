"""
Plotting module
"""

#                                                                       Modules
# =============================================================================

# Local
from l2co._src.plotting import (OPTIMIZER_COLORS, format_plot,
                                plot_2d_loss_landscape, plot_perf_profile,
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
           "plot_2d_loss_landscape",
           ]
