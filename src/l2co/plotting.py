"""
Plotting module
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.visualize.bbob_plot import (
    ertd_per_category,
    ertd_per_dimension,
    ertd_per_function_raster,
)
from ._src.visualize.colors import get_schedule_styles
from ._src.visualize.plotting import (
    FigureSaver,
    plot_2d_loss_landscape_only,
    plot_2d_loss_landscape_slice_grid,
    plot_3d_loss_landscape_only,
    plot_ertd,
    plot_ertd_profile,
    plot_r2,
    plot_single_r2,
    savefig,
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
    "FigureSaver",
    "plot_r2",
    "plot_single_r2",
    "plot_2d_loss_landscape_only",
    "plot_2d_loss_landscape_slice_grid",
    "plot_3d_loss_landscape_only",
    "get_schedule_styles",
    "ertd_per_category",
    "ertd_per_dimension",
    "ertd_per_function_raster",
    "savefig",
]
