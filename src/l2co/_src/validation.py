"""
Module for validating optimization datasets.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
import xarray as xr

# Local
from .naming import BUDGET

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

BUDGET_STEPS = 1000


def validation_fn(x: xr.Dataset,
                  budget_steps: int = BUDGET_STEPS) -> xr.Dataset:
    x = x.assign_coords({BUDGET: np.linspace(0., 1., x[BUDGET].size)})
    x_interp = x.interp({BUDGET: np.linspace(
        0., 1., budget_steps)}, method='nearest')
    n = budget_steps - 1
    step = 1.0 / (n - 1)
    arr_interp = np.linspace(0., 1., n)
    new_budget_coord = np.concatenate([[-step], arr_interp])

    return x_interp.assign_coords({BUDGET: new_budget_coord})
