"""
Module for validating optimization datasets.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
import xarray as xr

# Local
from l2co._src.naming import BUDGET

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DEFAULT_VALIDATION_BUDGET_STEPS = 1000


def normalize_budget_grid(ds: xr.Dataset, budget_steps: int) -> xr.Dataset:
    """Resample a Dataset onto the canonical L2CO budget grid.

    The source ``BUDGET`` coordinate is first labelled as
    ``linspace(0, 1, ds[BUDGET].size)`` and the data is resampled with
    nearest-neighbour interpolation onto ``budget_steps`` points in
    ``[0, 1]``. The resulting ``BUDGET`` coordinate is then *relabelled*
    (without resampling the data) to
    ``[-step, 0, step', 2*step', ..., 1]`` with
    ``step = step' = 1 / (budget_steps - 2)``: the convention used
    throughout L2CO is that slot 0 carries a "pre-optimization" tag
    (the negative coordinate) and slots 1..budget_steps-1 carry the
    trajectory in equal increments.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a ``BUDGET`` dimension.
    budget_steps : int
        Number of output budget steps. Must be ``>= 3`` so that the
        positive portion ``linspace(0, 1, budget_steps - 1)`` is
        well-defined.

    Returns
    -------
    xr.Dataset
        Dataset with ``BUDGET`` dim of size ``budget_steps`` and the
        canonical relabelled coordinate.

    Raises
    ------
    ValueError
        If ``budget_steps < 3``.
    """
    if budget_steps < 3:
        raise ValueError(
            f"budget_steps must be >= 3, got {budget_steps}; the "
            "trailing linspace(0, 1, budget_steps - 1) needs at "
            "least two points for the step size to be finite."
        )
    ds = ds.assign_coords(
        {BUDGET: np.linspace(0.0, 1.0, ds[BUDGET].size, dtype=np.float64)}
    )
    ds_interp = ds.interp(
        {BUDGET: np.linspace(0.0, 1.0, budget_steps, dtype=np.float64)},
        method="nearest",
    )
    n = budget_steps - 1
    step = 1.0 / (n - 1)
    arr_interp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    new_budget_coord = np.concatenate([[-step], arr_interp])

    return ds_interp.assign_coords({BUDGET: new_budget_coord})


def validation_fn(
    x: xr.Dataset, budget_steps: int = DEFAULT_VALIDATION_BUDGET_STEPS
) -> xr.Dataset:
    """Validate and interpolate optimization dataset to fixed budget steps.

    Thin wrapper around :func:`normalize_budget_grid` with a default
    ``budget_steps`` of :data:`DEFAULT_VALIDATION_BUDGET_STEPS`.

    Parameters
    ----------
    x : xr.Dataset
        Input dataset with budget dimension.
    budget_steps : int, optional
        Number of budget steps for interpolation,
        by default DEFAULT_VALIDATION_BUDGET_STEPS.

    Returns
    -------
    xr.Dataset
        Interpolated dataset with normalized budget coordinates.
    """
    return normalize_budget_grid(x, budget_steps)
