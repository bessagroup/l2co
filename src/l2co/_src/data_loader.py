"""Deprecated module path. Use ``l2co._src.data`` instead.

This shim preserves the historical private import path. New code should
import from ``l2co._src.data`` (or, preferably, from the public facade
``l2co.data``).
"""

from warnings import warn

from l2co._src.data import (  # noqa: F401
    DataLoader,
    calculate_best_solver,
    calculate_sbs,
    calculate_vbs,
    encode,
    log_relative_improvement,
)

# Some callers (e.g. ``l2co._src.models``) historically imported
# ``OptimizationDynamics`` via ``l2co._src.data_loader``; preserve that path
# so we don't have to touch every caller in the same commit.
from l2co._src.optimization_dynamics import OptimizationDynamics  # noqa: F401

warn(
    "l2co._src.data_loader is deprecated; import from l2co._src.data "
    "(or l2co.data) instead.",
    DeprecationWarning,
    stacklevel=2,
)
