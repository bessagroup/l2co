"""Deprecated module path. Use ``l2co._src.optimization_dynamics`` instead."""

from warnings import warn

from l2co._src.optimization_dynamics import *  # noqa: F401, F403

warn(
    "l2co._src.optimizationdynamics is deprecated; import from "
    "l2co._src.optimization_dynamics instead.",
    DeprecationWarning,
    stacklevel=2,
)
