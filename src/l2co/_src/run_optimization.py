"""Deprecated module path. Use ``l2co._src.rollout`` instead.

This shim preserves the historical private import path. New code should
import from ``l2co._src.rollout`` (or, preferably, from the public facade
``l2co.optimization``).
"""

from warnings import warn

from l2co._src.rollout import RolloutWrapper, RunOptimization  # noqa: F401

warn(
    "l2co._src.run_optimization is deprecated; import from "
    "l2co._src.rollout (or l2co.optimization) instead.",
    DeprecationWarning,
    stacklevel=2,
)
