"""Deprecated module path. Use ``l2co._src.optimizer_schedule`` instead.

This shim preserves the historical private import path; new code should
import from the renamed module (or, preferably, from the public facade
``l2co.optimization``).
"""

from warnings import warn

from l2co._src.optimizer_schedule import *  # noqa: F401, F403

warn(
    "l2co._src.optimizerschedule is deprecated; import from "
    "l2co._src.optimizer_schedule (or l2co.optimization) instead.",
    DeprecationWarning,
    stacklevel=2,
)
