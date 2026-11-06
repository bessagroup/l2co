"""Public API for data loading and baseline metrics.

This module is the stable import surface for everything related to
optimization-trajectory datasets. Import from here rather than
``l2co._src.data_loader``. ``scoring_loss`` is the one rule for which
loss series a score is computed from: a noisy task's noise-free series
when present, the observed one otherwise (ADR 0024).
"""

from l2co._src.data import (
    DataLoader,
)
from l2co._src.data_validation import create_combined_odata
from l2co._src.performance import performance_profile, scoring_loss

__all__ = [
    "DataLoader",
    "create_combined_odata",
    "performance_profile",
    "scoring_loss",
]
