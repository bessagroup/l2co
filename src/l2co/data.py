"""Public API for data loading and baseline metrics.

This module is the stable import surface for everything related to
optimization-trajectory datasets. Import from here rather than
``l2co._src.data_loader``. ``scoring_loss`` is the one rule for which
loss series a score is computed from: a noisy task's noise-free series
when present, the observed one otherwise (ADR 0024).
``time_shared_portfolio`` builds a derived schedule, the time-shared
portfolio of stored schedules, from their runs (l2co#115).
"""

from l2co._src.data import (
    DataLoader,
)
from l2co._src.data_validation import create_combined_odata
from l2co._src.performance import performance_profile, scoring_loss
from l2co._src.portfolio import time_shared_portfolio

__all__ = [
    "DataLoader",
    "create_combined_odata",
    "performance_profile",
    "scoring_loss",
    "time_shared_portfolio",
]
