"""Public API for data loading and baseline metrics.

This module is the stable import surface for everything related to
optimization-trajectory datasets. Import from here rather than
``l2co._src.data_loader``.
"""

from l2co._src.data import DataLoader, dump_databank, load_databank
from l2co._src.data_validation import create_combined_odata

__all__ = [
    "DataLoader",
    "create_combined_odata",
    "dump_databank",
    "load_databank",
]
