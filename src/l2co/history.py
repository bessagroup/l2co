"""Public API for optimization-history state objects.

``HistoryState`` is the live trajectory accumulator passed through the JAX
update loop; ``OptHistory`` is one generation's record; ``BatchState`` is
the mini-batch iterator the loop draws data from. All three live in
``l2co_optimizers`` with the run loop (ADR 0017) and are re-exported here.
The ``history_to_*`` functions turn a ``HistoryState`` into an xarray
``Dataset`` or a ``DataLoader``.
"""

from l2co_optimizers import BatchState, HistoryState, OptHistory

from l2co._src.history_state import (
    history_to_dataloader,
    history_to_xarray,
    history_to_xarray_realizations,
)

__all__ = [
    "BatchState",
    "HistoryState",
    "OptHistory",
    "history_to_dataloader",
    "history_to_xarray",
    "history_to_xarray_realizations",
]
