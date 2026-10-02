"""Public API for optimization-history state objects.

``HistoryState`` is the live trajectory accumulator passed through the JAX
update loop; ``OptHistory`` is the post-hoc xarray view of a finished run;
``BatchState`` is the mini-batch iterator used during training.
"""

from l2co_optimizers import OptHistory

from l2co._src.batching import BatchState
from l2co._src.history_state import HistoryState

__all__ = [
    "BatchState",
    "HistoryState",
    "OptHistory",
]
