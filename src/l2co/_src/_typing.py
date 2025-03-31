from typing import Callable, Dict, Tuple

from jax import Array
from jaxtyping import PyTree
from optax import OptState

# =============================================================================

ScanFunction = Callable[
    [Tuple[PyTree, OptState, Array, Dict[str, Array]]],
    Tuple[Tuple[PyTree, OptState, Array],
          Dict[str, Array]]]

InitFunction = Callable[[], OptState]

LossFunction = Callable[[PyTree], Array]
