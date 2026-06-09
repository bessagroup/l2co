"""Neural-network models for optimizer selection.

Submodules:

* ``base``          - ``AbstractModel`` and the ``T`` TypeVar
* ``components``    - reusable building blocks (``MLP``, ``LSTM_Encoder``)
* ``stage0``        - Stage-0 models (cold-start from task features)
* ``stage1``        - Stage-1 models (trajectory-aware reassessment)
* ``l2co_model``    - combined two-stage ``L2COModel``
* ``training``      - ``mse_loss`` / ``train_equinox`` helpers

Every public class is re-exported here so the qualified name
``l2co._src.models.<Class>`` still resolves for pickles created before the
package split.
"""

from .base import AbstractModel
from .components import MLP, LSTM_Encoder
from .l2co_model import L2COModel
from .stage0 import (
    EquinoxStage0Model,
    ScikitStage0Model,
    Stage0Loader,
    Stage0Model,
)
from .stage1 import EquinoxStage1Model, Stage1Loader, Stage1Model
from .training import mse_loss, train_equinox

__all__ = [
    "MLP",
    "AbstractModel",
    "EquinoxStage0Model",
    "EquinoxStage1Model",
    "L2COModel",
    "LSTM_Encoder",
    "ScikitStage0Model",
    "Stage0Loader",
    "Stage0Model",
    "Stage1Loader",
    "Stage1Model",
    "mse_loss",
    "train_equinox",
]
