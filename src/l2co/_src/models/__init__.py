"""Neural-network models for optimizer selection.

The package was split out of a single ``models.py`` module. Every public
class is re-exported here so that the qualified name
``l2co._src.models.<Class>`` continues to resolve - this is important for
pickled checkpoints, which store the fully-qualified class path.

Submodules:

* ``base``          - ``AbstractModel`` and the ``T`` TypeVar
* ``components``    - reusable building blocks (``MLP``, ``LSTM_Encoder``)
* ``stage0``        - Stage-0 models (cold-start from task features)
* ``stage1``        - Stage-1 models (trajectory-aware reassessment)
* ``l2co_model``    - combined two-stage ``L2COModel``
* ``training``      - ``mse_loss`` / ``train_equinox`` helpers
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

# Pickle stability: pre-restructure checkpoints reference
# ``l2co._src.models.<Class>`` as the fully-qualified path. Override each
# class's ``__module__`` so pickle continues to look up the class here
# rather than in the new submodule path.
for _cls in (
    AbstractModel,
    MLP,
    LSTM_Encoder,
    L2COModel,
    Stage0Model,
    ScikitStage0Model,
    EquinoxStage0Model,
    Stage0Loader,
    Stage1Model,
    EquinoxStage1Model,
    Stage1Loader,
):
    _cls.__module__ = __name__
del _cls

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
