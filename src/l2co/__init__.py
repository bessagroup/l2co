"""
l2co - A package for learning to choose optimizers
==================================================

This package investigates the use of machine learning to automatically select
the most suitable optimizer for a given optimization problem.

Usage
-----
>>> import l2co

Author
------
Martin van der Schelling (m.p.vanderschelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from l2co_tasks import Task

from l2co import (
    budget,
    data,
    features,
    history,
    models,
    naming,
    optimization,
    optimizers,
    plotting,
    sampling,
    tasks,
    validation,
)
from l2co._src import tree_utils
from l2co.data import DataLoader, create_combined_odata
from l2co.features import (
    FeatureProcessor,
    MixedEmbedding,
    preprocessing_factory,
)
from l2co.history import BatchState, HistoryState, OptHistory
from l2co.models import (
    MLP,
    AbstractModel,
    EquinoxStage0Model,
    EquinoxStage1Model,
    L2COModel,
    LSTM_Encoder,
    ScikitStage0Model,
    Stage0Loader,
    Stage0Model,
    Stage1Loader,
    Stage1Model,
    mse_loss,
    train_equinox,
)
from l2co.optimization import (
    OptimizationStep,
    RolloutWrapper,
    RunOptimization,
    RunState,
    UpdateClass,
    evaluate,
    reset,
    run,
)
from l2co.optimizers import (
    create_schedules_experimentdata,
    optimizer_mapping,
)
from l2co.validation import normalize_budget_grid, validation_fn

try:
    __version__ = _pkg_version("l2co")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "MLP",
    "AbstractModel",
    "BatchState",
    "DataLoader",
    "EquinoxStage0Model",
    "EquinoxStage1Model",
    "FeatureProcessor",
    "HistoryState",
    "L2COModel",
    "LSTM_Encoder",
    "MixedEmbedding",
    "OptHistory",
    "OptimizationStep",
    "RolloutWrapper",
    "RunOptimization",
    "RunState",
    "ScikitStage0Model",
    "Stage0Loader",
    "Stage0Model",
    "Stage1Loader",
    "Stage1Model",
    "Task",
    "UpdateClass",
    "budget",
    "create_combined_odata",
    "create_schedules_experimentdata",
    "data",
    "evaluate",
    "features",
    "history",
    "models",
    "mse_loss",
    "naming",
    "normalize_budget_grid",
    "optimization",
    "optimizer_mapping",
    "optimizers",
    "plotting",
    "preprocessing_factory",
    "reset",
    "run",
    "sampling",
    "tasks",
    "train_equinox",
    "tree_utils",
    "validation",
    "validation_fn",
]
