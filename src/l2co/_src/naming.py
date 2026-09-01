"""
Module for naming conventions and constants used in the L2CO package.
"""

#                                                                       Modules
# =============================================================================

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# Dimensions and coordinates
PROBLEM = "problem"
OUTPUT_DIM = "output_dim"
INPUT_DIM = "input_dim"
SCHEDULE = "schedule"
BUDGET = "budget"
KIND = "kind"
FEATURE = "feature"
GLOBAL_MIN = "global_min"
TASK_ID = "taskID"
REALIZATION = "realization"
OPT_CHOICE = "optimizer_choice"
OPT_HISTORY = "optimizer_history"

# Variables
OUTPUT_MEAN = "output_mean"
INPUT_MEAN = "input_mean"
OUTPUT_STD = "output_std"
OUTPUT_MIN = "output_min"
OPT_STEPS = "opt_steps"
UPDATE_STEP = "update_step"
TASK_LABELS = "task_labels"
TRAJECTORY = "trajectory"
STAIRCASE = "staircase"
PERFORMANCE = "performance"
BATCH = "batch"
FEVALS = "fevals"
COST = "cost"
ITERATIONS = "iterations"
SUCCESS = "success"
TARGET = "target"
SEED = "seed"

# Per-record databank (see l2co._src.databank). ``METRIC`` /
# ``TARGET_VALUE`` mirror the string keys already emitted by
# ``l2co._src.performance.calculate_ert``; ``OPTIMIZATIONSTEP`` is the
# JSON-recipe data variable stored in ``optimizer/<hash>.nc``.
METRIC = "metric"
TARGET_VALUE = "target_value"
OPTIMIZATIONSTEP = "optimizationstep"

# Policy-artifact provenance (see l2co._src.databank.provenance).
# ``PROVENANCE`` doubles as the loss-history dataset attribute the
# evaluation pipeline attaches the record under and the ``(schedule,)``
# store variable it lands in; ``IN_TRAINING`` is the derived
# ``(taskID, schedule)`` mask (1/0 for policy schedules, NaN for
# static optimizers).
PROVENANCE = "provenance"
IN_TRAINING = "in_training"
