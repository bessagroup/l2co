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
# Lowest *noise-free* loss per step, next to ``OUTPUT_MIN`` (the lowest
# observed one) for a run on a noisy task; absent for a noise-free task.
# Noisy tasks are scored on it (docs/adr/0024).
OUTPUT_MIN_TRUE = "output_min_true"
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

# Gap storage and the evaluation budget (docs/adr/0027). ``F_STAR`` is the
# per-task f* (the task's ``global_min``, float64) the stored losses are
# relative to, both a ``(taskID,)`` store variable and a raw netCDF
# attribute; ``VALUE_KIND`` is the raw attribute saying whether a raw
# file holds the gap (``"gap"``) or the loss (``"raw"``, no finite f*).
# ``EVALUATION_BUDGET`` is the per-task B a store's cells were checked
# against.
F_STAR = "f_star"
VALUE_KIND = "value_kind"
EVALUATION_BUDGET = "evaluation_budget"

# Policy-artifact provenance (see l2co._src.databank.provenance).
# ``PROVENANCE`` doubles as the loss-history dataset attribute the
# evaluation pipeline attaches the record under and the ``(schedule,)``
# store variable it lands in; ``IN_TRAINING`` is the derived
# ``(taskID, schedule)`` mask (1/0 for policy schedules, NaN for
# static optimizers).
PROVENANCE = "provenance"
IN_TRAINING = "in_training"

# Performance profiles (docs/adr/0021). ``POSITION`` is a ladder position
# -- 0 the easiest stored target, 1 the hardest -- and ``RATIO`` the
# ratio axis of the profile curve.
POSITION = "position"
RATIO = "ratio"
