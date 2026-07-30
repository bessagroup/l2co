"""Public API for feature processing.

Stable import surface for ``FeatureProcessor``, ``MixedEmbedding``, and the
``preprocessing_factory`` used to wire feature pipelines.
"""

from l2co._src.preprocessing import preprocessing_factory
from l2co._src.task_features import (
    DERIVED_FEATURES,
    FeatureProcessor,
    MixedEmbedding,
)

__all__ = [
    "DERIVED_FEATURES",
    "FeatureProcessor",
    "MixedEmbedding",
    "preprocessing_factory",
]
