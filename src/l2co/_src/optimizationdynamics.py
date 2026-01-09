from dataclasses import dataclass
from typing import Any

import xarray as xr

from .task_features import FeatureProcessor


@dataclass
class OptimizationDynamics:
    x_features: xr.DataArray
    x_history: xr.DataArray
    y: xr.DataArray
    feature_processor: FeatureProcessor

    def to_tag_list(self) -> list[dict[str, Any]]:
        return [
            {
                str(k): v
                for k, v in zip(
                    self.x_features.feature.values, xi.values, strict=True
                )
            }
            for xi in self.x_features
        ]

    def y_to_array(self) -> xr.DataArray:
        return self.y.values
