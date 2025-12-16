from dataclasses import dataclass
from typing import Any

import xarray as xr


@dataclass
class OptimizationDynamics:
    x: xr.DataArray
    y: xr.DataArray

    def to_tag_list(self) -> list[dict[str, Any]]:
        return [
            {
                str(k): v
                for k, v in zip(self.x.feature.values, xi.values, strict=True)
            }
            for xi in self.x
        ]

    def y_to_array(self) -> xr.DataArray:
        return self.y.values
