from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .naming import PROBLEM
from .task_features import FeatureProcessor


@dataclass
class OptimizationDynamics:
    x_features: xr.DataArray
    x_history: xr.DataArray
    y: xr.DataArray
    feature_processor: FeatureProcessor

    def y_to_array(self) -> xr.DataArray:
        return self.y.values

    def select(self, problems: xr.DataArray) -> OptimizationDynamics:
        return OptimizationDynamics(
            x_features=self.x_features.isel(problem=problems, drop=True),
            x_history=self.x_history.isel(problem=problems, drop=True),
            y=self.y.isel(problem=problems, drop=True),
            feature_processor=self.feature_processor,
        )

    def train_test_split(
        self, train_fraction: float = 0.8, seed: int = 42
    ) -> tuple[OptimizationDynamics, OptimizationDynamics]:
        rng = np.random.default_rng(seed)
        shuffled_problems = rng.permutation(self.x_features[PROBLEM].size)

        # 3. Split the shuffled list
        n_train = int(len(shuffled_problems) * train_fraction)
        train_problems_list = shuffled_problems[:n_train]
        test_problems_list = shuffled_problems[n_train:]

        # 5. Call your select method with the DataArrays
        train_data = self.select(train_problems_list)
        test_data = self.select(test_problems_list)

        return train_data, test_data
