"""
Wrapper for optimization dynamics data including features and trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .naming import BUDGET, FEATURE
from .task_features import FeatureProcessor


@dataclass
class OptimizationDynamics:
    """Container for optimization dynamics training data.

    Holds processed task features, optimizer features,
    trajectory histories, and target labels ready for
    model training or evaluation.
    """

    x_features_task: xr.DataArray
    x_features_optimizer: xr.DataArray
    x_history: xr.DataArray
    y: xr.DataArray
    feature_processor_task: FeatureProcessor
    feature_processor_optimizer: FeatureProcessor

    def y_to_array(self) -> xr.DataArray:
        """Return target labels as a NumPy array."""
        return self.y.values

    def select(self, problems: xr.DataArray) -> OptimizationDynamics:
        """Select a subset of samples by index.

        Parameters
        ----------
        problems : xr.DataArray
            Indices or boolean mask along the sample dim.

        Returns
        -------
        OptimizationDynamics
            Subset containing only the selected samples.
        """
        return OptimizationDynamics(
            x_features_task=self.x_features_task.isel(
                {"sample": problems}, drop=True
            ),
            x_features_optimizer=self.x_features_optimizer.isel(
                {"sample": problems}, drop=True
            ),
            x_history=self.x_history.isel({"sample": problems}, drop=True),
            y=self.y.isel({"sample": problems}, drop=True),
            feature_processor_task=self.feature_processor_task,
            feature_processor_optimizer=self.feature_processor_optimizer,
        )

    def train_test_split(
        self, train_fraction: float = 0.8, seed: int = 42
    ) -> tuple[OptimizationDynamics, OptimizationDynamics]:
        """Split data into training and test sets.

        Parameters
        ----------
        train_fraction : float, optional
            Fraction of samples for training, by default 0.8.
        seed : int, optional
            Random seed for shuffling, by default 42.

        Returns
        -------
        tuple[OptimizationDynamics, OptimizationDynamics]
            Training and test subsets.
        """
        rng = np.random.default_rng(seed)
        shuffled_problems = rng.permutation(
            self.x_features_task["sample"].size
        )

        # 3. Split the shuffled list
        n_train = int(len(shuffled_problems) * train_fraction)
        train_problems_list = shuffled_problems[:n_train]
        test_problems_list = shuffled_problems[n_train:]

        # 5. Call your select method with the DataArrays
        train_data = self.select(train_problems_list)
        test_data = self.select(test_problems_list)

        return train_data, test_data

    def filter(self):
        """Remove samples containing any NaN values.

        Returns
        -------
        OptimizationDynamics
            Filtered data with NaN-free samples only.
        """
        y_mask = ~self.y.isnull().any("label")
        x_features_task_mask = ~self.x_features_task.isnull().any(FEATURE)
        x_features_optimizer_mask = ~self.x_features_optimizer.isnull().any(
            "optimizer_feature"
        )
        x_history_mask = ~self.x_history.isnull().any([BUDGET, "kind"])

        # combine masks
        combined_mask = (
            y_mask
            & x_features_task_mask
            & x_features_optimizer_mask
            & x_history_mask
        )

        return self.select(combined_mask.values)
