"""``L2COModel`` - combined two-stage inference entry point."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from l2co_optimizers import OptimizationStep

from l2co._src.models.stage0 import Stage0Loader, Stage0Model
from l2co._src.models.stage1 import Stage1Loader, Stage1Model

# =============================================================================


class L2COModel(eqx.Module):
    """Combined two-stage model for optimizer selection.

    Main inference entry point that delegates to a stage-0
    model (task features only) for initial selection and a
    stage-1 model (task + optimizer features + history) for
    per-step dynamic selection during optimization.

    Attributes
    ----------
    stage0_model : Stage0Model
        Model for initial optimizer selection.
    stage1_model : Stage1Model
        Model for per-step optimizer selection.
    """

    stage0_model: Stage0Model
    stage1_model: Stage1Model

    def predict_stage0(
        self, task_features: dict[str, Any]
    ) -> OptimizationStep:
        """Select initial optimizer using stage-0 model.

        Parameters
        ----------
        task_features : dict[str, Any]
            Raw task feature dictionary.

        Returns
        -------
        OptimizationStep
            Selected optimizer configuration.
        """
        return self.stage0_model.predict(task_features)

    def predict_stage1(
        self,
        task_features: dict[str, Any],
        optimizer_features: dict[str, Any],
        x_output_mean: jax.Array,
        x_output_min: jax.Array,
        global_min: jax.Array,
    ) -> OptimizationStep:
        """Select optimizer using stage-1 model with history.

        Parameters
        ----------
        task_features : dict[str, Any]
            Raw task feature dictionary.
        optimizer_features : dict[str, Any]
            Raw optimizer feature dictionary.
        x_output_mean : jax.Array
            Mean trajectory values per budget step.
        x_output_min : jax.Array
            Minimum trajectory values per budget step.
        global_min : jax.Array
            Global minimum value observed so far.

        Returns
        -------
        OptimizationStep
            Selected optimizer configuration.
        """
        return self.stage1_model.predict(
            x_feature_task=task_features,
            x_feature_optimizer=optimizer_features,
            x_output_mean=jnp.array(x_output_mean),
            x_output_min=jnp.array(x_output_min),
            global_min=jnp.array(global_min),
        )

    @classmethod
    def load(cls, stage0_model: str, stage1_model: str) -> L2COModel:
        """Load a full L2COModel from saved stage-0 and stage-1 files.

        Parameters
        ----------
        stage0_model : str
            Path to stage-0 model file (without extension).
        stage1_model : str
            Path to stage-1 model file (without extension).

        Returns
        -------
        L2COModel
            Loaded two-stage model.
        """
        stage0_model_loaded = Stage0Loader.load(stage0_model)
        stage1_model_loaded = Stage1Loader.load(stage1_model)
        return cls(
            stage0_model=stage0_model_loaded,
            stage1_model=stage1_model_loaded,
        )

    def swap_optimizers(
        self, optimizers: Iterable[OptimizationStep]
    ) -> L2COModel:
        """Replace optimizers in both stages.

        Parameters
        ----------
        optimizers : Iterable[OptimizationStep]
            New optimizer configurations.

        Returns
        -------
        L2COModel
            New instance with updated optimizers in both
            stage-0 and stage-1 models.
        """
        new_stage0 = self.stage0_model.swap_optimizers(optimizers)
        new_stage1 = self.stage1_model.swap_optimizers(optimizers)
        return L2COModel(
            stage0_model=new_stage0,
            stage1_model=new_stage1,
        )

    def with_feature_impute(self, impute: dict[str, Any]) -> L2COModel:
        """Fill task-tag features neither stage's checkpoint schema
        finds in a foreign task set's tag.

        Lets an already-trained model run on a task set that omits a
        feature its schema names -- e.g. no ``separable`` tag -- by
        asserting the given value instead of raising ``KeyError``. See
        :meth:`Stage0Model.with_feature_impute` /
        :meth:`Stage1Model.with_feature_impute` and
        ``FeatureProcessor.with_imputation`` for the mechanism and its
        caveats (the fill is asserted, not measured, and nothing
        downstream records which features were invented).

        Parameters
        ----------
        impute : dict[str, Any]
            Feature name to fill value, applied to both stages'
            ``feature_processor_task``.

        Returns
        -------
        L2COModel
            New instance with both stages' task processor filled.
        """
        new_stage0 = self.stage0_model.with_feature_impute(impute)
        new_stage1 = self.stage1_model.with_feature_impute(impute)
        return L2COModel(
            stage0_model=new_stage0,
            stage1_model=new_stage1,
        )
