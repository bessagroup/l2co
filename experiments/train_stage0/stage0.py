import hydra
import l2co
import equinox as eqx
import jax.random as jr
import optax
import math
import ast
import numpy as np
import logging
from time import perf_counter
from pathlib import Path
from l2co import OptimizationStep
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from l2co._src.models import (
    LSTM_Encoder,
    EquinoxStage1Model,
    MixedEmbedding,
    MLP,
    Stage0Loader,
    Stage1Loader,
    ScikitStage0Model,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from l2co.naming import TASK_ID, REALIZATION, PROBLEM, SCHEDULE
from l2co.plotting import (
    ertd_per_category,
    ertd_per_dimension,
    ertd_per_function_raster,
    plot_single_r2,
    savefig,
)

import xarray as xr
from sklearn.metrics import r2_score

from f3dasm.hydra_tools import update_config_with_experiment_sample
from f3dasm import DataGenerator
from f3dasm import ExperimentSample
from l2co._src.visualize.bbob_plot import _add_eval_line
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Stage0Trainer(DataGenerator):
    def __init__(self, config: DictConfig):
        self.config = config
        self.__post_init__()

    def __post_init__(self):
        _path_train = Path(self.config.project_dir) / str(
            self.config.path_train
        )
        _path_test = Path(self.config.project_dir) / str(self.config.path_test)
        self.d_train = l2co.DataLoader.from_file(
            _path_train / "entire_dataset.odata"
        )
        self.o_train = self.d_train.to_params(
            t=self.config.t,
            t_end=self.config.t_end,
            schema=self.config.tasks.schema,
        )

        self.d_test = l2co.DataLoader.from_file(
            _path_test / "entire_dataset.odata"
        )
        self.o_test = self.d_test.to_params(
            t=self.config.t,
            t_end=self.config.t_end,
            schema=self.config.tasks.schema,
        )

        self.optimizers = [
            OptimizationStep.from_yaml(opt) for opt in self.config.optimizers
        ]

    def execute(self, experiment_sample: ExperimentSample, id: int):
        start_time = perf_counter()
        config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=experiment_sample
        )

        model_str = experiment_sample.get("model")
        model = ast.literal_eval(model_str)

        stage0model = ScikitStage0Model.init(
            feature_processor_task=self.o_train.feature_processor_task,
            model=instantiate(model),
            optimizers=self.optimizers,
        )

        stage0model, _ = stage0model.train(
            data=self.o_train,
        )

        y_pred_logits = stage0model.pred_logits(data=self.o_test.filter())

        r2_dict = calculate_r2_dict(self.o_test.filter().y, y_pred_logits)
        for label in r2_dict:
            fig, ax = plot_single_r2(
                y_true=self.o_test.filter().y,
                y_pred=y_pred_logits,
                label=label,
            )
        experiment_sample.store(
            name=f"r2_plot_stage0_{label}",
            object=l2co.plotting.set_size(
                fig=fig, n_side_by_side=4, span_columns=True
            ),
            to_disk=True,
            store_function=l2co.plotting.FigureSaver.store,
            load_function=l2co.plotting.FigureSaver.load,
        )

        for key, value in r2_dict.items():
            experiment_sample.store(name=key, object=value)

        experiment_sample.store(
            name="model",
            object=stage0model,
            to_disk=True,
            store_function=Stage0Loader.store,
            load_function=Stage0Loader.load,
        )

        if config.plot_strategy:
            y_pred = stage0model.evaluate(data=self.o_test.filter())

            choices = (
                y_pred.unstack("sample")
                .isel({SCHEDULE: 0}, drop=True)
                .stack({PROBLEM: (TASK_ID, REALIZATION)}, create_index=True)
            )
            d_with_l2co_stage0 = self.d_test.create_strategy(
                choices=choices, name="l2co"
            )

            fig, ax = l2co.plotting.plot_ertd_profile(d_with_l2co_stage0)
            l2co.plotting.savefig(fig=fig, name=f"ertd_profile_{id}")

        # print a logging message
        logger.info(
            (
                f"[{id}]Trained Stage0 model: {model_str} "
                f"(Took {perf_counter() - start_time:.2f} seconds)"
            )
        )

        return experiment_sample


def calculate_r2_dict(y_true: xr.DataArray, y_pred: xr.DataArray) -> dict:
    """
    Calculates R2 scores for each label and returns them in a dictionary.

    Parameters:
    y_true (xr.DataArray): True values, coordinate "label" required.
    y_pred (xr.DataArray): Predicted values, coordinate "label" required.

    Returns:
    dict: {label_name: r2_score_value}
    """
    r2_results = {}
    labels = y_true["label"].values

    for label in labels:
        # Extract values for the specific label
        # .values is used to pass numpy arrays to sklearn
        y_t = y_true.sel(label=label).values
        y_p = y_pred.sel(label=label).values

        # Calculate and store the score
        r2_results[str(label)] = r2_score(y_t, y_p)

    return r2_results
