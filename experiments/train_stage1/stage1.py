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


class Stage1Trainer(DataGenerator):
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

        key = jr.key(config.seed)
        k_encoder, k_task, k_opt, k_mlp = jr.split(key, 4)

        history_encoder = LSTM_Encoder(
            in_size=2,
            hidden_size=config.stage1_model.history_encoder.hidden_size,
            out_size=config.stage1_model.history_encoder.out_size,
            key=k_encoder,
        )

        mixed_embedding_task = MixedEmbedding.from_feature_processor(
            feature_processor=self.o_train.feature_processor_task,
            embed_dim=config.stage1_model.embedding.dim,
            output_dim=config.stage1_model.embedding.out_size,
            key=k_task,
        )

        mixed_embedding_optimizer = MixedEmbedding.from_feature_processor(
            feature_processor=self.o_train.feature_processor_optimizer,
            embed_dim=config.stage1_model.embedding.dim,
            output_dim=config.stage1_model.embedding.out_size,
            key=k_opt,
        )

        mlp_in_size = (
            config.stage1_model.embedding.out_size * 2
            + config.stage1_model.history_encoder.out_size
        )

        model = MLP.init(
            in_size=mlp_in_size,
            out_size=len(self.optimizers),
            width_size=config.stage1_model.mlp.width_size,
            depth=config.stage1_model.mlp.depth,
            key=k_mlp,
        )

        stage1model = EquinoxStage1Model(
            feature_processor_task=self.o_train.feature_processor_task,
            feature_processor_optimizer=self.o_train.feature_processor_optimizer,
            mixed_embedding_task=mixed_embedding_task,
            mixed_embedding_optimizer=mixed_embedding_optimizer,
            history_encoder=history_encoder,
            model=model,
            optimizers=self.optimizers,
        )

        logger.info("Starting training...")

        stage1model, metrics = stage1model.train(
            data=self.o_train,
            optimizer=optax.adam(config.train.learning_rate),
            num_steps=config.train.num_steps,
            verbose=config.train.verbose,
        )

        y_pred_logits = stage1model.pred_logits(data=self.o_test.filter())

        r2_dict = calculate_r2_dict(self.o_test.filter().y, y_pred_logits)
        for label in r2_dict:
            fig, ax = plot_single_r2(
                y_true=self.o_test.filter().y,
                y_pred=y_pred_logits,
                label=label,
            )

            savefig(
                fig=fig,
                name=f"r2_stage1_{label}_{id}",
                n_side_by_side=4,
                span_columns=True,
            )

        for key, value in r2_dict.items():
            experiment_sample.store(name=key, object=value)

        experiment_sample.store(
            name="model",
            object=stage1model,
            to_disk=True,
            store_function=Stage1Loader.store,
            load_function=Stage1Loader.load,
        )

        if config.plot_strategy:
            y_pred = stage1model.evaluate(data=self.o_test.filter())

            choices = (
                y_pred.unstack("sample")
                .isel({SCHEDULE: 0}, drop=True)
                .stack({PROBLEM: (TASK_ID, REALIZATION)}, create_index=True)
            )
            d_with_l2co_stage1 = self.d_test.create_strategy(
                choices=choices, name="l2co"
            )

            fig, ax = l2co.plotting.plot_ertd_profile(d_with_l2co_stage1)
            l2co.plotting.savefig(fig=fig, name=f"ertd_profile_{id}")

        # print a logging message
        logger.info(
            (
                f"[{id}]Trained Stage1 model:"
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
