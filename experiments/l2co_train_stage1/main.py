"""l2co_train_stage1 pipeline using f3dasm Pipeline API."""
#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
import hydra
from f3dasm import (
    CollectArrayResults,
    ExperimentData,
    Pipeline,
    SlurmCluster,
    SlurmResources,
    Step,
    create_sampler,
)
from f3dasm.design import Domain
from l2co._src.models import Stage1Loader
from omegaconf import DictConfig

import hydra
import l2co
import equinox as eqx
import jax.random as jr
import optax
import math
import ast
import numpy as np
from typing import Any
from collections.abc import Iterable
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
    plot_single_r2,
    plot_metrics,
)

from bessaplots import FigureSaver, savefig, set_size

import xarray as xr
from sklearn.metrics import r2_score

from f3dasm.hydra_tools import update_config_with_experiment_sample
from f3dasm import DataGenerator
from f3dasm import ExperimentSample
from omegaconf import DictConfig


#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

SUBPLOT_ADJUST_5COL = dict(left=0.22, right=0.95, top=0.92, bottom=0.18)
logger = logging.getLogger(__name__)

#  Step 1: Create ExperimentData with tasks, model, and experience placeholders
# =============================================================================


def create_experimentdata(project_dir: Path, config: DictConfig) -> None:
    """Set up ExperimentData with grid-sampled stage1 model configs."""
    domain = Domain.from_yaml(config.stage1_models)
    domain.add_output(
        name="model",
        to_disk=True,
        store_function=Stage1Loader.store,
        load_function=Stage1Loader.load,
    )
    experiment_data = ExperimentData(domain=domain)
    sampler = create_sampler("grid")
    sampler.arm(experiment_data)
    experiment_data = sampler.call(experiment_data)
    experiment_data.store(project_dir)


# =============================================================================


class Stage1Trainer(DataGenerator):
    def __init__(
        self,
        path_train: str,
        path_test: str,
        t: float,
        t_end: float,
        schema: dict[str, Any],
        seed: int,
        stage1_config: DictConfig,
        optimizers: Iterable[OptimizationStep],
        learning_rate: float,
        num_steps: int,
        verbose: bool = False,
        plot_strategy: bool = False,
    ):
        self.path_train = path_train
        self.path_test = path_test
        self.t = t
        self.t_end = t_end
        self.seed = seed
        self.schema = schema
        self.stage1_config = stage1_config
        self.optimizers = optimizers
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.verbose = verbose
        self.plot_strategy = plot_strategy
        self.__post_init__()

    def __post_init__(self):
        _path_train = Path(str(self.path_train))
        _path_test = Path(str(self.path_test))
        self.d_train = l2co.DataLoader.from_file(
            _path_train / "entire_dataset.odata"
        )
        self.o_train = self.d_train.to_params(
            t=self.t,
            t_end=self.t_end,
            schema=self.schema,
        )

        self.d_test = l2co.DataLoader.from_file(
            _path_test / "entire_dataset.odata"
        )
        self.o_test = self.d_test.to_params(
            t=self.t,
            t_end=self.t_end,
            schema=self.schema,
        )

        self.optimizers = [
            OptimizationStep.from_yaml(opt) for opt in self.optimizers
        ]

    def execute(self, experiment_sample: ExperimentSample, id: int):
        start_time = perf_counter()
        stage1_config = update_config_with_experiment_sample(
            config=self.stage1_config, experiment_sample=experiment_sample
        )

        key = jr.key(self.seed)
        k_encoder, k_task, k_opt, k_mlp = jr.split(key, 4)

        history_encoder = LSTM_Encoder(
            in_size=2,
            hidden_size=stage1_config.history_encoder.hidden_size,
            out_size=stage1_config.history_encoder.out_size,
            key=k_encoder,
        )

        mixed_embedding_task = MixedEmbedding.from_feature_processor(
            feature_processor=self.o_train.feature_processor_task,
            embed_dim=stage1_config.embedding.dim,
            output_dim=stage1_config.embedding.out_size,
            key=k_task,
        )

        mixed_embedding_optimizer = MixedEmbedding.from_feature_processor(
            feature_processor=self.o_train.feature_processor_optimizer,
            embed_dim=stage1_config.embedding.dim,
            output_dim=stage1_config.embedding.out_size,
            key=k_opt,
        )

        mlp_in_size = (
            stage1_config.embedding.out_size * 2
            + stage1_config.history_encoder.out_size
        )

        model = MLP.init(
            in_size=mlp_in_size,
            out_size=len(self.optimizers),
            width_size=stage1_config.mlp.width_size,
            depth=stage1_config.mlp.depth,
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
            optimizer=optax.adam(self.learning_rate),
            num_steps=self.num_steps,
            verbose=self.verbose,
        )

        fig, ax = plot_metrics(metrics)

        experiment_sample.store(
            name=f"metrics",
            object=set_size(
                fig=fig,
                n_side_by_side=5,
                span_columns=True,
                subplot_adjust=SUBPLOT_ADJUST_5COL,
            ),
            to_disk=True,
            store_function=FigureSaver.store,
            load_function=FigureSaver.load,
        )

        y_pred_logits = stage1model.pred_logits(data=self.o_test.filter())

        r2_dict = calculate_r2_dict(self.o_test.filter().y, y_pred_logits)
        for label in r2_dict:
            fig, ax = plot_single_r2(
                y_true=self.o_test.filter().y,
                y_pred=y_pred_logits,
                label=label,
            )

            experiment_sample.store(
                name=f"r2_plot_stage1_{label}",
                object=set_size(
                    fig=fig,
                    n_side_by_side=5,
                    span_columns=True,
                    subplot_adjust=SUBPLOT_ADJUST_5COL,
                ),
                to_disk=True,
                store_function=FigureSaver.store,
                load_function=FigureSaver.load,
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

        if self.plot_strategy:
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

            experiment_sample.store(
                name=f"ertd_profile_{id}",
                object=set_size(fig=fig, n_side_by_side=4, span_columns=True),
                to_disk=True,
                store_function=FigureSaver.store,
                load_function=FigureSaver.load,
            )

        # print a logging message
        logger.info(
            (
                f"[{id}]Trained Stage1 model: "
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


#                                                            Building pipeline
# =============================================================================


def build_pipeline(config: DictConfig) -> Pipeline:
    """Construct the l2co_train_stage1 pipeline."""
    # Pre-compute number of grid-sampled experiments for array_jobs
    domain = Domain.from_yaml(config.stage1_models)
    experiment_data = ExperimentData(domain=domain)
    sampler = create_sampler("grid")
    sampler.arm(experiment_data)
    experiment_data = sampler.call(experiment_data)
    n_models = len(experiment_data)

    trainer = Stage1Trainer(
        path_train=config.path_train,
        path_test=config.path_test,
        t=config.t,
        t_end=config.t_end,
        schema=config.tasks.schema,
        seed=config.seed,
        stage1_config=config.stage1_models,
        optimizers=config.optimizers,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
        verbose=config.verbose,
    )

    return Pipeline(
        name="l2co_train_stage1",
        steps=[
            Step(
                name="create",
                block=create_experimentdata,
                resources=SlurmResources(
                    time="00:30:00",
                    mem="4G",
                    cpus_per_task=1,
                ),
                kwargs={"config": config},
            ),
            Step(
                name="run",
                block=trainer,
                dependency="afterok",
                parallel=True,
                array_jobs=n_models,
                resources=SlurmResources(
                    time="00:15:00",
                    mem="4G",
                    cpus_per_task=1,
                ),
                kwargs={"pass_id": True},
            ),
            Step(
                name="post",
                block=CollectArrayResults(cleanup=True),
                dependency="afterany",
                resources=SlurmResources(
                    time="01:00:00",
                    mem="16G",
                    cpus_per_task=1,
                ),
            ),
        ],
    )


# =============================================================================


@hydra.main(config_path="../../conf", config_name="config_l2co_train_stage1")
def main(config: DictConfig) -> None:
    pipeline = build_pipeline(config)

    cluster = (
        SlurmCluster.from_yaml(config.cluster)
        if config.cluster.enabled
        else None
    )
    rootdir = Path(config.rootdir) if config.rootdir else None

    pipeline.run(mode=config.mode, cluster=cluster, rootdir=rootdir)


if __name__ == "__main__":
    main()
