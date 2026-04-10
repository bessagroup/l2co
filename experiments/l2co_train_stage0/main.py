"""l2co_train_stage0 pipeline using f3dasm Pipeline API."""

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
)
from f3dasm.design import Domain
from omegaconf import DictConfig, OmegaConf

import hydra
import l2co
import equinox as eqx
import jax.random as jr
import optax
import math
import ast
import numpy as np
from typing import Any
import bessaplots
import logging
from collections.abc import Iterable
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
from bessaplots import FigureSaver
from sklearn.ensemble import HistGradientBoostingRegressor
from l2co.naming import TASK_ID, REALIZATION, PROBLEM, SCHEDULE
from l2co.plotting import (
    plot_single_r2,
)
from bessaplots import savefig
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


logger = logging.getLogger(__name__)


#  Step 1: Create ExperimentData with tasks, model, and experience placeholders
# =============================================================================


def create_experimentdata(project_dir: Path, config: DictConfig) -> None:
    """Set up ExperimentData with one row per stage0 model."""
    domain = Domain()
    domain.add_parameter(name="model")
    domain.add_parameter(name="name")

    input_data = [
        {"model": v, "name": v["_target_"].split(".")[-1]}
        for v in OmegaConf.to_container(config.stage0_models, resolve=True)
    ]

    experiment_data = ExperimentData(domain=domain, input_data=input_data)
    experiment_data.store(project_dir)


# =============================================================================


class Stage0Trainer(DataGenerator):
    def __init__(
        self,
        path_train: str,
        path_test: str,
        t: float,
        t_end: float,
        schema: dict[str, Any],
        optimizers: Iterable[OptimizationStep],
        plot_strategy: bool = False,
    ):
        self.path_train = path_train
        self.path_test = path_test
        self.t = t
        self.t_end = t_end
        self.schema = schema
        self.optimizers = optimizers
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
                store_function=FigureSaver.store,
                load_function=FigureSaver.load,
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

        if self.plot_strategy:
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
            bessaplots.savefig(fig=fig, path=f"./ertd_profile_{id}")

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


#                                                            Building pipeline
# =============================================================================


def build_pipeline(config: DictConfig) -> Pipeline:
    """Construct the l2co_train_stage0 pipeline."""
    n_models = len(config.stage0_models)

    trainer = Stage0Trainer(
        path_train=config.path_train,
        path_test=config.path_test,
        t=config.t,
        t_end=config.t_end,
        schema=config.tasks.schema,
        optimizers=config.optimizers,
        plot_strategy=config.plot_strategy,
    )

    return Pipeline(
        name="l2co_train_stage0",
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


@hydra.main(config_path="../../conf", config_name="config_l2co_train_stage0")
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
