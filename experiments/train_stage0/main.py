from f3dasm._src.design import domain
import hydra
from jax._src.config import config
from omegaconf import OmegaConf
import l2co
import equinox as eqx
import jax.random as jr
import optax
import math
import numpy as np
import logging
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
    plot_r2,
)

from f3dasm.hydra_tools import update_config_with_experiment_sample
from f3dasm import ExperimentData
from f3dasm.design import Domain
from l2co._src.visualize.bbob_plot import _add_eval_line
from stage0 import Stage0Trainer
from f3dasm import create_sampler

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def main(config):
    domain = Domain()
    domain.add_parameter(name="model")
    domain.add_parameter(name="name")

    input_data = [
        {"model": v, "name": v["_target_"].split(".")[-1]}
        for v in OmegaConf.to_container(config.models, resolve=True)
    ]

    experiment_data = ExperimentData(domain=domain, input_data=input_data)

    experiment_data.store()

    data_generator = Stage0Trainer(config)
    experiment_data = data_generator.call(
        data=experiment_data, mode="cluster", pass_id=True
    )


if __name__ == "__main__":
    main()
