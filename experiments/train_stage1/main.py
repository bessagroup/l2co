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
from stage1 import Stage1Trainer
from f3dasm import create_sampler

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def main(config):
    domain = Domain.from_yaml(config.domain)
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
    experiment_data.store()

    data_generator = Stage1Trainer(config)
    experiment_data = data_generator.call(
        data=experiment_data, mode="cluster", pass_id=True
    )


if __name__ == "__main__":
    main()
