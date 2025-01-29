from pathlib import Path

import hydra
from f3dasm import ExperimentData
from f3dasm.design import Domain
from omegaconf import DictConfig
from utils import FunctionSampler, RunOptimization

from l2co import logger


def pre_process(config: DictConfig):
    domain = Domain.from_yaml(config.domain)
    function_sampler = FunctionSampler(
        seed=config.seed, noise_distribution=config.noise_distribution)
    experiment_data = ExperimentData.from_sampling(
        sampler=function_sampler, domain=domain,
        n_samples=config.n_experiments)

    experiment_data.store()


def process(config: DictConfig):
    run_opt = RunOptimization(config=config)

    experiment_data = ExperimentData.from_file(Path().cwd())
    experiment_data.evaluate(data_generator=run_opt, mode='cluster')


@hydra.main(config_path=".", config_name="config")
def main(config):

    logger.setLevel(config.log_level)

    pre_process(config)
    process(config)


if __name__ == "__main__":
    main()
