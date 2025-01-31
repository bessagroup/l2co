from itertools import product
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from f3dasm import ExperimentData
from f3dasm.design import Domain
from omegaconf import DictConfig
from utils import RunOptimization

from l2co import logger
from l2co._src.tasks.benchmarkfunctions.optimizerschedule import (
    OptimizationSchedule, OptimizationStep, get_all_combinations)
from l2co._src.tasks.benchmarkfunctions.task import (Task,
                                                     create_benchmark_tasks)


def pre_process(config: DictConfig):

    config.benchmark_functions.fn

    fn = config.benchmark_functions.fn
    dims = config.benchmark_functions.dimensionality
    noises = config.benchmark_functions.noise
    seed = config.seed
    noise_distribution = config.noise_distribution
    n_samples = config.n_experiments

    tasks = create_benchmark_tasks(
        dims, fn, noises, seed, noise_distribution, n_samples)
    schedules = get_all_combinations(steps=[OptimizationStep.from_yaml(
        c) for c in config.optimizers], total_evals=config.n_evals,
        switch=config.switch)

    # Tasks
    domain_tasks = Domain()
    domain_tasks.add_parameter(name='task', to_disk=True,
                               store_function=Task.save,
                               load_function=Task.load)
    experiment_data_tasks = ExperimentData(
        domain=domain_tasks,
        input_data=[{'task': x} for x in tasks])

    # Schedules
    domain_schedules = Domain()
    domain_schedules.add_parameter(
        name='optimizer schedule', to_disk=True,
        store_function=OptimizationSchedule.save,
        load_function=OptimizationSchedule.load)
    experiment_data_schedules = ExperimentData(
        domain=domain_schedules,
        input_data=[{'optimizer schedule': y} for y in schedules])

    df_tasks, _ = experiment_data_tasks.to_pandas(keep_references=True)
    df_schedules, _ = experiment_data_schedules.to_pandas(keep_references=True)

    df_product = pd.DataFrame([x[0] | x[1] for x in product(
        df_tasks.to_dict('records'), df_schedules.to_dict('records'))])

    # Create random integers between 0 and 2**30 in a new column
    rng = np.random.default_rng(seed=seed)
    df_product['seed'] = rng.integers(0, 2**30, size=len(df_product))

    # Combine
    domain_combined = domain_tasks + domain_schedules
    domain_combined.add_int(name='seed', low=0, high=2**30)
    experiment_data = ExperimentData(
        domain=domain_combined, input_data=df_product)

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
