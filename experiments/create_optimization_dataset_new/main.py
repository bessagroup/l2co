from pathlib import Path
from time import sleep

import hydra
import numpy as np
from f3dasm import ExperimentData
from f3dasm import logger as f3dasm_logger
from f3dasm.design import Domain
from hydra.utils import instantiate
from omegaconf import DictConfig

from l2co import logger
from l2co._src.tasks.benchmarkfunctions.optimizerschedule import (
    OptimizationSchedule, OptimizationStep, get_combination)
from l2co._src.tasks.benchmarkfunctions.run_optimization import RunOptimization
from l2co._src.tasks.benchmarkfunctions.task import Task


def pre_process(config: DictConfig):

    # Task
    experiment_data_tasks = ExperimentData.from_yaml(config.tasks)

    task_fn = instantiate(config.tasks.data_generator, _partial_=True)

    if config.tasks.task_kwargs is None:
        task_kwargs = {}
    else:
        task_kwargs = config.tasks.task_kwargs

    experiment_data_tasks.evaluate(
        data_generator=task_fn, output_names=['task'],
        **task_kwargs)

    # Schedule
    domain_schedules = Domain()
    steps = [OptimizationStep.from_yaml(c) for c in config.optimizers]
    domain_schedules.add_category(name='optimizer_1', categories=steps)
    domain_schedules.add_category(name='optimizer_2', categories=steps)

    experiment_data_schedules = ExperimentData.from_sampling(
        sampler='grid', domain=domain_schedules)

    experiment_data_schedules.evaluate(
        data_generator=get_combination,
        output_names=['optimizer schedule'],
        switch=config.optimization.switch,
        total_evals=config.optimization.n_evals)

    # Grid sample tasks + schedules
    domain_grid_sampling = Domain()

    _, df_task = experiment_data_tasks.to_pandas(
        keep_references=False)
    _, df_optimizer = experiment_data_schedules.to_pandas(
        keep_references=False)

    domain_grid_sampling.add_category(
        name='task',
        categories=df_task['task'].to_list())
    domain_grid_sampling.add_category(
        name='optimizer schedule',
        categories=df_optimizer['optimizer schedule'].to_list())

    experiment_data_combined = ExperimentData.from_sampling(
        sampler='grid', domain=domain_grid_sampling)

    # if applicable, select a subset of all combinations
    if config.fraction_combinations is not None:
        rng = np.random.default_rng(42)
        indices = rng.integers(
            len(experiment_data_combined),
            size=int(
                len(experiment_data_combined) * config.fraction_combinations))
        experiment_data_combined = experiment_data_combined[indices]

        # Save everything
    domain_combined = Domain()
    domain_combined.add_parameter(
        name='task', to_disk=True,
        store_function=Task.save,
        load_function=Task.load)

    domain_combined.add_parameter(
        name='optimizer schedule',
        to_disk=True,
        store_function=OptimizationSchedule.save,
        load_function=OptimizationSchedule.load)

    domain_combined.add_output(
        name='loss history',
        to_disk=True,
    )

    experiment_data_combined.domain = domain_combined

    for id, es in experiment_data_combined:
        experiment_data_combined.store_experimentsample(es, id)

    experiment_data_combined.store()


def process(config: DictConfig):
    logger.info("== PROCESS ==")

    project_dir = Path().cwd()

    # Retrieve the ExperimentData object
    max_tries = 500
    tries = 0

    while tries < max_tries:
        try:
            logger.info(f"Open ExperimentData at {project_dir}")
            experiment_data = ExperimentData.from_file(project_dir)
            break  # Break out of the loop if successful
        except FileNotFoundError:
            tries += 1
            logger.info(f"FileNotFoundError ({project_dir}), sleeping!")
            sleep(10)

    if tries == max_tries:
        logger.error(
            f"Could not open ExperimentData after {max_tries} attempts.")
        raise FileNotFoundError(f"Could not open ExperimentData after "
                                f"{max_tries} attempts.")

    logger.debug("Loaded experiment_data")

    run_opt = RunOptimization(
        n_realizations=config.optimization.n_realizations,
        seed=config.seed,
        bounded=(config.optimization.lower_bound,
                 config.optimization.upper_bound),
        include_input=False,
        clear_jax_cache=True,
    )

    experiment_data.evaluate(data_generator=run_opt, mode='cluster')

    if config.mode == 'sequential':
        # Store the ExperimentData to a csv file
        experiment_data.store()


@hydra.main(config_path="./conf", config_name="config")
def main(config):

    logger.setLevel(config.log_level)
    f3dasm_logger.setLevel(config.log_level)

    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_process(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_process(config)
        process(config)

    else:
        sleep(0.1*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


if __name__ == "__main__":
    main()
