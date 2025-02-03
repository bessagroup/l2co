from pathlib import Path

import hydra
from f3dasm import ExperimentData
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

    experiment_data_combined.domain = domain_combined

    for id, es in experiment_data_combined:
        experiment_data_combined.store_experimentsample(es, id)

    experiment_data_combined.store()


def process(config: DictConfig):
    run_opt = RunOptimization(
        n_realizations=config.optimization.n_realizations,
        seed=config.seed,
        bounded=(config.optimization.lower_bound,
                 config.optimization.upper_bound),
        include_input=False,
        clear_jax_cache=True,
    )

    experiment_data = ExperimentData.from_file(Path().cwd())
    experiment_data.evaluate(data_generator=run_opt, mode='cluster')


@hydra.main(config_path="./conf", config_name="config")
def main(config):

    logger.setLevel(config.log_level)

    pre_process(config)
    process(config)


if __name__ == "__main__":
    main()
