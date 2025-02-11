# Standard
from pathlib import Path
from time import sleep

# Third-party
import hydra
import numpy as np
from f3dasm import ExperimentData
from f3dasm import logger as f3dasm_logger
from f3dasm.design import Domain
from hydra.utils import instantiate
from mpi4py import MPI
from omegaconf import DictConfig

# Local
from l2co import logger
from l2co._src.tasks.benchmarkfunctions.optimizerschedule import (
    OptimizationSchedule, OptimizationStep, get_combination)
from l2co._src.tasks.benchmarkfunctions.run_optimization import RunOptimization
from l2co._src.tasks.benchmarkfunctions.task import Task

# =============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================================


@hydra.main(config_path="./conf", config_name="config")
def main(config):

    logger.setLevel(config.log_level)
    f3dasm_logger.setLevel(config.log_level)

    if rank == 0:
        logger.info(f"Running Hydra with MPI: {size} processes")
        pre_process(config)
        comm.bcast("ready", root=0)  # Signal other ranks to start
        process(config)
    else:
        # Wait for rank 0 to finish preprocessing
        signal = comm.bcast(None, root=0)
        if signal == "ready":
            sleep(2)
            process(config)


def pre_process(config: DictConfig):
    logger.info("== PRE PROCESS ==")
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
        es.store_experimentsample_references(idx=id)

    experiment_data_combined.store()


def process(config: DictConfig):
    logger.info("== PROCESS ==")

    project_dir = Path().cwd()

    logger.debug("Loaded experiment_data")

    run_opt = RunOptimization(
        n_realizations=config.optimization.n_realizations,
        seed=config.seed,
        bounded=(config.optimization.lower_bound,
                 config.optimization.upper_bound),
        include_input=False,
        clear_jax_cache=True,
    )

    experiment_data = ExperimentData.from_file(project_dir=project_dir,
                                               wait_for_creation=True,
                                               max_tries=500)

    if config.mode == 'sequential':
        experiment_data = run_opt.call(data=experiment_data, mode='sequential')
        experiment_data.store()

    elif config.mode == 'mpi':
        _ = run_opt.call(data=experiment_data, mode='mpi',
                         comm=comm)

    elif config.mode == 'cluster':
        _ = run_opt.call(data=experiment_data, mode='cluster',
                         wait_for_creation=True, max_tries=500,
                         )


if __name__ == "__main__":
    main()
