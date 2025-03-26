# Standard
import socket
from pathlib import Path
from time import sleep

# Third-party
import hydra
import jax
import numpy as np
from f3dasm import Block, ExperimentData, datagenerator
from f3dasm import logger as f3dasm_logger
from f3dasm.design import Domain, grid
from hydra.utils import instantiate
from mpi4py import MPI
from omegaconf import DictConfig

# Local
from l2co import logger
from l2co._src.optimizerschedule import OptimizationSchedule
from l2co._src.tasks.benchmarkfunctions.run_optimization import RunOptimization
from l2co._src.tasks.benchmarkfunctions.task import Task

# =============================================================================


# =============================================================================


@hydra.main(config_path="./conf", config_name="config")
def main(config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger.setLevel(config.log_level)
    f3dasm_logger.setLevel(config.log_level)

    f3dasm_logger.info(f"{jax.device_count()=} - {jax.local_device_count()=}")

    f3dasm_logger.info(f"[{rank}/{size}] Running on {socket.gethostname()}")

    if rank == 0:
        pre_process(config)
        comm.bcast("ready", root=0)  # Signal other ranks to start
        process(config)
    else:
        signal = comm.bcast(None, root=0)  # Ensure all ranks receive
        if signal == "ready":
            sleep(2)
            process(config)


def pre_process(config: DictConfig):
    logger.info("== PRE PROCESS ==")
    # Task
    experiment_data_tasks = ExperimentData.from_yaml(
        config.tasks.experimentdata)

    sampler = Block.from_yaml(init_config=config.tasks.sampler,
                              call_config=config.tasks.sampler_kwargs)

    sampler.arm(experiment_data_tasks)
    experiment_data_tasks = sampler.call(data=experiment_data_tasks)

    task_fn = datagenerator(output_names='task')(
        instantiate(config.tasks.data_generator, _partial_=True)
    )

    if config.tasks.task_kwargs is None:
        task_kwargs = {}
    else:
        task_kwargs = config.tasks.task_kwargs

    experiment_data_tasks = task_fn.call(
        data=experiment_data_tasks,
        **task_kwargs)

    # Schedule
    domain_schedule = Domain()
    for k, v in config.schedules.schedules.items():
        domain_schedule.add_category(name=k, categories=instantiate(v))

    experiment_data_schedules = grid().call(
        data=ExperimentData(domain=domain_schedule))

    schedule_fn = datagenerator(output_names='optimizer schedule')(instantiate(
        config.schedules.data_generation.data_generator, _partial_=True))

    if config.schedules.data_generation.kwargs is None:
        schedules_kwargs = {}
    else:
        schedules_kwargs = config.schedules.data_generation.kwargs

    experiment_data_schedules = schedule_fn.call(
        data=experiment_data_schedules,
        **schedules_kwargs
    )

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

    experiment_data_combined = grid().call(
        data=ExperimentData(domain=domain_grid_sampling))

    # if applicable, select a subset of all combinations
    if 'fraction_combinations' in config.schedules:
        rng = np.random.default_rng(int(config.seed))
        indices = rng.integers(
            len(experiment_data_combined),
            size=int(
                len(experiment_data_combined)
                * config.schedules.fraction_combinations))
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
        n_realizations=config.n_realizations,
        n_evals_per_dim=config.n_evals_per_dim,
        max_iterations=config.max_iterations,
        partition_factor=config.schedules.partition_factor,
        seed=int(config.seed),
        bounded=config.tasks.bounded,
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

print((
    f"[Rank {rank}/{size}] Running on {hostname}")
)

main()
