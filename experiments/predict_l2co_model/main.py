# Standard
import socket
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
from l2co._src.l2co_model import L2CO_model
from l2co._src.tasks.benchmarkfunctions.optimizerschedule import (
    OptimizationSchedule, OptimizationStep, get_combination)
from l2co._src.tasks.benchmarkfunctions.run_optimization import (
    RunOptimizationL2CO, load_schedules, save_schedules)
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
    experiment_data_tasks = ExperimentData.from_yaml(config.tasks)

    task_fn = instantiate(config.tasks.data_generator, _partial_=True)

    if config.tasks.task_kwargs is None:
        task_kwargs = {}
    else:
        task_kwargs = config.tasks.task_kwargs

    experiment_data_tasks.evaluate(
        data_generator=task_fn, output_names=['task'],
        **task_kwargs)

    _, df_task = experiment_data_tasks.to_pandas(
        keep_references=False)

    # Save everything
    domain_combined = Domain()
    domain_combined.add_parameter(
        name='task', to_disk=True,
        store_function=Task.save,
        load_function=Task.load)
    domain_combined.add_output(
        name='loss history',
        to_disk=True,
    )
    domain_combined.add_output(
        name='optimizer schedules',
        to_disk=True,
        store_function=save_schedules,
        load_function=load_schedules,
    )

    experiment_data_combined = ExperimentData(input_data=df_task,
                                              domain=domain_combined)

    experiment_data_combined.store()


def process(config: DictConfig):
    logger.info("== PROCESS ==")

    project_dir = Path().cwd()

    logger.debug("Loaded experiment_data")

    trained_l2co_model = L2CO_model.load(config.l2co_path)

    run_opt = RunOptimizationL2CO(
        trained_l2co_model=trained_l2co_model,
        n_realizations=config.optimization.n_realizations,
        seed=config.seed,
        n_evals_per_switch=config.n_evals_per_switch,
        n_switches=config.n_switches,
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

print((
    f"[Rank {rank}/{size}] Running on {hostname}")
)

main()
