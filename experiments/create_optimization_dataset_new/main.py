# Standard
import logging
import socket
from dataclasses import replace
from pathlib import Path
from time import sleep

# Third-party
import hydra
from f3dasm import Block, ExperimentData, datagenerator
from f3dasm.design import Domain, grid
from hydra.utils import instantiate
from jax.extend.backend import get_backend
from mpi4py import MPI
from omegaconf import DictConfig

# Local
from l2co import OptimizationStep, RunOptimization, Task
from l2co._src.tasks.continue_from_path import retrieve_tasks

# =============================================================================

logger = logging.getLogger(__name__)
f3dasm_logger = logging.getLogger("f3dasm")

# =============================================================================


@hydra.main(config_path="./conf", config_name="config")
def main(config):
    if config.mode == "mpi":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        f3dasm_logger.setLevel(config.log_level)

        f3dasm_logger.info(
            f"[{rank}/{size}] Running on {socket.gethostname()}"
        )

        if rank == 0:
            pre_process(config)
            comm.bcast("ready", root=0)  # Signal other ranks to start
            process(config)
        else:
            signal = comm.bcast(None, root=0)  # Ensure all ranks receive
            if signal == "ready":
                sleep(2)
                process(config)
    elif config.mode in ["cluster", "sequential"]:
        pre_process(config)
        process(config)


def pre_process(config: DictConfig):
    if config.project_job is None:
        create_experimentdata(config)
    else:
        project_dir = Path(config.project_dir) / str(config.project_job)
        if not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory {project_dir} does not exist."
            )

        logger.info(f"Loaded experiment_data from {project_dir}")

        experiment_data = ExperimentData.from_file(
            project_dir=project_dir, wait_for_creation=True, max_tries=500
        )

        in_progress_indices = experiment_data.select_with_status(
            "in_progress"
        ).index

        experiment_data.mark(
            indices=in_progress_indices, status="open", in_place=True
        )

        experiment_data.store(project_dir=project_dir)


def create_experimentdata(config: DictConfig):
    logger.info("== PRE PROCESS ==")

    # Task
    if "retrieve_tasks" in config.tasks:
        experiment_data_tasks = retrieve_tasks(
            experiment_data=ExperimentData.from_file(
                project_dir=Path(config.project_dir) / str(config.tasks.path)
            )
        )

    else:
        experiment_data_tasks = ExperimentData.from_yaml(
            config.tasks.experimentdata
        )

        experiment_data_tasks = experiment_data_tasks.set_project_dir(
            "./tasks"
        )

        experiment_data_tasks.domain.add_output(
            name="task",
            to_disk=True,
            store_function=Task.save,
            load_function=Task.load,
        )

        sampler = Block.from_yaml(
            init_config=config.tasks.sampler,
            call_config=config.tasks.sampler_kwargs,
        )

        sampler.arm(experiment_data_tasks)
        experiment_data_tasks = sampler.call(data=experiment_data_tasks)

        task_fn = datagenerator(
            output_names="task", domain=experiment_data_tasks.domain
        )(instantiate(config.tasks.data_generator, _partial_=True))

        if config.tasks.task_kwargs is None:
            task_kwargs = {}
        else:
            task_kwargs = config.tasks.task_kwargs

        experiment_data_tasks = task_fn.call(
            data=experiment_data_tasks, **task_kwargs
        )

        experiment_data_tasks.store()

    steps = [
        {"optimizer": OptimizationStep.from_yaml(opt)}
        for opt in config.optimizers
    ]

    domain_step = Domain()

    domain_step.add_parameter(
        name="optimizer",
        to_disk=True,
        store_function=OptimizationStep.save,
        load_function=OptimizationStep.load,
    )

    experiment_data_schedules = ExperimentData(
        domain=domain_step, input_data=steps, project_dir="./schedules"
    )

    experiment_data_schedules.store()

    # Grid sample tasks + schedules
    domain_grid_sampling = Domain()

    _, df_task = experiment_data_tasks.to_pandas(keep_references=True)
    df_optimizer, _ = experiment_data_schedules.to_pandas(keep_references=True)

    ds_task = df_task["task"].apply(
        # lambda row: Path("../tasks/experiment_data") / row
        lambda row: replace(
            row, reference=Path("../tasks/experiment_data") / row.reference
        )
    )
    ds_optimizer = df_optimizer["optimizer"].apply(
        # lambda row: Path("../schedules/experiment_data") / row
        lambda row: replace(
            row, reference=Path("../schedules/experiment_data") / row.reference
        )
    )

    domain_grid_sampling.add_category(
        name="task", categories=[s.reference for s in ds_task.to_list()]
    )
    domain_grid_sampling.add_category(
        name="optimizer schedule",
        categories=[s.reference for s in ds_optimizer.to_list()],
    )

    experiment_data_combined = grid().call(
        data=ExperimentData(domain=domain_grid_sampling)
    )

    # Save everything
    domain_combined = Domain()
    domain_combined.add_parameter(
        name="task",
        to_disk=True,
        store_function=Task.save,
        load_function=Task.load,
    )

    domain_combined.add_parameter(
        name="optimizer schedule",
        to_disk=True,
        store_function=OptimizationStep.save,
        load_function=OptimizationStep.load,
    )

    domain_combined.add_output(
        name="loss history",
        to_disk=True,
    )

    experiment_data_combined.domain = domain_combined

    # for id, es in experiment_data_combined:
    #     es.store_experimentsample_references(experiment_data_combined.domain)

    experiment_data_combined.store()


def process(config: DictConfig):
    logger.info("== PROCESS ==")

    # Check if the optimization is box constraint
    if "bounded" in config.tasks:
        bounded = (
            config.tasks.bounded.lower_bound,
            config.tasks.bounded.upper_bound,
        )
    else:
        bounded = (None, None)

    run_opt = RunOptimization(
        sampler=instantiate(config.samplers),
        budget_fn=instantiate(config.budgets),
        seed=int(config.seed),
        n_realizations=config.n_realizations,
        bounded=bounded,
        clear_jax_cache=config.clear_jax_cache,
        verbose=config.verbose,
        save_input_parameters=config.save_input_parameters,
    )

    if config.project_job is None:
        project_dir = Path().cwd()
    else:
        project_dir = Path(config.project_dir) / str(config.project_job)

    if not project_dir.exists():
        raise FileNotFoundError(
            f"Project directory {project_dir} does not exist."
        )

    logger.info(f"Loaded experiment_data from {project_dir}")

    experiment_data = ExperimentData.from_file(
        project_dir=project_dir, wait_for_creation=True, max_tries=500
    )

    if config.mode == "sequential":
        experiment_data = run_opt.call(data=experiment_data, mode="sequential")
        experiment_data.store()

    elif config.mode == "mpi":
        _ = run_opt.call(data=experiment_data, mode="mpi", comm=comm)

    elif config.mode == "cluster":
        _ = run_opt.call(
            data=experiment_data,
            mode="cluster",
            wait_for_creation=True,
            max_tries=500,
        )


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

print(f"[Rank {rank}/{size}] Running on {hostname}")

print(f"[Rank {rank}/{size}] Jax backend used: {get_backend().platform}")

main()
