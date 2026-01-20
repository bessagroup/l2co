# Standard
import logging
from dataclasses import replace
from pathlib import Path
from time import sleep

# Third-party
import hydra
from f3dasm import Block, ExperimentData, datagenerator
from f3dasm.design import Domain, grid
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local
from l2co import OptimizationStep, Task

# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


@hydra.main(config_path="../conf", config_name="config")
def main(config: DictConfig):
    experiment_data_tasks = ExperimentData.from_yaml(
        config.tasks.experimentdata
    )

    experiment_data_tasks = experiment_data_tasks.set_project_dir("./tasks")

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
        lambda row: replace(
            row, reference=Path("../tasks/experiment_data") / row.reference
        )
    )
    ds_optimizer = df_optimizer["optimizer"].apply(
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

    experiment_data_combined.store()


if __name__ == "__main__":
    main()
