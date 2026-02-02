# Standard
import logging
from dataclasses import replace
from pathlib import Path
from omegaconf import OmegaConf
from time import sleep

# Third-party
import hydra
from f3dasm import Block, ExperimentData, datagenerator, create_sampler
from l2co._src.models import Stage1Loader
from f3dasm.design import Domain, grid
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local
from l2co import OptimizationStep, Task

# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config: DictConfig):
    if config.project_job is None:
        logger.info(
            f"Project_job does not exist: Creating new experiment_data..."
        )
        create_experimentdata(config)
    else:
        project_dir = Path(config.project_dir) / str(config.project_job)
        logger.info(
            (f"Continuing from existing experiment_data: {project_dir}...")
        )

        if not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory {project_dir} does not exist."
            )
        experiment_data = ExperimentData.from_file(
            project_dir=project_dir, wait_for_creation=True, max_tries=500
        )
        logger.info(f"Loaded experiment_data from {project_dir}")

        in_progress_indices = experiment_data.select_with_status(
            "in_progress"
        ).index

        experiment_data.mark(
            indices=in_progress_indices, status="open", in_place=True
        )

        experiment_data.store(project_dir=project_dir)

        sleep(10)  # Ensure file system consistency


def create_experimentdata(config: DictConfig):
    domain = Domain()
    domain.add_parameter(name="model")
    domain.add_parameter(name="name")

    input_data = [
        {"model": v, "name": v["_target_"].split(".")[-1]}
        for v in OmegaConf.to_container(config.models, resolve=True)
    ]

    experiment_data = ExperimentData(domain=domain, input_data=input_data)

    experiment_data.store()


if __name__ == "__main__":
    main()
