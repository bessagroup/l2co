# Standard
import logging
from pathlib import Path

# Third-party
import hydra
from f3dasm import ExperimentData
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local
from l2co import RunOptimization
from stage1 import Stage1Trainer

# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config):
    data_generator = Stage1Trainer(config)

    project_dir = Path(config.project_dir) / str(config.project_job)

    if not project_dir.exists():
        raise FileNotFoundError(
            f"Project directory {project_dir} does not exist."
        )

    logger.debug(f"Loaded experiment_data from {project_dir}")

    experiment_data = ExperimentData.from_file(
        project_dir=project_dir,
    )

    open_experiments = experiment_data.select_with_status(
        "open"
    ).index.tolist()

    # Determine which experiments to run
    max_number_of_arrayjobs = 900
    jobs_to_run = open_experiments[
        config.job_number :: max_number_of_arrayjobs
    ]

    logger.info(f"Running experiments with indices: {jobs_to_run}")

    for job_idx in jobs_to_run:
        _ = data_generator.call(
            data=experiment_data,
            mode="cluster_array",
            job_number=job_idx,
            pass_id=True,
        )


if __name__ == "__main__":
    main()
