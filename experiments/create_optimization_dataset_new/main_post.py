import logging
from pathlib import Path

import hydra
from f3dasm import ExperimentData
from omegaconf import DictConfig

# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


@hydra.main(config_path="./conf", config_name="config")
def main(config: DictConfig):
    project_dir = Path(config.project_dir) / str(config.project_job)

    if not project_dir.exists():
        raise FileNotFoundError(
            f"Project directory {project_dir} does not exist."
        )
    experiment_data = ExperimentData.from_file(
        project_dir=project_dir,
    )

    experiment_data = experiment_data.update_from_experimentssample_json()

    experiment_data.store()

    # Remove all the 'experiment_sample' directory

    sample_directory = project_dir / "experiment_sample"
    for file in sample_directory.glob("*.json"):
        file.unlink()
        logger.info(f"Remove file: {file}")

    # Remove the folder
    sample_directory.rmdir()


if __name__ == "__main__":
    main()
