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

# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


@hydra.main(config_path="../conf", config_name="config")
def main(config):
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
    # total_experiments = len(experiment_data)
    # jobs_to_run = list(
    #     range(config.job_number, total_experiments, max_number_of_arrayjobs)
    # )

    jobs_to_run = open_experiments[
        config.job_number :: max_number_of_arrayjobs
    ]

    logger.info(f"Running experiments with indices: {jobs_to_run}")

    for job_idx in jobs_to_run:
        _ = run_opt.call(
            data=experiment_data,
            mode="cluster_array",
            job_number=job_idx,
        )


if __name__ == "__main__":
    main()
