"""
Main entrypoint of the experiment

Functions
---------

main
    Main script to call
pre_processing
    Pre-processing steps
process
    Main process to execute
"""

#
#                                                                       Modules
# =============================================================================

# Standard
import logging
from pathlib import Path
from time import sleep
from typing import Optional

# Third-party
import f3dasm
import hydra
import numpy as np
import pandas as pd
from config import Config
from f3dasm import logger as f3dasm_logger
from f3dasm._src.datageneration.functions.function_factory import \
    FUNCTION_MAPPING
from f3dasm.design import Domain
from hydra.core.config_store import ConfigStore
from run_opt import BenchmarkOptimizationRoutine

# Local
from l2co import logger
from l2co._src.opfunu_functions import CEC_FUNCTIONS, NAMED_FUNCTIONS
from l2co._src.run_optimization import (_get_all_combinations,
                                        _get_all_combinations_no_switch,
                                        _get_name_from_scheduler)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def pre_processing(config: Config):
    logger.setLevel(logging.DEBUG)

    logger.info("== PRE-PROCESS ==")
    data = f3dasm.ExperimentData.from_yaml(config.experimentdata)
    domain = Domain.from_yaml(config.domain)

    dim_filters = {d: list(filter(
        lambda f: FUNCTION_MAPPING[f.lower()].is_dim_compatible(d), list(
            domain.space['function_name'].categories)))
        for d in domain.space['dimensionality'].categories}

    def sample(dim_filters, domain: Domain,
               n_samples: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed=seed)

        dims = rng.choice(
            domain.space['dimensionality'].categories, size=n_samples)
        function_names = np.array([rng.choice(dim_filters[d]) for d in dims])
        noise = rng.choice(domain.space['noise'].categories, size=n_samples,
                           p=config.noise_distribution)
        seed = rng.integers(
            domain.space['seed'].lower_bound,
            domain.space['seed'].upper_bound, size=n_samples)
        budget = np.array([domain.space['budget'].value] * n_samples)
        return pd.DataFrame(
            [dims, function_names, noise, seed, budget],
            index=['dimensionality', 'function_name', 'noise', 'seed', 'budget']).T

    data = f3dasm.ExperimentData.from_sampling(
        sampler=sample, domain=domain, n_samples=2000, seed=42,
        dim_filters=dim_filters)

    # OUTPUT DATA
    optimizers = [opt_config['name'] for
                  opt_config in config.optimization.optimizers]

    if config.optimization.combinations:
        optimizer_names = [_get_name_from_scheduler(
            schedule) for schedule in _get_all_combinations(optimizers, 100)]
    else:
        optimizer_names = [_get_name_from_scheduler(
            schedule) for schedule in _get_all_combinations_no_switch(
                optimizers, 100)]

    for name in optimizer_names:
        data.add_output_parameter(name=name, is_disk=True, exist_ok=True)

    # add random samples column
    data.add_output_parameter(name='random_samples',
                              is_disk=True, exist_ok=True)

    data.mark_all_in_progress_open()
    data.mark_all_error_open()
    data.mark_all_nan_open()

    # Store the ExperimentData to a csv file
    data.store()
    logger.info("Stored experimentdata")


def post_processing(config: Config):
    logger.info("== POST-PROCESSING ==")

    if 'from_file' in config.experimentdata:
        project_dir = config.experimentdata.from_file

    else:
        project_dir = Path().cwd()

    experimentdata = f3dasm.ExperimentData.from_file(project_dir)

    while True:
        if experimentdata.is_all_finished():
            logger.info("All jobs are finished. Exiting filehandler loop!")
            break

        sleep(5)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """

    FUNCTION_MAPPING.update(NAMED_FUNCTIONS)
    FUNCTION_MAPPING.update(CEC_FUNCTIONS)

    logger.setLevel(config.log_level)
    f3dasm_logger.setLevel(config.log_level)
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)
        post_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)
        post_processing(config)

    else:
        sleep(0.1*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


def process(config: Config):
    """Main script that handles the execution of open jobs

    Parameters
    ----------
    config
        Hydra configuration file object
    """

    logger.info("== PROCESS ==")

    if 'from_file' in config.experimentdata:
        project_dir = config.experimentdata.from_file

    else:
        project_dir = Path().cwd()

    # Retrieve the ExperimentData object
    max_tries = 500
    tries = 0

    while tries < max_tries:
        try:
            logger.info(f"Open ExperimentData at {project_dir}")
            data = f3dasm.ExperimentData.from_file(project_dir)
            break  # Break out of the loop if successful
        except FileNotFoundError:
            tries += 1
            logger.info(f"FileNotFoundError ({project_dir}), sleeping!")
            sleep(10)

    if tries == max_tries:
        logger.error(
            f"Could not open ExperimentData after {max_tries} attempts.")
        raise FileNotFoundError(f"Could not open ExperimentData after "
                                f"{max_tries} attempts.")

    logger.debug("Loaded experimentdata")

    benchmark_optimization_routine = BenchmarkOptimizationRoutine(config)

    data.evaluate(benchmark_optimization_routine,
                  mode=config.mode)

    if config.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store()


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
