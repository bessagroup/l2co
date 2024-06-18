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

# Third-party
import f3dasm
import hydra
from f3dasm import logger as f3dasm_logger

# Local
from l2co import logger
from l2co._src.computervision.datagenerator_api import MNIST
from l2co._src.data_loader import create_entire_dataset

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def pre_processing(config):
    logger.setLevel(logging.DEBUG)

    logger.info("== PRE-PROCESS ==")
    data = f3dasm.ExperimentData.from_yaml(config.experimentdata)

    data.mark_all_in_progress_open()
    data.mark_all_error_open()
    data.mark_all_nan_open()

    # Store the ExperimentData to a csv file
    data.store()
    logger.info("Stored experimentdata")


def post_processing(config):
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

    ds = create_entire_dataset(experiment_data=experimentdata)
    ds.to_netcdf("mnist_dataset.nc")


@hydra.main(config_path=".", config_name="config")
def main(config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """

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
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


def process(config):
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

    mnist_routine = MNIST(config)

    data.evaluate(mnist_routine,
                  mode=config.mode)

    if config.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store()


if __name__ == "__main__":
    main()
