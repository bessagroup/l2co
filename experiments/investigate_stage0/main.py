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

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import sleep

# Third-party
import f3dasm
import hydra
import xarray as xr
from f3dasm import logger as f3dasm_logger
from train_classifier import ClassifierConstructer

# Local
from l2co._logging import logger as logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================


def pre_processing(config):
    """Pre-processing steps

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    data = f3dasm.ExperimentData.from_yaml(config.experimentdata)
    data.set_project_dir(Path().cwd() / config.project_dir)

    data.store()
    logger.info("Stored experiment data")


def process(config):
    """Main process to execute

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    # Retrieve the ExperimentData object
    try:
        data = f3dasm.ExperimentData.from_file(
            Path().cwd() / config.project_dir)
    except FileNotFoundError:
        sleep(60)
        data = f3dasm.ExperimentData.from_file(
            Path().cwd() / config.project_dir)

    if 'from_experimentdata' in config.data:
        # Retrieve the training and testing data
        training_data = f3dasm.ExperimentData.from_file(
            config.data.from_experimentdata.training)

        if config.portion is not None:
            training_data = training_data.select(
                range(config.portion)).select_with_status('finished')

        else:
            training_data = training_data.select_with_status('finished')

        if config.data.from_experimentdata.testing is not None:
            testing_data = f3dasm.ExperimentData.from_file(
                config.data.from_experimentdata.testing)

        else:
            testing_data = None

        # Construct the classifier constructer
        classifier = ClassifierConstructer.from_experimentdata(
            config=config,
            training_data=training_data,
            testing_data=testing_data)

    elif 'from_preprocessed_files' in config.data:
        experimentdata = f3dasm.ExperimentData.from_file(
            config.data.from_preprocessed_files.experimentdata)

        ds = xr.open_dataset(config.data.from_preprocessed_files.ds)
        ds_samples = xr.open_dataset(
            config.data.from_preprocessed_files.ds_samples)

        classifier = ClassifierConstructer.from_preprocessed_files(
            config=config,
            experimentdata=experimentdata,
            ds=ds,
            ds_samples=ds_samples)

    # Evaluate the different classifiers
    data.evaluate(classifier,
                  mode=config.mode)

    if config.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store(Path().cwd() / config.project_dir)


@hydra.main(config_path="configs", config_name="config")
def main(config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    logger.setLevel(config.log_level)
    f3dasm_logger.setLevel(config.log_level)

    logger.info("Starting experiment")

    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)

    else:
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


if __name__ == '__main__':
    main()
