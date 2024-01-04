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

# Local
from l2co._logging import logger as logger
from l2co.data_loader import DataLoader
from l2co.trainer import ClassifierConstructor

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

    # Load the data to DataLoaders
    if 'from_experimentdata' in config.data:
        data_full = DataLoader.from_experimentdata(
            config=config,
            experimentdata=config.data.from_experimentdata.training)

        if config.data.from_experimentdata.testing is not None:
            data_train, data_test = data_full.train_test_split(
                test_size=config.test_size)

        else:
            data_train = data_full
            data_test = DataLoader.from_experimentdata(
                config=config,
                experimentdata=config.data.from_experimentdata.testing)

    elif 'from_preprocessed_files' in config.data:
        experimentdata = f3dasm.ExperimentData.from_file(
            config.data.from_preprocessed_files.experimentdata)

        ds = xr.open_dataset(config.data.from_preprocessed_files.ds)
        ds_samples = xr.open_dataset(
            config.data.from_preprocessed_files.ds_samples)

        data_full = DataLoader.from_preprocessed_files(
            config=config,
            experimentdata=experimentdata,
            data=ds,
            data_samples=ds_samples)

        data_train, data_test = data_full.train_test_split(
            test_size=config.test_size)

    # Construct the classifier
    classifier_constructor = ClassifierConstructor(
        config=config, data_train=data_train, data_test=data_test)

    # Evaluate the different classifiers
    data.evaluate(classifier_constructor,
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
