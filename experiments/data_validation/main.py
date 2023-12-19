import logging
from pathlib import Path
from time import perf_counter
from typing import List

import hydra
import numpy as np
import xarray as xr
from f3dasm import ExperimentData

from l2co._logging import logger as l2co_logger


@hydra.main(config_path=".", config_name="config")
def main(config):

    # Create child logger

    l2co_logger.setLevel(config.log_level)
    logger = l2co_logger.getChild(__file__)

    # Configure a separate logger for warnings to a file
    warning_logger = l2co_logger.getChild('warning_logger')
    warning_logger.setLevel(logging.WARNING)
    warning_handler = logging.FileHandler('warning.log')
    warning_logger.addHandler(warning_handler)

    experiment_data_all = ExperimentData.from_file(config.project_dir)

    logger.info(f"Loaded experiment data from {config.project_dir}")

    # Select portion if applicable
    if config.portion is not None:
        experiment_data_all = experiment_data_all.select(range(config.portion))

    experiment_data = experiment_data_all.select_with_status('finished')

    logger.info(
        f"Number of finished jobs: {len(experiment_data)} / {len(experiment_data_all)}")

    _, df_output = experiment_data.to_pandas()

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.apply(lambda x: experiment_data.project_dir / x)

    # don't include the column 'random_samples'
    df_output = df_output.drop(columns=['random_samples'])

    files: List[List[Path]] = df_output.values.tolist()

    # First file
    xarr_file = xr.open_dataset(np.array(files).ravel()[0])
    xarr_experiment = xr.open_mfdataset(files[0])

    for index, experiment in enumerate(files):
        logger.info(
            f"({index+1} / {len(files)}) - Validating experiment {experiment[0].stem}")
        for file in experiment:
            logger.debug(
                f"({index+1} / {len(np.array(files).ravel())}) - Validating file {file}")
            _file = xr.open_dataset(file)

            validation(xarr_file, _file, file, warning_logger)

        _exp = xr.open_mfdataset(experiment)
        validate_experiment(xarr_experiment, _exp, experiment, warning_logger)

    # Open the whole thing:
    t_start = perf_counter()
    logger.info("Opening the whole thing")
    _ = xr.open_mfdataset(
        files, concat_dim=['itemID', 'optimizer'], combine='nested')
    logger.info(
        f"Finished opening the whole thing (Took {perf_counter() - t_start} seconds)")


def validate_experiment(expected_xarr: xr.Dataset, actual_xarr: xr.Dataset, filenames: List[Path], warning_logger):
    if not all(expected_xarr.optimizer == actual_xarr.optimizer):
        warning_logger.warning(f'{filenames} has different optimizers')

    if not all(expected_xarr.output_dim == actual_xarr.output_dim):
        warning_logger.warning(f'{filenames} has different output dimensions')
    # for file in filenames:
    #     _file = xr.open_dataset(file)
    #     validation(expected_xarr, _file, file, warning_logger)


def validation(
        expected_xarr: xr.Dataset,
        actual_xarr: xr.Dataset,
        filename: Path, warning_logger):
    if not expected_xarr.data_vars.keys() == actual_xarr.data_vars.keys():
        warning_logger.warning(f'{filename} has different data variables')
    if not expected_xarr.coords.keys() == actual_xarr.coords.keys():
        warning_logger.warning(f'{filename} has different coordinates')
    if not expected_xarr.attrs.keys() == actual_xarr.attrs.keys():
        warning_logger.warning(f'{filename} has different attributes')

    if not all(expected_xarr.realization == actual_xarr.realization):
        warning_logger.warning(f'{filename} has different realizations')


if __name__ == '__main__':
    main()
