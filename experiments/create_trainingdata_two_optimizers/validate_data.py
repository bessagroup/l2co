import logging
from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from f3dasm import ExperimentData

from l2co._logging import logger as l2co_logger


@hydra.main(config_path=".", config_name="data_validation")
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

    experiment_data = experiment_data_all.select_with_status('finished')

    logger.info(
        f"Number of finished jobs: {len(experiment_data)} / {len(experiment_data_all)}")

    _, df_output = experiment_data.to_pandas()

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.apply(lambda x: experiment_data.project_dir / x)

    # don't in clude the column 'random_samples'
    df_output = df_output.drop(columns=['random_samples'])

    files = df_output.values.tolist()

    # First file
    xarr = xr.open_dataset(np.array(files).ravel()[0])

    for index, file in enumerate(np.array(files).ravel()):

        logger.info(
            f"({index+1} / {len(np.array(files).ravel())}) - validating {file}")
        _x = xr.open_dataset(file)

        validation(xarr, _x, file, warning_logger)


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
