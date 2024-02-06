"""
Main entrypoint of the experiment

Functions
---------

main
    Main script to call
"""

#
#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import perf_counter
from typing import List

# Third-party
import hydra
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
# Local
from setup_logger import logger, warning_logger

from l2co import logger as l2co_logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config):
    # logger, warning_logger = create_loggers()
    logger.setLevel(config.log_level)
    l2co_logger.setLevel(config.log_level)

    experiment_data_all = ExperimentData.from_file(config.project_dir)

    logger.info(f"Loaded experiment data from {config.project_dir}")

    # Select portion if applicable
    if config.portion is not None:
        experiment_data_all = experiment_data_all.select(range(config.portion))

    experiment_data = experiment_data_all.select_with_status('finished')

    logger.info(
        f"Number of finished jobs: "
        f"{len(experiment_data)} / {len(experiment_data_all)}")

    _, df_output = experiment_data.to_pandas()

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.apply(lambda x: experiment_data.project_dir / x)

    df_random_samples = df_output['random_samples']
    files_random_samples: List[Path] = df_random_samples.values.tolist()

    # don't include the column 'random_samples'
    df_output = df_output.drop(columns=['random_samples'])

    files: List[List[Path]] = df_output.values.tolist()

    # Open the first file
    xarr_file = xr.open_dataset(np.array(files).ravel()[0])
    xarr_experiment = xr.open_mfdataset(files[0])

    for index, experiment in enumerate(files):
        logger.info(
            f"({index+1} / {len(files)}) - "
            f"Validating experiment {experiment[0].stem}")
        for file in experiment:
            logger.debug(
                f"({index+1} / {len(np.array(files).ravel())}) "
                f"- Validating file {file}")
            _file = xr.open_dataset(file)

            validation(xarr_file, _file, file)

        _exp = xr.open_mfdataset(experiment)
        validate_experiment(xarr_experiment, _exp, experiment)

    # Open the whole thing:
    t_start = perf_counter()
    logger.info("Opening the whole thing")
    xarr_full = xr.open_mfdataset(
        files,
        concat_dim=[xr.DataArray([int(f[0].stem) for f in files],
                                 dims='itemID'), 'optimizer'],
        combine='nested')
    logger.info(
        f"Finished opening the whole thing "
        f"(Took {perf_counter() - t_start} seconds)")

    t_start = perf_counter()
    logger.info("Opening the whole thing: random samples")
    xarr_full_samples = xr.open_mfdataset(
        files_random_samples,
        concat_dim=xr.DataArray([int(f.stem) for f in files_random_samples],
                                dims='itemID'),
        combine='nested')

    logger.info(
        f"Finished opening the whole thing: random samples "
        f"(Took {perf_counter() - t_start} seconds)")

    if config.select_output_and_combine:
        xarr_only_output = xarr_full.drop('input')
        xarr_only_output = xarr_only_output.drop('input_dim')
        xarr_only_output.load().to_netcdf('entire_dataset.nc')

        xarr_random_all = xarr_full_samples.drop('input')
        xarr_random_all = xarr_random_all.drop('input_dim')
        xarr_random_all.load().to_netcdf('entire_dataset_samples.nc')

        # Store ExperimentData
        experiment_data.store(Path().cwd())


def validate_experiment(expected_xarr: xr.Dataset,
                        actual_xarr: xr.Dataset,
                        filenames: List[Path],
                        ):
    if not all(expected_xarr.optimizer == actual_xarr.optimizer):
        warning_logger.warning(f'{filenames} has different optimizers')

    if not all(expected_xarr.output_dim == actual_xarr.output_dim):
        warning_logger.warning(f'{filenames} has different output dimensions')


def validation(
        expected_xarr: xr.Dataset,
        actual_xarr: xr.Dataset,
        filename: Path):
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
