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
from l2co._src.ela import create_ela_feature_1

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

    # remove all values that say 'ERROR' and 'invalid'
    df_output = df_output.replace('ERROR', np.nan)
    df_output = df_output.replace('invalid', np.nan)
    df_output = df_output.replace('not working', np.nan)

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.applymap(
        lambda x: experiment_data.project_dir / x, na_action='ignore')
    # df_output = df_output.apply(lambda x: experiment_data.project_dir / x)

    df_random_samples = df_output['random_samples']
    files_random_samples: List[Path] = df_random_samples.values.tolist()

    # don't include the column 'random_samples'
    df_output = df_output.drop(columns=['random_samples'])

    # remove all experiments where none of the optimizers have finished

    # Open the whole thing:
    t_start = perf_counter()
    logger.info("Opening the whole thing")

    combined_rows = []

    for row, values in df_output.iterrows():
        a = xr.open_mfdataset(values.dropna(), combine='by_coords')
        combined_rows.append(a.assign_coords(
            {'itemID': [row]}).drop('input'))

    xarr_full = xr.concat(combined_rows, dim='itemID')

    # xarr_full = xr.open_mfdataset(
    #     files,
    #     concat_dim=[xr.DataArray([int(f[0].stem) for f in files],
    #                              dims='itemID'), 'optimizer'],
    #     combine='nested')

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

    # ELA features
    ela_list_train, ela_list_test = [], []

    for index, file_r in enumerate(files_random_samples):
        if file_r == np.nan:
            continue

        ela_list_train.append(create_ela_feature_1(
            xr.open_dataset(file_r), itemID=int(file_r.stem)))
        l2co_logger.info(
            f"({index+1}/{len(files_random_samples)}) - Created ELA (train) for itemID = {file_r.stem}")

    xr_ela_train = xr.concat(ela_list_train, dim='problem')
    xr_ela_train.unstack().to_netcdf('ela_features_train.nc')

    for index, file_r in enumerate(files_random_samples):
        ela_list_test.append(create_ela_feature_1(
            xr.open_dataset(file_r), itemID=int(file_r.stem), iterations=config.t_switch))
        l2co_logger.info(
            f"({index+1}/{len(files_random_samples)}) - Created ELA (test) for itemID = {file_r.stem}")

    xr_ela_test = xr.concat(ela_list_train, dim='problem')
    xr_ela_test.unstack().to_netcdf('ela_features_test.nc')

    if config.select_output_and_combine:
        # xarr_only_output = xarr_full.drop('input')
        xarr_only_output = xarr_full.drop('input_dim')
        xarr_only_output.load().to_netcdf('entire_dataset.nc')

        # xarr_random_all = xarr_full_samples.drop('input')
        # xarr_random_all = xarr_random_all.drop('input_dim')
        xarr_full_samples.load().to_netcdf('entire_dataset_samples.nc')

        xarr_ela_features.load().to_netcdf('ela_features.nc')

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
