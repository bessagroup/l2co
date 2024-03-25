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
from typing import List

# Third-party
import hydra
import numpy as np
import xarray as xr
from ela_features import (create_ela, create_entire_dataset,
                          create_entire_dataset_samples)
from f3dasm import ExperimentData
# Local
from setup_logger import logger

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

    # remove all values that say 'ERROR' and 'invalid'
    df_output = df_output.replace('ERROR', np.nan)
    df_output = df_output.replace('invalid', np.nan)
    df_output = df_output.replace('not working', np.nan)

    # preceed every entry of df_output with experimentdata.project_dir
    df_output = df_output.applymap(
        lambda x: experiment_data.project_dir / x, na_action='ignore')
    # df_output = df_output.apply(lambda x: experiment_data.project_dir / x)

    df_random_samples = df_output['random_samples']
    files_random_samples: List[Path] = df_random_samples.dropna(
    ).values.tolist()

    # don't include the column 'random_samples'
    df_output = df_output.drop(columns=['random_samples'])

    # remove all experiments where none of the optimizers have finished
    df_output = df_output.dropna(how='all')

    create_entire_dataset(df_output)

    create_entire_dataset_samples(files_random_samples)

    # ELA features
    create_ela(files=files_random_samples, ela_folder='ela', t_switch=1000)

    ela_train = xr.concat([xr.open_dataarray(p) for p in (
        Path().cwd() / 'ela_train').glob('*.nc')], dim='itemID')

    ela_train.to_netcdf('ela_train.nc')

    ela_test = xr.concat([xr.open_dataarray(p) for p in (
        Path().cwd() / 'ela_test').glob('*.nc')], dim='itemID')

    ela_test.to_netcdf('ela_test.nc')


if __name__ == '__main__':
    main()
