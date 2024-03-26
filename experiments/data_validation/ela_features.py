import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from f3dasm.datageneration import DataGenerator
from pathos import multiprocessing as mp

from l2co import logger as l2co_logger
from l2co._src.ela import create_ela_feature_1


def create_entire_dataset(df_output: pd.DataFrame):

    # Open the whole thing:
    l2co_logger.info("Opening the whole thing")

    combined_rows = []

    for row, values in df_output.iterrows():
        a = xr.open_mfdataset(values.dropna(), combine='by_coords')
        combined_rows.append(a.assign_coords(
            {'itemID': [row]}).drop('input'))
        l2co_logger.info(f"Entire dataset: {row}/{len(df_output)}")

    xarr_full = xr.concat(combined_rows, dim='itemID')

    xarr_only_output = xarr_full.drop('input_dim')
    xarr_only_output.to_netcdf('entire_dataset.nc')


def create_entire_dataset_samples(files_random_samples: List[Path]):

    t_start = time.perf_counter()
    l2co_logger.info("Opening the whole thing: random samples")
    xarr_full_samples = xr.open_mfdataset(
        files_random_samples,
        concat_dim=xr.DataArray([int(f.stem) for f in files_random_samples],
                                dims='itemID'),
        combine='nested')

    l2co_logger.info(
        f"Finished opening the whole thing: random samples "
        f"(Took {time.perf_counter() - t_start} seconds)")

    xarr_full_samples.to_netcdf('entire_dataset_samples.nc')


def create_one_ela(file_r: Path, ela_folder: Path, t_switch: Optional[int]):
    t = time.perf_counter()
    if file_r == np.nan:
        l2co_logger.info(f"File {file_r} is NaN. Skipping...")
        return

    xarr_file = xr.open_dataset(file_r, chunks='auto')

    if bool(xarr_file.isnull().any().output.values):
        l2co_logger.info(f"File {file_r} has NaN values. Skipping...")
        return

    ela = create_ela_feature_1(xarr_file, itemID=int(
        file_r.stem), iterations=t_switch)
    ela.unstack().to_netcdf(
        ela_folder / f'{file_r.stem}_ela.nc')
    l2co_logger.info(
        f"Created ELA for itemID = {file_r.stem} ({time.perf_counter() - t} s)")


def create_ela(files: List[Path], ela_folder: str, t_switch: int):
    # ELA features
    ela_folder_train = Path().cwd() / f"{ela_folder}_train"
    ela_folder_train.mkdir(exist_ok=True)

    with mp.Pool() as pool:
        pool.starmap(create_one_ela, [
                     (file_r, ela_folder_train, None) for file_r in files])

    ela_folder_test = Path().cwd() / f"{ela_folder}_test"
    ela_folder_test.mkdir(exist_ok=True)

    with mp.Pool() as pool:
        pool.starmap(create_one_ela, [
                     (file_r, ela_folder_test, t_switch) for file_r in files])


class ELAGenerator(DataGenerator):
    def __init__(self, t_switch: Optional[int]):
        self.t_switch = t_switch

    def execute(self):
        file_r = Path(self.experiment_sample.get('filename'))

        t = time.perf_counter()
        if file_r == np.nan:
            l2co_logger.info(f"File {file_r} is NaN. Skipping...")
            return

        xarr_file = xr.open_dataset(file_r, chunks='auto')

        if bool(xarr_file.isnull().any().output.values):
            l2co_logger.info(f"File {file_r} has NaN values. Skipping...")
            return

        ela_train = create_ela_feature_1(xarr_file, itemID=int(
            file_r.stem), iterations=None)

        self.experiment_sample.store(
            object=ela_train.unstack(), name='ela_train', to_disk=True)

        ela_test = create_ela_feature_1(xarr_file, itemID=int(
            file_r.stem), iterations=self.t_switch)

        self.experiment_sample.store(
            object=ela_test.unstack(), name='ela_test', to_disk=True)

        l2co_logger.info(
            f"Created ELA for itemID = {file_r.stem} ({time.perf_counter() - t} s)")
