
import time
from pathlib import Path
from typing import List

import f3dasm
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from config import Config
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from l2co._logging import logger
from l2co.performance import margin_of_victory
from l2co._legacy.rollout import create_recommendation_results
from l2co._legacy.strategy import (best_area_under_curve_strategy, best_mov_strategy,
                           create_strategy, random_strategy)
from l2co.saving import load_netcdf
from l2co.create_recommender_dataset import RecommenderDatasetCreator


@hydra.main(config_path=".", config_name="config")
def main(config: Config):

    start_time = time.perf_counter()

    orig_cwd = Path(get_original_cwd())

    # Load experiment data
    experimentdata_path = orig_cwd / Path(config.experiment.existing_data_path) / Path(config.experiment.name)
    dirname = orig_cwd / Path(config.files.dirname)
    experimentdata = f3dasm.ExperimentData.from_csv(experimentdata_path)
    logger.info(f"Loaded experiment data from {experimentdata_path}")

    for i, (value_input, value_output) in enumerate(experimentdata):
        # Get index of experimentdata of loc i
        index = experimentdata.data.index.get_loc(i)


        raw_datafile_path = dirname / Path(config.files.dirname_raw) / Path(value_output['status'])
        results = load_netcdf(file_path=raw_datafile_path)

        # Create datarecommender object
        rdc = RecommenderDatasetCreator(results=results, jobnumber=index)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
