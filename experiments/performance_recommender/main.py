import logging
import time
from pathlib import Path
from typing import List

import f3dasm
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from config import Config
from hydra.core.config_store import ConfigStore

from l2co.performance import margin_of_victory
from l2co.rollout import create_recommendation_results
from l2co.strategy import (best_area_under_curve_strategy, best_mov_strategy,
                           create_strategy, random_strategy)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):

    start_time = time.perf_counter()

    # Load experiment data
    experimentdata_path = Path(config.files.dirname) / Path(config.experiment.name)
    experimentdata = f3dasm.ExperimentData.from_csv(experimentdata_path)
    logging.info(f"Loaded experiment data from {experimentdata_path}")

    # Create strategies
    strategies: List[pd.DataFrame] = []
    default_strategy_options: dict = {"files_path": Path(
        config.files.dirname), "experimentdata": experimentdata, "strategy_determiner": None,
        "label": None, "parallelization":
        config.recommender.parallelization, }

    strategies.append(create_strategy(
        **{**default_strategy_options, **{"strategy_determiner": best_mov_strategy, "label": "best_mov_strategy"}}))

    strategies.append(create_strategy(
        **{**default_strategy_options, **{"strategy_determiner": best_area_under_curve_strategy,
                                          "label": "best_auc_strategy"}}))

    strategies.append(create_strategy(
        **{**default_strategy_options, **{"strategy_determiner": random_strategy, "label": "random_strategy"}}))

    # Create resulting dataframe
    df = create_recommendation_results(files_path=config.files.dirname, experimentdata=experimentdata,
                                       strategies=strategies,
                                       performance_metric=margin_of_victory,
                                       parallelization=config.recommender.parallelization)
    logging.info("Created recommendation results")

    # Save results
    df.to_csv(config.files.output_filename)
    logging.info(f"Saved recommendation results to {config.files.output_filename}")

    # Plot the dataframe and save to disk
    df.plot()
    logging.info("Plotted recommendation results")
    plt.savefig(config.files.output_filename.replace(".csv", ".png"))

    end_time = time.perf_counter()
    logging.info(f"Finished in {end_time - start_time:0.4f} seconds")


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
