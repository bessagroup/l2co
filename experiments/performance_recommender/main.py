from pathlib import Path
from typing import List

import f3dasm
import hydra
import pandas as pd
from config import Config
from hydra.core.config_store import ConfigStore

from l2co.performance import area_under_curve
from l2co.rollout import create_recommendation_results
from l2co.strategy import best_mov_strategy, create_strategy

@hydra.main(config_path=".", config_name="config")
def main(config: Config):

    # Load experiment data
    experimentdata_path = Path(config.files.dirname) / Path(config.experiment.name)
    experimentdata = f3dasm.ExperimentData.from_csv(experimentdata_path)

    # Create strategies
    strategies: List[pd.DataFrame] = []
    strategies.append(create_strategy(files_path=Path(config.files.dirname),
                      experimentdata=experimentdata, strategy_determiner=best_mov_strategy, label="best_mov_strategy"))

    # Create resulting datafram
    df = create_recommendation_results(files_path=Path(config.files.dirname), experimentdata=experimentdata,
                                       strategies=strategies,
                                       performance_metric=area_under_curve)

    # Save results
    df.to_csv(config.files.output_filename)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
