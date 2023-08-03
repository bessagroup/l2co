"""
3. Rollout and recommender strategy
"""
#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import perf_counter, sleep

# Third-party
import f3dasm
import hydra
import pandas as pd
from config import Config
from hydra.core.config_store import ConfigStore

# Local
from l2co._logging import logger
from l2co.filehandlers import CombineMarginOfVictoryFileHandler
from l2co.strategy import (BestMoVStrategy, ContentBasedRecommenderStategy,
                           RandomStrategy, StrategyManager)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

#                                                                       Process
# =============================================================================


def process(config: Config):
    logger.info("== JOB ==")

    start_time = perf_counter()

    # Read files
    raw_data = Path(config.files.raw_data)
    if not raw_data.exists():
        raise FileNotFoundError(f"Folder {raw_data} not found!")

    logger.info(f"raw_data folder: {raw_data}")

    experimentdata = f3dasm.ExperimentData.from_file(filename=Path(config.files.experimentdata_file))
    logger.info(f"Loaded experimentdata from {config.files.experimentdata_file}")

    features = pd.read_csv(Path(config.files.training_features_file), index_col=0)
    logger.info(f"Loaded features from {config.files.training_features_file}")

    interactions = pd.read_csv(Path(config.files.interactions_file), index_col=0)
    logger.info(f"Loaded interactions from {config.files.interactions_file}")

    # Create strategies
    recommender_strategy = ContentBasedRecommenderStategy(features=features, interactions=interactions)
    random_strategy = RandomStrategy()
    best_mov_strategy = BestMoVStrategy()

    strategy_manager = StrategyManager(raw_data_path=raw_data, strategy_list=[
                                       recommender_strategy, random_strategy, best_mov_strategy],
                                       output_directory=Path(config.files.output_dir), mode=config.experiment.mode)

    experimentdata.jobs.mark_all_open()

    for strategy in [recommender_strategy, random_strategy, best_mov_strategy]:
        experimentdata.add_new_output_column(strategy.label, f3dasm.ConstantParameter(''))
    experimentdata.store(filename=config.files.output_experimentdata_file)

    experimentdata = f3dasm.ExperimentData.from_file(filename=config.files.output_experimentdata_file)

    # Run
    experimentdata.run(strategy_manager, mode=config.experiment.mode,
                       kwargs={'save_mov': config.files.output_filename})

    # Store results
    if config.experiment.mode == 'sequential':
        experimentdata.store(filename=config.files.output_experimentdata_file)

    logger.info(f"Finished in {perf_counter() - start_time:0.4f} seconds")


#                                                               Post-processing
# =============================================================================

def post_processing(config: Config):
    logger.info("== POST-PROCESSING ==")
    experimentdata = f3dasm.ExperimentData.from_file(filename=config.files.output_experimentdata_file)

    dir_name = Path.cwd() / Path(config.files.output_dir)
    master_file = Path(config.files.output_master_file)

    logger.info(f"dir_name: {dir_name}")
    logger.info(f"master_file: {master_file}")

    fh = CombineMarginOfVictoryFileHandler(
        dir_name=dir_name, master_file=master_file, remove_files=True)

    while True:
        fh.run()
        if experimentdata.is_all_finished():
            logger.info("All jobs are finished. Exiting filehandler loop!")
            break

        sleep(5)

#                                                               Main entrypoint
# =============================================================================


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):

    if config.hpc.jobid == 0:
        post_processing(config)

    # If sequential
    if config.hpc.jobid == -1:
        process(config)
        post_processing(config)

    else:
        process(config)


if __name__ == "__main__":
    main()
