#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import sleep

# Third-party
import f3dasm
import hydra
import numpy as np
from config import Config
from f3dasm.logger import logger as f3dasm_logger
from hydra.core.config_store import ConfigStore
from run_opt import execute

# Local
from l2co._logging import logger
from l2co.filehandlers import ConcatDataFrameFilehandler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def pre_processing(config: Config):

    logger.info("== PRE-PROCESS ==")
    data = f3dasm.ExperimentData.from_yaml(config)

    # Dirty hack for TypeError: Cannot setitem on a Categorical with a new category (), set the categories first
    data.input_data.data['function_name'] = data.input_data.data['function_name'].astype('object')

    function_class_list = [f3dasm.datageneration.functions.find_function(
        query=f) for f in list(data.domain.input_space['function_name'].categories)]
    for index in range(len(data)):
        i = data.input_data.data.iloc[index]
        function_class = f3dasm.datageneration.functions.find_function(
            query=i['function_name'])
        if not function_class.is_dim_compatible(int(i['dimensionality'])):
            filtered_class_list = [ff.__name__ for ff in filter(lambda f: f.is_dim_compatible(
                int(i['dimensionality'])), function_class_list)]
            i['function_name'] = np.random.choice(filtered_class_list)

            data.input_data.set_data(index, value=i['function_name'], column='function_name')

    # Dirty hack for TypeError: Cannot setitem on a Categorical with a new category (), set the categories first
    data.input_data.data['function_name'] = data.input_data.data['function_name'].astype('category')

    # Store the ExperimentData to a csv file
    data.store(filename=config.experimentdata.name)
    logger.info("Stored experimentdata")


def post_processing(config: Config):
    logger.info("== POST-PROCESSING ==")
    experimentdata = f3dasm.ExperimentData.from_file(filename=config.experimentdata.name)

    dirname_recommender_interactions = Path.cwd() / Path(config.files.dirname_recommender_interactions)
    dirname_recommender_features = Path.cwd() / Path(config.files.dirname_recommender_features)

    master_file_features = Path('features_master_file')
    master_file_interactions = Path('interactions_master_file')

    fh_interactions = ConcatDataFrameFilehandler(
        dir_name=dirname_recommender_interactions, master_file=master_file_interactions, remove_files=True)

    fh_features = ConcatDataFrameFilehandler(
        dir_name=dirname_recommender_features, master_file=master_file_features, remove_files=True)

    while True:
        fh_interactions.run()
        fh_features.run()
        if experimentdata.is_all_finished():
            logger.info("All jobs are finished. Exiting filehandler loop!")
            break

        sleep(5)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """

    logger.setLevel(config.log_level)
    f3dasm_logger.setLevel(config.log_level)
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)
        post_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)
        post_processing(config)

    else:
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


def process(config: Config):
    """Main script that handles the execution of open jobs

    Parameters
    ----------
    config
        Hydra configuration file object
    execute_callable
        function to execute a job. Requires to arguments; the config and any **kwargs from the design
    """

    logger.info("== PROCESS ==")

    # Retrieve the ExperimentData object
    data = f3dasm.ExperimentData.from_file(filename=config.experimentdata.name)
    logger.debug("Loaded experimentdata")

    data.run(execute, mode=config.experimentdata.mode, kwargs={'config': config})

    if config.experimentdata.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store(filename=config.experimentdata.name)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
