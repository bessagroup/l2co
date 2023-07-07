#                                                                       Modules
# =============================================================================

# Standard
import logging
from pathlib import Path
from shutil import copyfile
from time import sleep
from typing import Callable

# Third-party
import f3dasm
import hydra
import numpy as np
from config import Config
from f3dasm.logger import logger
from hydra.core.config_store import ConfigStore
from run_opt import execute

# Local
from l2co.filehandlers import CombineFeaturesFileHandler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def initial_script(config: Config):
    # Check if we use an existing dataset
    if config.experimentdata.existing_data_path is not False:
        data = f3dasm.ExperimentData.from_file(
            filename=f"{config.experimentdata.existing_data_path}/{config.experimentdata.name}")

    # Else, create the dataset
    else:
        # Create the Domain from the hydra yaml file
        design = f3dasm.Domain.from_yaml(config.design)

        logger.info("Created designspace")

        # Step 2: Filling the design space
        sampler = f3dasm.sampling.Sampler.from_yaml(config)
        data = f3dasm.ExperimentData.from_sampling(sampler)
        logger.info("Sampled from Domain")

        # Dirty hack for TypeError: Cannot setitem on a Categorical with a new category (), set the categories first
        data.data.data[('input', 'function_name')] = data.data.data[('input', 'function_name')].astype('object')

        function_class_list = [f3dasm.datageneration.functions.find_function(
            query=f) for f in list(design.input_space['function_name'].categories)]
        for index in range(data.get_number_of_datapoints()):
            i = data.get_inputdata_by_index(index)
            function_class = f3dasm.datageneration.functions.find_function(
                query=i['function_name'])
            if not function_class.is_dim_compatible(int(i['dimensionality'])):
                filtered_class_list = [ff.__name__ for ff in filter(lambda f: f.is_dim_compatible(
                    int(i['dimensionality'])), function_class_list)]
                i['function_name'] = np.random.choice(filtered_class_list)

                data.set_inputdata_by_index(index, i['function_name'], column=('input', 'function_name'))

        # Dirty hack for TypeError: Cannot setitem on a Categorical with a new category (), set the categories first
        data.data.data[('input', 'function_name')] = data.data.data[('input', 'function_name')].astype('category')

    # Store the ExperimentData to a csv file
    data.store(filename=config.experimentdata.name)
    logger.info("Stored experimentdata")
    # Make directory for recommender results
    Path(f"./{config.files.dirname_recommender}").mkdir(parents=True, exist_ok=True)

    if config.experimentdata.existing_data_path is not False:
        copyfile(f'{config.experimentdata.existing_data_path}/'
                 f'{config.files.dirname_recommender}/totalfeatures_functions.csv',
                 f'./{config.files.dirname_recommender}/totalfeatures_functions.csv')
        copyfile(f'{config.experimentdata.existing_data_path}/'
                 f'{config.files.dirname_recommender}/totalfeatures_optimizers.csv',
                 f'./{config.files.dirname_recommender}/totalfeatures_optimizers.csv')
        copyfile(f'{config.experimentdata.existing_data_path}/'
                 f'{config.files.dirname_recommender}/totalinteractions.csv',
                 f'./{config.files.dirname_recommender}/totalinteractions.csv')

    # Initialize filehandler
    features_filehandler = CombineFeaturesFileHandler(dir_name='features', master_file='totalfeatures_functions')
    interactions_filehandler = CombineFeaturesFileHandler(
        dir_name=config.files.dirname_interactions, master_file='totalinteractions')

    # Continuous check on newly created datafiles

    if config.hpc.jobid == -1:  # Pass if on local machine
        logger.info("Running on local machine, skipping filehandler loop")
        return

    while True:
        features_filehandler.run()
        interactions_filehandler.run()

        if features_filehandler.does_master_file_exist():
            copyfile(f'./{config.files.dirname_features}/totalfeatures_functions.csv',
                     f'./{config.files.dirname_recommender}/totalfeatures_functions.csv')

        if interactions_filehandler.does_master_file_exist():
            copyfile(f'./{config.files.dirname_interactions}/totalinteractions.csv',
                     f'./{config.files.dirname_recommender}/totalinteractions.csv')

        if data.is_all_finished():
            logging.info("All jobs are finished. Exiting filehandler loop!")
            break

        sleep(10)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        initial_script(config)

    elif config.hpc.jobid == -1:  # Sequential
        initial_script(config)
        job(config, execute)

    else:
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        job(config, execute)


def job(config: Config, execute_callable: Callable):
    """Main script that handles the execution of open jobs

    Parameters
    ----------
    config
        Hydra configuration file object
    execute_callable
        function to execute a job. Requires to arguments; the config and any **kwargs from the design
    """

    # Retrieve the ExperimentData object
    data = f3dasm.ExperimentData.from_file(filename=config.experimentdata.name)
    logger.debug("Loaded experimentdata")

    data.run(execute, mode='cluster', kwargs={'config': config})

    # # Let the node repeat the execution of jobs
    # while True:
    #     try:
    #         jobnumber, input_value, _ = data.get_open_job_data()
    #     except Exception:
    #         logger.info("There are no open jobs left!")
    #         break

    #     logger.info(f"Retrieved job {jobnumber}")

    #     try:
    #         returnvalue = execute_callable(config, jobnumber, **input_value)
    #         data.write_outputdata_by_index(index=jobnumber, value=returnvalue)
    #         logger.info(f"Wrote outputdata for job {jobnumber}")
    #     except Exception as e:
    #         logging.exception(f"An exception of type {type(e).__name__} occurred: {e}")
    #         returnvalue = 'ERROR'
    #         data.write_error(index=jobnumber)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
