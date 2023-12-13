"""
Main entrypoint of the experiment

Functions
---------

main
    Main script to call
pre_processing
    Pre-processing steps
process
    Main process to execute
"""

#
#                                                                        Modules
# =============================================================================

# Standard
import logging
from pathlib import Path
from time import sleep

# Third-party
import f3dasm
import hydra
import numpy as np
from config import Config
from f3dasm import logger as f3dasm_logger
from f3dasm._src.datageneration.functions.function_factory import \
    is_dim_compatible
from hydra.core.config_store import ConfigStore
from run_opt import BenchmarkOptimizationRoutine

# Local
from l2co._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def pre_processing(config: Config):
    logger.setLevel(logging.DEBUG)

    logger.info("== PRE-PROCESS ==")
    data = f3dasm.ExperimentData.from_yaml(config.experimentdata)

    # Dirty hack for TypeError: Cannot setitem on a Categorical with a
    # new category (), set the categories first
    function_name_column = data._input_data.columns.iloc('function_name')
    data._input_data.data[function_name_column] = data._input_data.data[
        function_name_column].astype(
        'object')

    for index, input_data in enumerate(data):
        i = data._input_data.data.iloc[index]

        if not is_dim_compatible(
                i[function_name_column].values[0], data.domain):
            filtered_class_list = [ff for ff in filter(
                lambda f: is_dim_compatible(f, data.domain),
                                   list(
                    data.domain.space['function_name'].categories))]
            new_function_name = np.random.choice(filtered_class_list)

            data._input_data.set_data(
                index, value=new_function_name, column='function_name')

    # Dirty hack for TypeError: Cannot setitem on a Categorical with
    # a new category (), set the categories first
    data._input_data.data[function_name_column] = data.\
        _input_data.data[function_name_column].astype(
        'category')

    # Store the ExperimentData to a csv file
    data.store()
    logger.info("Stored experimentdata")


def post_processing(config: Config):
    logger.info("== POST-PROCESSING ==")
    experimentdata = f3dasm.ExperimentData.from_file(Path().cwd())

    while True:
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
    """

    logger.info("== PROCESS ==")

    # Retrieve the ExperimentData object
    try:
        data = f3dasm.ExperimentData.from_file(Path().cwd())
    except FileNotFoundError:
        sleep(60)
        data = f3dasm.ExperimentData.from_file(Path().cwd())

    logger.debug("Loaded experimentdata")

    benchmark_optimization_routine = BenchmarkOptimizationRoutine(config)

    data.evaluate(benchmark_optimization_routine,
                  mode=config.mode)

    if config.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store()


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
