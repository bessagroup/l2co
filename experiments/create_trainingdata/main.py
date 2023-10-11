#                                                                       Modules
# =============================================================================

# Standard
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

    logger.info("== PRE-PROCESS ==")
    data = f3dasm.ExperimentData.from_yaml(config)

    # Dirty hack for TypeError: Cannot setitem on a Categorical with a new category (), set the categories first
    data.input_data.data['function_name'] = data.input_data.data['function_name'].astype('object')

    # function_class_list = [f3dasm.datageneration.find_function(
    #     query=f) for f in list(data.domain.space['function_name'].categories)]
    for index, input_data in enumerate(data):
        i = data.input_data.data.iloc[index]
        # function_class = f3dasm.datageneration.find_function(
        #     query=i['function_name'])
        if not is_dim_compatible(i['function_name'], data.domain):
            filtered_class_list = [ff for ff in filter(lambda f: is_dim_compatible(f, data.domain),
                                   list(data.domain.space['function_name'].categories))]
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
    execute_callable
        function to execute a job. Requires to arguments; the config and any **kwargs from the design
    """

    logger.info("== PROCESS ==")

    # Retrieve the ExperimentData object
    data = f3dasm.ExperimentData.from_file(filename=config.experimentdata.name)
    logger.debug("Loaded experimentdata")

    benchmark_optimization_routine = BenchmarkOptimizationRoutine(config)

    data.evaluate(benchmark_optimization_routine, mode=config.experimentdata.mode)

    if config.experimentdata.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store(filename=config.experimentdata.name)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
