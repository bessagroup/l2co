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
#                                                                       Modules
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
    FUNCTION_MAPPING
from hydra.core.config_store import ConfigStore
from run_opt import BenchmarkOptimizationRoutine

# Local
from l2co import logger
from l2co._src.opfunu_functions import CEC_FUNCTIONS, NAMED_FUNCTIONS
from l2co._src.run_optimization import (_get_all_combinations,
                                        _get_name_from_scheduler)

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

    for sample in data:
        dim = sample.get("dimensionality")
        f = FUNCTION_MAPPING[sample.get("function_name").lower()]
        if not f.is_dim_compatible(sample.get("dimensionality")):
            compatible_functions = [
                ff for ff in filter(lambda f: FUNCTION_MAPPING[f.lower(
                )].is_dim_compatible(dim),
                    list(config.domain.function_name.categories))]

            new_function_name = np.random.choice(compatible_functions)

            data._input_data.set_data(
                index=sample.job_number,
                value=new_function_name, column="function_name")

    # Dirty hack for TypeError: Cannot setitem on a Categorical with
    # a new category (), set the categories first
    data._input_data.data[function_name_column] = data.\
        _input_data.data[function_name_column].astype(
        'category')

    # OUTPUT DATA
    optimizers = [opt_config['name'] for
                  opt_config in config.optimization.optimizers]
    optimizer_names = [_get_name_from_scheduler(
        schedule) for schedule in _get_all_combinations(optimizers, 100)]

    for name in optimizer_names:
        data.add_output_parameter(name=name, is_disk=True, exist_ok=True)

    # add random samples column
    data.add_output_parameter(name='random_samples',
                              is_disk=True, exist_ok=True)

    data.mark_all_in_progress_open()
    data.mark_all_error_open()
    data.mark_all_nan_open()

    # Store the ExperimentData to a csv file
    data.store()
    logger.info("Stored experimentdata")


def post_processing(config: Config):
    logger.info("== POST-PROCESSING ==")

    if 'from_file' in config.experimentdata:
        project_dir = config.experimentdata.from_file

    else:
        project_dir = Path().cwd()

    experimentdata = f3dasm.ExperimentData.from_file(project_dir)

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

    FUNCTION_MAPPING.update(NAMED_FUNCTIONS)
    FUNCTION_MAPPING.update(CEC_FUNCTIONS)

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

    if 'from_file' in config.experimentdata:
        project_dir = config.experimentdata.from_file

    else:
        project_dir = Path().cwd()

    # Retrieve the ExperimentData object
    max_tries = 100
    tries = 0

    while tries < max_tries:
        try:
            logger.info(f"Open ExperimentData at {project_dir}")
            data = f3dasm.ExperimentData.from_file(project_dir)
            break  # Break out of the loop if successful
        except FileNotFoundError:
            tries += 1
            logger.info(f"FileNotFoundError ({project_dir}), sleeping!")
            sleep(10)

    if tries == max_tries:
        logger.error(
            f"Could not open ExperimentData after {max_tries} attempts.")
        raise FileNotFoundError(f"Could not open ExperimentData after "
                                f"{max_tries} attempts.")

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
