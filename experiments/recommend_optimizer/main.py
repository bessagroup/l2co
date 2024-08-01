#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import sleep

# Third-party
import hydra
from f3dasm import ExperimentData, logger
from recommendoptimizer import RecommendOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """

    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)
        # post_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)
        # post_processing(config)
    else:
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


def pre_processing(config):
    experimentdata = ExperimentData.from_yaml(config.experimentdata)

    experimentdata.store()

    recommend_optimizer = RecommendOptimizer(config)

    experimentdata.evaluate(data_generator=recommend_optimizer,
                            mode='cluster')


def process(config):
    # Retrieve the ExperimentData object
    max_tries = 100
    tries = 0

    while tries < max_tries:
        try:
            logger.info("Open ExperimentData")

            experimentdata = ExperimentData.from_file(Path().cwd())
            break  # Break out of the loop if successful
        except FileNotFoundError:
            tries += 1
            logger.info("FileNotFoundError, sleeping!")
            sleep(10)

    recommend_optimizer = RecommendOptimizer(config)

    experimentdata.evaluate(data_generator=recommend_optimizer,
                            mode='cluster')


if __name__ == "__main__":
    main()
