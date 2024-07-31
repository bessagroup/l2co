"""
Datagenerator API for Experiments with f3dasm
"""

#
#                                                                       Modules
# =============================================================================

# Standard
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from hydra.utils import instantiate

# Local
from ..data import Dataset
from .training import training_multiple_realizations

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ComputerVisionTask(DataGenerator):
    def __init__(self, config):
        self.config = config

    def execute(self):
        config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=self.experiment_sample)

        for name, optimizer in config.optimization.optimizers.items():
            loss = training_multiple_realizations(
                optimizer=instantiate(optimizer),
                model=instantiate(config.model),
                data=instantiate(config.dataset),
                iterations=config.optimization.iterations,
                seed=config.seed,
                batch_size=config.optimization.batch_size,
                n_realizations=config.optimization.realizations,
                optimizer_name=name,
            )

            self.experiment_sample.store(
                name=name, object=loss, to_disk=True)


class ComputerVisionTask_Dataset(DataGenerator):
    def __init__(self, config):
        self.config = config

    def execute(self):
        config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=self.experiment_sample)

        dataset = Dataset.from_mnist1d(
            labels=[int(i) for i in self.experiment_sample.input_data.keys()
                    if self.experiment_sample.get(i)])

        for name, optimizer in config.optimization.optimizers.items():
            loss = training_multiple_realizations(
                optimizer=instantiate(optimizer),
                model=instantiate(config.model),
                data=dataset,
                iterations=config.optimization.iterations,
                seed=config.seed,
                batch_size=config.optimization.batch_size,
                n_realizations=config.optimization.realizations,
                optimizer_name=name,
            )

            self.experiment_sample.store(
                name=name, object=loss, to_disk=True)
