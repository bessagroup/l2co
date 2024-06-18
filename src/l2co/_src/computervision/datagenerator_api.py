"""
Datagenerator API for Experiments with f3dasm
"""

#
#                                                                       Modules
# =============================================================================

# Standard
import optax
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample

# Local
from .data import Dataset
from .models import SimpleMLP
from .training import training_multiple_realizations

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


OPTIMIZERS_MAPPING = {
    'adam': optax.adam,
    'sgd': optax.sgd,
    'amsgrad': optax.amsgrad
}

DATASET_MAPPING = {
    'mnist': Dataset.from_mnist,
    'fashion_mnist': Dataset.from_fashion_mnist,
    'cifar10': Dataset.from_cifar10
}

# =============================================================================


class MNIST(DataGenerator):
    def __init__(self, config):
        self.config = config

    def execute(self):
        self.config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=self.experiment_sample)

        for optimizer in self.config.optimization.optimizers:
            loss = training_multiple_realizations(
                optimizer=OPTIMIZERS_MAPPING[optimizer['optimizer']],
                model=SimpleMLP(n_features=self.config.model.n_features,
                                n_layers=self.config.model.n_layers,
                                n_labels=self.config.model.n_labels),
                data=DATASET_MAPPING[self.config.dataset](),
                seed=self.config.seed,
                epochs=self.config.optimization.epochs,
                batch_size=self.config.optimization.batch_size,
                n_realizations=self.config.optimization.realizations,
                optimizer_name=optimizer['name'],
                **optimizer['hyperparameters'],
            )

            self.experiment_sample.store(
                name=optimizer['name'], object=loss, to_disk=True)
