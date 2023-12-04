#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
from config import Config
from f3dasm.datageneration import DataGenerator

# Local
from l2co.run_optimization import (_get_all_combinations,
                                   _get_name_from_scheduler,
                                   run_multiple_realizations)
from l2co.run_sampling import run_sampling

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class BenchmarkOptimizationRoutine(DataGenerator):
    def __init__(self, config: Config):
        self.config = config

    def execute(self):
        function_name = self.experiment_sample.get('function_name')
        dimensionality = self.experiment_sample.get('dimensionality')
        budget = int(self.experiment_sample.get('budget'))
        noise = self.experiment_sample.get('noise')
        seed = int(self.experiment_sample.get('seed'))

        optimizers = [opt_config['name'] for
                      opt_config in self.config.optimization.optimizers]

        optimizer_schedulers = _get_all_combinations(optimizers, budget)

        # Sample from the optimizer_schedulers:
        n = int(len(optimizer_schedulers) * self.config.
                optimization.fraction_of_all_combinations)

        # Make sure we have minimally one sample
        if n == 0:
            n = 1
        idx = np.random.choice(len(optimizer_schedulers), n, replace=False)
        optimizer_schedulers_sample = [optimizer_schedulers[i] for i in idx]

        results = []
        for optimizer_scheduler in optimizer_schedulers_sample:
            result = run_multiple_realizations(
                optimizer_scheduler=optimizer_scheduler,
                data_generator=function_name,
                sampler=self.config.optimization.sampler_name,
                realizations=self.config.optimization.realizations,
                dimensionality=dimensionality,
                noise=noise,
                offset=True,
                n_samples=self.config.optimization.number_of_samples,
                parallelization=self.config.optimization.parallelization,
                seed=seed)

            self.experiment_sample.store(
                _get_name_from_scheduler(optimizer_scheduler),
                object=result, to_disk=True)

            results.append(result)

        # Now sample with random search
        results_samples = run_sampling(
            data_generator=function_name,
            sampler='random',
            dimensionality=dimensionality,
            noise=noise,
            offset=True,
            n_samples=budget,
            parallelization=self.config.optimization.parallelization,
            seed=seed)

        self.experiment_sample.store(
            'random_samples', results_samples, to_disk=True)

        return
