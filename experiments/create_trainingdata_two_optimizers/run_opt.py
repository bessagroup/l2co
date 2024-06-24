#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
import xarray as xr
from config import Config
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from f3dasm.datageneration import DataGenerator
from f3dasm.design import make_nd_continuous_domain

# Local
from l2co import logger, run_multiple_realizations, run_sampling
from l2co._src.run_optimization import (_get_all_combinations,
                                        _get_all_combinations_no_switch,
                                        _get_name_from_scheduler,
                                        needs_gradient)

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

        if self.config.optimization.combinations:
            optimizer_schedulers = _get_all_combinations(
                optimizers=optimizers,
                iterations=budget,
                switch=self.config.optimization.switch)
        else:
            optimizer_schedulers = _get_all_combinations_no_switch(
                optimizers=optimizers,
                iterations=budget)

            # Sample from the optimizer_schedulers:
        n = int(len(optimizer_schedulers) * self.config.
                optimization.fraction_of_all_combinations)

        # Make sure we have minimally one sample
        if n == 0:
            n = 1
        idx = np.random.choice(len(optimizer_schedulers), n, replace=False)
        optimizer_schedulers_sample = [optimizer_schedulers[i] for i in idx]

        #####

        for optimizer_scheduler in optimizer_schedulers:

            # Check if already computed
            if isinstance(self.experiment_sample.output_data_with_references[
                    _get_name_from_scheduler(optimizer_scheduler)], str):
                continue

            # Check if function is not differentiable and optimizer needs
            # gradient information
            if needs_gradient(optimizer_scheduler) and not is_differentiable(
                    function_name, dimensionality, self.config):

                logger.info(f"Function {function_name} is not differentiable,"
                            f"but optimizer {optimizer_scheduler} needs "
                            f"gradient. Writing 'NA'")

                self.experiment_sample.store(
                    name=_get_name_from_scheduler(optimizer_scheduler),
                    object='invalid', to_disk=False)
                continue

            if optimizer_scheduler in optimizer_schedulers_sample:
                result: xr.Dataset = run_multiple_realizations(
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

                # If there are nan values in the result, don't store it and log
                # as invalid
                if result.output.isnull().any():
                    logger.info(
                        f"Function {function_name} with optimizer "
                        f"{optimizer_scheduler} has nan values. Writing "
                        f"'invalid'")
                    self.experiment_sample.store(
                        name=_get_name_from_scheduler(optimizer_scheduler),
                        object='invalid', to_disk=False)

                else:
                    self.experiment_sample.store(
                        _get_name_from_scheduler(optimizer_scheduler),
                        object=result, to_disk=True)

        if not isinstance(self.experiment_sample.output_data_with_references[
                'random_samples'], str):
            # Now sample with random search
            results_samples = run_sampling(
                data_generator=function_name,
                sampler='random',
                dimensionality=dimensionality,
                noise=noise,
                offset=True,
                n_samples=budget,
                parallelization=self.config.optimization.parallelization,
                seed=seed,
                realizations=self.config.optimization.realizations)

            self.experiment_sample.store(
                'random_samples', results_samples, to_disk=True)

        return


def is_differentiable(
        function_name: str, dimensionality: int, config: Config) -> bool:
    function = _datagenerator_factory(
        data_generator=function_name,
        domain=make_nd_continuous_domain(
            bounds=np.tile(
                [config.optimization.lower_bound,
                 config.optimization.upper_bound],
                (dimensionality, 1)),
            dimensionality=dimensionality))

    return function.differentiable
