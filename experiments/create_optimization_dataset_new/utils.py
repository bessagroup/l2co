from typing import List

import numpy as np
import pandas as pd
from f3dasm import Block, ExperimentData
from f3dasm.datageneration import DataGenerator
from omegaconf import DictConfig

from l2co._src.tasks.benchmarkfunctions.opfunu_functions import (
    CEC_FUNCTIONS, NAMED_FUNCTIONS)
from l2co._src.tasks.benchmarkfunctions.run_optimization import (
    OptimizerScheduler, get_all_combinations, name_from_schedule,
    run_optimization_schedule_multiple_realizations)

MY_FUNCTION_MAPPING = {**CEC_FUNCTIONS, **NAMED_FUNCTIONS}

# =============================================================================

# =============================================================================


class FunctionSampler(Block):
    def __init__(self, seed: int, noise_distribution: np.ndarray,
                 ):
        self.seed = seed
        self.noise_distribution = noise_distribution

    def arm(self, data: ExperimentData) -> None:
        self.dim_filters = {d: list(filter(
            lambda f: MY_FUNCTION_MAPPING[f.lower()].is_dim_compatible(d)
            and MY_FUNCTION_MAPPING[f.lower()].differentiable, list(
                data.domain.input_space['function_name'].categories)))
            for d in data.domain.input_space['dimensionality'].categories}

    def call(self, data: ExperimentData, n_samples: int) -> ExperimentData:
        rng = np.random.default_rng(seed=self.seed)

        dims = rng.choice(
            data.domain.input_space['dimensionality'].categories,
            size=n_samples)
        function_names = np.array(
            [rng.choice(self.dim_filters[d]) for d in dims])
        noise = rng.choice(data.domain.input_space['noise'].categories,
                           size=n_samples,
                           p=self.noise_distribution)
        seed = rng.integers(
            data.domain.input_space['seed'].lower_bound,
            data.domain.input_space['seed'].upper_bound, size=n_samples)
        budget = np.array(
            [data.domain.input_space['budget'].value] * n_samples)
        df = pd.DataFrame(
            [dims, function_names, noise, seed, budget],
            index=[
                'dimensionality', 'function_name', 'noise',
                'seed', 'budget']).T

        return ExperimentData(input_data=df,
                              domain=data.domain,
                              project_dir=data.project_dir)


class RunOptimization(DataGenerator):
    def __init__(self, config: DictConfig):
        self.config = config
        self.optimizer_combinations: List[List[OptimizerScheduler]] \
            = get_all_combinations(optimizers=config.optimizers,
                                   total_evals=config.n_evals,
                                   switch=config.switch)

    def execute(self):
        es = self.experiment_sample.input_data

        for optimizer_scheduler in self.optimizer_combinations:
            data_xarr = run_optimization_schedule_multiple_realizations(
                function_name=es['function_name'],
                optimizer_schedules=optimizer_scheduler,
                dimensionality=es['dimensionality'],
                noise=es['noise'],
                seed=es['seed'],
                n_realizations=self.config.optimization.n_realizations,
            )

            self.experiment_sample.store(
                name=name_from_schedule(optimizer_scheduler),
                object=data_xarr, to_disk=True)
