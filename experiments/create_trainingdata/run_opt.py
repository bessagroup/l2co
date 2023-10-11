#                                                                       Modules
# =============================================================================


# Standard
from copy import copy, deepcopy
from typing import List

# Third-party
import f3dasm
import numpy as np
import pandas as pd
import xarray as xr
from config import Config
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from SALib.sample import latin

# Local
from l2co.create_recommender_dataset import RecommenderDatasetCreator
from l2co.performance import margin_of_victory
from l2co.saving import convert_to_xarray

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def convert_config_to_input(config: Config, function_name: str, dimensionality: str,
                            noise: str, seed: int, budget: int) -> List[dict]:
    """Convert the hydra configuration and design parameters to classes and objects

    Parameters
    ----------
    config
        full configuration parameters from Hydra YAML file
    function_name
        design parameter: classname of the benchmark function
    dimensionality
        design parameter: dimensionality of the benchmark function
    noise
        design parameter: standard deviation of Gaussian noise on benchmark function
    seed
        design parameter: seed of the benchmark function
    budget
        design parameter: number of iterations of the benchmark function

    Returns
    -------
        List of dictionaries, compatible to be used with the f3dasm.run_multiple_realizations function
    """

    dimensionality = int(dimensionality)
    budget = int(budget)
    function_noise = float(noise)
    seed = int(seed)

    bounds = np.tile([config.optimization.lower_bound, config.optimization.upper_bound], (dimensionality, 1))
    domain = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensionality=dimensionality)
    kwargs = {'noise': function_noise, 'scale_bounds': domain.get_bounds(), 'seed': seed}

    hyperparameters_list, optimizers_list = [], []
    for opt_config in config.optimization.optimizers:
        h = opt_config['hyperparameters'] if 'hyperparameters' in opt_config else {}
        h.update({'seed': seed})
        hyperparameters_list.append(h)
        optimizers_list.append(opt_config['name'])

    return [
        {
            "optimizer": optimizer,
            "data_generator": function_name,
            "sampler": config.optimization.sampler_name,
            "number_of_samples": config.optimization.number_of_samples,
            "iterations": budget,
            "domain": domain,
            "kwargs": kwargs,
            "hyperparameters": hyperparameters,
            "realizations": config.optimization.realizations,
            "parallelization": config.optimization.parallelization,
            "seed": seed,
        }
        for optimizer, hyperparameters in zip(optimizers_list, hyperparameters_list)
    ]


class BenchmarkOptimizationRoutine(DataGenerator):
    def __init__(self, config: Config):
        self.config = config

    def execute(self) -> ExperimentSample:
        function_name = self.experiment_sample.get('function_name')
        dimensionality = self.experiment_sample.get('dimensionality')
        budget = int(self.experiment_sample.get('budget'))
        noise = self.experiment_sample.get('noise')
        seed = self.experiment_sample.get('seed')

        results: List[f3dasm.OptimizationResult] = []

        options_list = convert_config_to_input(config=self.config, function_name=function_name,
                                               dimensionality=dimensionality, noise=noise, seed=seed, budget=budget)

        try:
            for options in options_list:

                if options['optimizer'] in ['EvoSaxCMAES', 'EvoSaxSimAnneal', 'EvoSaxPSO', 'EvoSaxDE']:
                    options['parallelization'] = False

                results.append(f3dasm.run_multiple_realizations(**options))
        except OSError:
            # for every dict in the options list, change the key 'parallelization' to False
            for options in options_list:
                options['parallelization'] = False
                results.append(f3dasm.run_multiple_realizations(**options))

        results_xarray: xr.Dataset = convert_to_xarray(results=copy(results))

        self.experiment_sample.store(object=results_xarray, name=self.config.files.dirname_raw, to_disk=True)

        if not self.config.recommender.create_recommender_dataset:
            return

        # Augment the designs
        augmented_designs = augment_designs(experiment_sample=self.experiment_sample,
                                            n_samples=self.config.recommender.n_augmented_samples,
                                            upper_bound_budget=budget,
                                            lower_bound_budget=self.config.recommender.lower_bound_budget)

        # Create the recommender dataset
        features_matrix_list: List[pd.DataFrame] = []
        mov_list: List[pd.DataFrame] = []
        perf_profile_list: List[xr.DataArray] = []
        task_numbers: List[int] = []

        for index, augmented_design in enumerate(augmented_designs):
            budget: int = int(augmented_design.get('budget'))
            task_number: int = int(self.experiment_sample.job_number *  # NOQA
                                   (self.config.recommender.n_augmented_samples + 1) + index)
            task_numbers.append(task_number)
            recommender_dataset_creator = RecommenderDatasetCreator(results=results_xarray, jobnumber=task_number)

            features_matrix_list.append(
                recommender_dataset_creator.create_recommender_data_features_functions_csv(
                    budget=int(augmented_design.get('budget'))))

            mov_list.append(margin_of_victory(
                results_xarray, budget=budget))

            perf_profile_list.append(perf_profile(results_xarray, budget=budget, itemID=task_number))

        df_features_matrix = pd.concat(features_matrix_list, ignore_index=True)

        # Create the xarray dataset
        df_features_matrix.set_index('itemID', inplace=True)
        xarr_all = xr.Dataset(df_features_matrix)
        df_list = []
        for df in mov_list:
            df.index.name = 'fractional_budget'
            xr_ = xr.DataArray(df, dims=['fractional_budget', 'optimizer'], coords={
                'fractional_budget': np.linspace(0., 1., len(df)), 'optimizer': df.columns})

            df_list.append(xr_)

        c = xr.concat(df_list, dim=xr.DataArray(task_numbers, dims='itemID'))
        xarr_all['mov'] = c
        xarr_all['perf_profile'] = xr.concat(perf_profile_list, dim='itemID')

        self.experiment_sample.store(xarr_all, name=self.config.files.dirname_movxarr, to_disk=True)

        return


def augment_designs(experiment_sample: f3dasm.ExperimentSample, n_samples: int,
                    lower_bound_budget: int, upper_bound_budget: int) -> List[ExperimentSample]:

    experiment_samples = []

    salib_problem = {
        'num_vars': 1,
        'names': ['budget'],
        'bounds': [[lower_bound_budget, upper_bound_budget]],
    }

    # add the original design
    design_0 = deepcopy(experiment_sample)
    design_0._jobnumber = design_0.job_number * (n_samples + 1)
    experiment_samples.append(design_0)

    samples = latin.sample(salib_problem, n_samples).flatten()

    for index, sample in enumerate(samples):
        design_ = deepcopy(design_0)
        design_._jobnumber = design_0.job_number + index + 1
        design_._dict_input['budget'] = int(sample)
        experiment_samples.append(design_)

    return experiment_samples


def perf_profile(data, budget: int, itemID: int) -> xr.DataArray:
    x: xr.DataArray = data['output'].sel(iterations=range(0, budget)).min(dim='iterations')

    # standard scale x
    x = (x - x.mean()) / x.std()
    x = x - x.min() + 1.0

    # add a new dimension named itemID and set to value itemID
    x = x.expand_dims({'itemID': [itemID]})

    return x
