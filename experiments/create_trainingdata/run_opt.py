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
from f3dasm import Design
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
    function_class = f3dasm.datageneration.functions.find_function(query=function_name)
    function_noise = float(noise)
    sampler_class: f3dasm.Sampler = f3dasm.sampling.find_sampler(config.optimization.sampler_name)
    bounds = np.tile([config.optimization.lower_bound, config.optimization.upper_bound], (dimensionality, 1))

    domain = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensionality=dimensionality)
    function = function_class(
        dimensionality=dimensionality, noise=function_noise, scale_bounds=domain.get_bounds(), seed=seed
    )
    data = f3dasm.ExperimentData(domain=domain)

    optimizers: List[f3dasm.Optimizer] = []
    for opt_config in config.optimization.optimizers:
        hyperparameters = opt_config['hyperparameters'] if 'hyperparameters' in opt_config else {}
        optimizers.append(f3dasm.optimization.find_optimizer(opt_config['class'])(
            data=data, seed=seed, hyperparameters=hyperparameters, name=opt_config['name']))

    sampler = sampler_class(design=domain, seed=seed)

    return [
        {
            "optimizer": optimizer,
            "function": function,
            "sampler": sampler,
            "number_of_samples": config.optimization.number_of_samples,
            "iterations": budget,
            "realizations": config.optimization.realizations,
            "parallelization": config.optimization.parallelization,
            "seed": seed,
        }
        for optimizer in optimizers
    ]


def execute(design: Design, config: Config) -> Design:
    function_name = design['function_name']
    dimensionality = design['dimensionality']
    budget = int(design['budget'])
    noise = design['noise']
    seed = design['seed']

    results: List[f3dasm.OptimizationResult] = []

    options_list = convert_config_to_input(config=config, function_name=function_name,
                                           dimensionality=dimensionality, noise=noise, seed=seed, budget=budget)

    try:
        for options in options_list:

            if options['optimizer'].get_name() in ['EvoSaxCMAES', 'EvoSaxSimAnneal', 'EvoSaxPSO', 'EvoSaxDE']:
                options['parallelization'] = False

            results.append(f3dasm.run_multiple_realizations(**options))
    except OSError:
        # for every dict in the options list, change the key 'parallelization' to False
        for options in options_list:
            options['parallelization'] = False
            results.append(f3dasm.run_multiple_realizations(**options))

    results_xarray: xr.Dataset = convert_to_xarray(results=copy(results))

    design.store(object=results_xarray, name=config.files.dirname_raw)

    design.store(object=margin_of_victory(results_xarray), name=config.files.dirname_mov)

    if not config.recommender.create_recommender_dataset:
        return design

    # Augment the designs
    dataset_augmenter = DatasetAugmenter(n_samples=config.recommender.n_augmented_samples,
                                         upper_bound_budget=budget,
                                         lower_bound_budget=config.recommender.lower_bound_budget)

    _ = dataset_augmenter(design)
    augmented_designs = dataset_augmenter.designs

    # Create the recommender dataset
    interaction_matrix_list: List[pd.DataFrame] = []
    features_matrix_list: List[pd.DataFrame] = []

    for index, augmented_design in enumerate(augmented_designs):
        task_number: int = int(design.job_number * (config.recommender.n_augmented_samples + 1) + index)
        recommender_dataset_creator = RecommenderDatasetCreator(results=results_xarray, jobnumber=task_number)
        interaction_matrix_list.append(recommender_dataset_creator.create_recommender_data_interactions_csv(
            scaler=config.recommender.scaler, budget=int(augmented_design['budget'])))

        features_matrix_list.append(
            recommender_dataset_creator.create_recommender_data_features_functions_csv(
                budget=int(augmented_design['budget'])))

    df_interaction_matrix = pd.concat(interaction_matrix_list, ignore_index=True)
    df_features_matrix = pd.concat(features_matrix_list, ignore_index=True)

    design.store(df_interaction_matrix, name=config.files.dirname_recommender_interactions)
    design.store(df_features_matrix, name=config.files.dirname_recommender_features)

    return design


class DatasetAugmenter:
    def __init__(self, n_samples: int, upper_bound_budget: int, lower_bound_budget: int = 0):
        self.designs: List[f3dasm.Design] = []
        self.n_samples = n_samples
        self.upper_bound_budget = upper_bound_budget
        self.lower_bound_budget = lower_bound_budget

        self.salib_problem = self.create_salib_problem()

    def create_salib_problem(self):
        return {
            'num_vars': 1,
            'names': ['budget'],  # Replace 'parameter1' with the actual parameter name
            # Replace with the desired lower and upper bounds
            'bounds': [[self.lower_bound_budget, self.upper_bound_budget]],
        }

    def get_experimentdata(self, domain: f3dasm.Domain) -> f3dasm.ExperimentData:
        exp_data = f3dasm.ExperimentData(domain)
        designs_dataframe_dict = [design_.to_dataframe_dict() for design_ in self.designs]
        df = pd.DataFrame(designs_dataframe_dict, columns=exp_data.data.data.columns)
        exp_data.add(df)
        return exp_data

    def get_samples(self) -> np.ndarray:
        return latin.sample(self.salib_problem, self.n_samples).flatten()

    def __call__(self, design: f3dasm.Design):

        # add the original design
        self.designs.append(design)

        for sample in self.get_samples():
            design_ = deepcopy(design)
            design_._dict_input['budget'] = int(sample)
            self.designs.append(design_)
        # print(f"Augmenting {design.job_number}...")

        return design
