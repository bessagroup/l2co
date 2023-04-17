#                                                                       Modules
# =============================================================================

# Standard
from copy import copy
from os import path
from pathlib import Path
from typing import List

# Third-party
import f3dasm
import numpy as np
from config import Config

# Local
from l2co.create_recommender_dataset import (
    create_recommender_data_features_functions_csv,
    create_recommender_data_features_optimizers_csv,
    create_recommender_data_interactions_csv)
from l2co.saving import save_as_cdf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def convert_config_to_input(config: Config, function_name: str, dimensionality: str,
                            noise: str, seed: int) -> List[dict]:
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

    Returns
    -------
        List of dictionaries, compatible to be used with the f3dasm.run_multiple_realizations function
    """

    dimensionality = int(dimensionality)

    function_class = f3dasm.find_class(module=f3dasm.functions, query=function_name)

    function_noise = float(noise)

    optimizers_class: List[f3dasm.Optimizer] = [
        f3dasm.find_class(f3dasm.optimization, optimizer_name)
        for optimizer_name in config.optimization.optimizers_names
    ]
    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.optimization.sampler_name)

    bounds = np.tile([config.optimization.lower_bound, config.optimization.upper_bound], (dimensionality, 1))

    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=dimensionality)
    function = function_class(
        dimensionality=dimensionality, noise=function_noise, scale_bounds=design.get_bounds(), seed=seed
    )
    data = f3dasm.ExperimentData(design=design)

    optimizers = [optimizer(data=data, seed=seed) for optimizer in optimizers_class]
    sampler = sampler_class(design=data.design, seed=seed)

    return [
        {
            "optimizer": optimizer,
            "function": function,
            "sampler": sampler,
            "number_of_samples": config.optimization.number_of_samples,
            "iterations": config.optimization.iterations,
            "realizations": config.optimization.realizations,
            "parallelization": config.optimization.parallelization,
            "seed": seed,
        }
        for optimizer in optimizers
    ]


def execute(config: Config, function_name: str, dimensionality: str, noise: str, seed: int) -> str:
    """Main program to create optimization training results and create the recommender matrices  

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

    Returns
    -------
        filename of the raw results file
    """

    results: List[f3dasm.OptimizationResult] = []

    options_list = convert_config_to_input(config=config, function_name=function_name,
                                           dimensionality=dimensionality, noise=noise, seed=seed)

    filename: str = f"{function_name}_seed{seed}_dim{dimensionality}"

    for options in options_list:
        results.append(f3dasm.run_multiple_realizations(**options))

    # Create dirs
    Path(f"./{config.files.dirname_raw}").mkdir(parents=True, exist_ok=True)
    Path(f"./{config.files.dirname_interactions}").mkdir(parents=True, exist_ok=True)
    Path(f"./{config.files.dirname_features}").mkdir(parents=True, exist_ok=True)

    # Create features functions matrix
    features_functions = create_recommender_data_features_functions_csv(results)
    features_functions.to_csv(f"./{config.files.dirname_features}/{filename}.csv", index=False)

    # Create features optimizers matrix

    if not path.exists(f"./{config.files.dirname_recommender}/totalfeatures_optimizers.csv"):  # done only once
        features_optimizer = create_recommender_data_features_optimizers_csv(results)
        features_optimizer.to_csv(f"./{config.files.dirname_recommender}/totalfeatures_optimizers.csv")

    # Create interactions matrix
    rating = create_recommender_data_interactions_csv(results)
    rating.to_csv(f"./{config.files.dirname_interactions}/{filename}.csv", index=False)

    # Save raw data cdf
    save_as_cdf(name=f"./{config.files.dirname_raw}/{filename}", results=copy(results))

    # Return the raw data filename
    return filename
