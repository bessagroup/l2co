#                                                                       Modules
# =============================================================================

# Standard
from copy import copy
from pathlib import Path
from typing import List

# Third-party
import f3dasm
import numpy as np
from config import Config
from f3dasm import Design

# Local
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

    design = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensionality=dimensionality)
    function = function_class(
        dimensionality=dimensionality, noise=function_noise, scale_bounds=design.get_bounds(), seed=seed
    )
    data = f3dasm.ExperimentData(design=design)

    optimizers: List[f3dasm.Optimizer] = []
    for opt_config in config.optimization.optimizers:
        hyperparameters = opt_config['hyperparameters'] if 'hyperparameters' in opt_config else {}
        optimizers.append(f3dasm.optimization.find_optimizer(opt_config['class'])(
            data=data, seed=seed, hyperparameters=hyperparameters, name=opt_config['name']))

    sampler = sampler_class(design=data.design, seed=seed)

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
    function_name = design.get('function_name')
    dimensionality = design.get('dimensionality')
    budget = design.get('budget')
    noise = design.get('noise')
    seed = design.get('seed')

    results: List[f3dasm.OptimizationResult] = []

    options_list = convert_config_to_input(config=config, function_name=function_name,
                                           dimensionality=dimensionality, noise=noise, seed=seed, budget=budget)

    filename: str = f"{function_name}_seed{seed}_dim{dimensionality}"

    for options in options_list:
        results.append(f3dasm.run_multiple_realizations(**options))

    save_as_cdf(name=Path().cwd() / Path(config.files.dirname_raw) / Path(filename), results=copy(results))

    design.set('filename', filename)

    return design
