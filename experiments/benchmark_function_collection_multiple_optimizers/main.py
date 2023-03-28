import json
import logging
from os import path
from pathlib import Path
from shutil import copyfile
from time import sleep
from typing import List

import f3dasm
import hydra
import numpy as np
from config import Config
from hydra.core.config_store import ConfigStore

from l2co.create_recommender_dataset import (
    create_recommender_data_features_functions_csv,
    create_recommender_data_features_optimizers_csv,
    create_recommender_data_interactions_csv)
from l2co.filehandlers import CombineFeaturesFileHandler


def convert_config_to_input(config: Config) -> List[dict]:

    # Set the seed
    seed = np.random.randint(low=0, high=1e5)

    # Sample a dimension
    dimensionality = int(np.random.choice(config.design.dimensionality_list))
    # dimensionality = np.random.randint(
    #     low=config.design.dimensionality_lower_bound, high=config.design.dimensionality_upper_bound + 1
    # )

    functions_list = f3dasm.functions.get_functions(d=dimensionality)

    function_class = np.random.choice(functions_list)

    function_noise = float(np.random.choice([config.function.noise, 0.0]))

    optimizers_class: List[f3dasm.Optimizer] = [
        f3dasm.find_class(f3dasm.optimization, optimizer_name) for optimizer_name in config.optimizers.optimizers_names
    ]
    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)

    bounds = np.tile([config.design.lower_bound, config.design.upper_bound], (dimensionality, 1))

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
            "number_of_samples": config.sampler.number_of_samples,
            "iterations": config.execution.iterations,
            "realizations": config.execution.realizations,
            "parallelization": config.execution.parallelization,
            "seed": seed,
        }
        for optimizer in optimizers
    ]


def filehandler():
    Path("./recommender").mkdir(parents=True, exist_ok=True)

    features_filehandler = CombineFeaturesFileHandler(dir_name='features', master_file='totalfeatures_functions')
    interactions_filehandler = CombineFeaturesFileHandler(
        dir_name='interactions', master_file='totalinteractions')

    while True:
        features_filehandler.run()
        interactions_filehandler.run()

        if features_filehandler.does_master_file_exist():
            copyfile('./features/totalfeatures_functions.csv', './recommender/totalfeatures_functions.csv')

        if interactions_filehandler.does_master_file_exist():
            copyfile('./interactions/totalinteractions.csv', './recommender/totalinteractions.csv')

        sleep(10)


@hydra.main(config_path=".", config_name="config")
def main(cfg: Config):
    if cfg.hpc.jobid == 0:
        filehandler()

    else:
        job(cfg)


def job(cfg: Config):
    for _ in range(cfg.execution.number_of_functions):
        options_list = convert_config_to_input(config=cfg)

        results: List[f3dasm.OptimizationResult] = []

        for options in options_list:
            results.append(f3dasm.run_multiple_realizations(**options))

        filename: str = f"{options_list[0]['function'].get_name()}_seed{options_list[0]['seed']}_dim{options_list[0]['function'].dimensionality}"

        # Create dirs
        Path("./raw").mkdir(parents=True, exist_ok=True)
        Path("./interactions").mkdir(parents=True, exist_ok=True)
        Path("./features").mkdir(parents=True, exist_ok=True)

        # Save raw data
        json_data = json.dumps([result.to_json() for result in results])
        f3dasm.write_json(name="./raw/"+filename, json_string=json_data)

        # Create features functions matrix
        features_functions = create_recommender_data_features_functions_csv(results)
        features_functions.to_csv(f"./features/{filename}.csv", index=False)

        # Create features optimizers matrix

        if not path.exists("./recommender/totalfeatures_optimizers.csv"):  # done only once
            features_optimizer = create_recommender_data_features_optimizers_csv(results)
            features_optimizer.to_csv("./recommender/totalfeature_optimizers.csv")

        # Create interactions matrix
        rating = create_recommender_data_interactions_csv(results)
        rating.to_csv(f"./interactions/{filename}.csv", index=False)

        # f3dasm.write_pickle('mov_'+filename, df)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
