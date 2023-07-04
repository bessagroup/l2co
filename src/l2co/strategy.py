"""Functions that"""
#                                                                       Modules
# =============================================================================

# Standard

import logging
import random
from copy import copy
from itertools import combinations
from typing import Protocol, Union

import f3dasm
# Third-party core
import pandas as pd
import xarray as xr
from f3dasm.design import ExperimentData
from sklearn.neural_network import MLPClassifier

# Local
from ._logging import logger
from .performance import area_under_curve, margin_of_victory
from .rollout import determine_loser, determine_winner
from .saving import load_netcdf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class Strategy(Protocol):
    def __call__(self, data: xr.Dataset, options: Union[None, dict]) -> str:
        """Strategy protocol function that determines the strategy for each problem

        Parameters
        ----------
        data
            raw data in the form of an xarray dataset

        Returns
        -------
            label of optimizer that is chosen for each problem
        """
        ...


def create_strategy(files_path: str, experimentdata: ExperimentData,
                    strategy_determiner: Strategy, label: str,
                    load_dataset: bool = True, parallelization: bool = False,
                    options: Union[None, dict] = None) -> pd.Series:
    """Create a strategy for each problem in the experimentdata"""
    recommender = f3dasm.experiment.run_operation_on_experiments(
        experimentdata, _create_one_strategy, parallel=parallelization, experimentdata=experimentdata,
        files_path=files_path, strategy_determiner=strategy_determiner, label=label,
        load_dataset=load_dataset, options=options)

    strategy_series = pd.Series(recommender, index=experimentdata.data.index, name=label)

    logger.info(f"Created strategy series for {label}")
    return strategy_series


def _create_one_strategy(index: int, value_input: dict,
                         value_output: dict, experimentdata: ExperimentData, files_path: str,
                         strategy_determiner: Strategy, label: str,
                         load_dataset: bool, options: Union[None, dict]) -> str:
    i = experimentdata.data.index[index]

    # if options is None, create an empty dict
    if options is None:
        options = {}

    # if options is a dict, update the key 'index' with the current i
    options.update({'index': i})

    if load_dataset:
        experimentdata: xr.Dataset = load_netcdf(f"{files_path}/raw/{value_output['status']}.cd")
    else:
        experimentdata = 0

    logging.debug(f"Created strategy for {i}")

    return strategy_determiner(experimentdata, options)


def best_mov_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = margin_of_victory(data)
    return determine_winner(mov)


def worst_mov_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = margin_of_victory(data)
    return determine_loser(mov)


def best_area_under_curve_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = area_under_curve(data)
    return determine_winner(mov)


def worst_area_under_curve_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = area_under_curve(data)
    return determine_loser(mov)


def random_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    return random.choice(list(data.optimizer.values))


def lightfm_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    index: int = options['index']
    df_predictions: pd.DataFrame = options['df_predictions']

    optimizer_mapping = {i+1: opt for i, opt in enumerate(data.optimizer.values)}
    optimizer_number = df_predictions[df_predictions['itemID'] == index].sort_values(
        by=['prediction'], ascending=False).iloc[0]['userID']
    return optimizer_mapping[int(optimizer_number)]


def lightfm_strategy_reverse(data: xr.Dataset, options: Union[None, dict]) -> str:
    index: int = options['index']
    df_predictions: pd.DataFrame = options['df_predictions']

    optimizer_mapping = {i+1: opt for i, opt in enumerate(data.optimizer.values)}
    optimizer_number = df_predictions[df_predictions['userID'] == index].sort_values(
        by=['prediction'], ascending=True).iloc[0]['itemID']
    return optimizer_mapping[int(optimizer_number)]


def manual_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    index: int = options['index']
    features_functions: pd.DataFrame = options['features_functions']
    recommender_dict: dict = options['recommender_dict']

    features = features_functions[features_functions['itemID'] == index]['genres'].values[0].split(',')
    f_combinations = list(combinations(features, 1))
    f_combinations.extend(list(combinations(features, 2)))
    f = [",".join(f_combinations[ii]) for ii in range(len(f_combinations))]
    f_opt = [recommender_dict[feature] for feature in f]
    return max(set(f_opt), key=f_opt.count)


def mlp_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:

    index: int = options['index']
    features_functions: pd.DataFrame = options['features_functions']
    unique_features: dict = options['unique_features']
    classifier: MLPClassifier = options['classifier']
    # # Reverse map
    # reverse_mapping = {value: key for key, value in output_map.items()}

    features = features_functions[features_functions['itemID'] == index]['genres'].values[0].split(',')

    features_dict = copy(unique_features)
    for value in features:
        features_dict[value] = 1

    mlp_input = pd.DataFrame([features_dict]).to_numpy()

    mlp_output = classifier.predict(mlp_input)
    return mlp_output  # reverse_mapping[mlp_output.item()]
