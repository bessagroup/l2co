"""Functions that"""
#                                                                       Modules
# =============================================================================

# Standard

import random
from typing import Callable

# Third-party core
import pandas as pd
import xarray as xr
from f3dasm.design import ExperimentData

# Local
from .margin_of_victory import margin_of_victory
from .rollout import determine_loser, determine_winner

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def create_strategy(files_path: str, experimentdata: ExperimentData,
                    strategy_determiner: Callable[[xr.Dataset], str], label: str, *args, **kwargs) -> pd.Series:
    recommender = []
    for i, (data_input, data_output) in enumerate(experimentdata):
        index = experimentdata.data.index[i]
        data = xr.load_dataset(f"{files_path}/raw/{data_output['status']}.cd")
        recommender.append(strategy_determiner(data, index, *args, **kwargs))

    strategy_series = pd.Series(recommender, index=experimentdata.data.index, name=label)
    return strategy_series


def best_mov_strategy(data: xr.Dataset, *args, **kwargs) -> str:
    mov = margin_of_victory(data)
    return determine_winner(mov)


def worst_mov_strategy(data: xr.Dataset, *args, **kwargs) -> str:
    mov = margin_of_victory(data)
    return determine_loser(mov)


def random_strategy(data: xr.Dataset, *args, **kwargs) -> str:
    return random.choice(list(data.optimizer.values))


def lightfm_strategy(data: xr.Dataset, index: int, df_predictions: pd.DataFrame) -> str:
    optimizer_mapping = {i+1: opt for i, opt in enumerate(data.optimizer.values)}
    optimizer_number = df_predictions[df_predictions['itemID'] == index].sort_values(
        by=['prediction'], ascending=False).iloc[0]['userID']
    return optimizer_mapping[int(optimizer_number)]


def lightfm_strategy_reverse(data: xr.Dataset, index: int, df_predictions: pd.DataFrame) -> str:
    optimizer_mapping = {i+1: opt for i, opt in enumerate(data.optimizer.values)}
    optimizer_number = df_predictions[df_predictions['userID'] == index].sort_values(
        by=['prediction'], ascending=True).iloc[0]['itemID']
    return optimizer_mapping[int(optimizer_number)]
