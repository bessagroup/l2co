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
                    strategy_determiner: Callable[[xr.Dataset], str], label: str) -> pd.Series:
    recommender = []
    for index, (data_input, data_output) in enumerate(experimentdata):
        data = xr.load_dataset(f"{files_path}/raw/{data_output['status']}.cd")
        recommender.append(strategy_determiner(data))

    strategy_series = pd.Series(recommender, index=experimentdata.data.index, name=label)
    return strategy_series


def best_mov_strategy(data: xr.Dataset):
    mov = margin_of_victory(data)
    return determine_winner(mov)


def worst_mov_strategy(data: xr.Dataset):
    mov = margin_of_victory(data)
    return determine_loser(mov)


def random_strategy(data: xr.Dataset):
    return random.choice(list(data.optimizer.values))
