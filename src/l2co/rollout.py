#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Union

# Third-party core
import pandas as pd
import xarray as xr
from f3dasm.design import ExperimentData

# Local
from .performance import Performance, margin_of_victory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def determine_winner(df: pd.DataFrame) -> str:
    # return the highest performance of the last index!
    return df.iloc[-1].idxmax()


def determine_loser(df: pd.DataFrame) -> str:
    # return the highest performance of the last index!
    return df.iloc[-1].idxmin()


def create_recommendation_results(files_path: str, experimentdata: ExperimentData,
                                  strategies: Union[List[pd.Series], None] = None,
                                  performance_metric: Performance = margin_of_victory) -> pd.DataFrame:
    total_mov = 0
    for index, (value_input, value_output) in enumerate(experimentdata):
        mov = _create_one_recommendation_result(index, value_input, value_output,
                                                files_path, strategies, performance_metric)
        # data: xr.Dataset = xr.load_dataset(f"{files_path}/raw/{value_output['status']}.cd")
        # mov: pd.DataFrame = performance_metric(data)

        # if strategies is not None:
        #     for strategy in strategies:
        #         mov[strategy.name] = mov[strategy.iloc[index]]

        total_mov += mov
    return total_mov


def _create_one_recommendation_result(index: int, value_input: dict, value_output: dict, files_path: str, strategies: List[pd.Series], performance_metric: Performance) -> pd.DataFrame:
    data: xr.Dataset = xr.load_dataset(f"{files_path}/raw/{value_output['status']}.cd")
    mov: pd.DataFrame = performance_metric(data)

    for strategy in strategies:
        mov[strategy.name] = mov[strategy.iloc[index]]

    return mov
