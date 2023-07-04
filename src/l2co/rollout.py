#                                                                       Modules
# =============================================================================

# Standard
import logging
from pathlib import Path
from typing import List, Union

import f3dasm
# Third-party core
import pandas as pd
import xarray as xr
from f3dasm.design import ExperimentData

# Local
from .performance import Performance, margin_of_victory
from .saving import load_netcdf

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
                                  performance_metric: Performance = margin_of_victory,
                                  parallelization: bool = False) -> pd.DataFrame:

    results = f3dasm.experiment.run_operation_on_experiments(
        experimentdata, _create_one_recommendation_result, parallel=parallelization,
        files_path=files_path, strategies=strategies, performance_metric=performance_metric)

    total_mov = 0
    for mov in results:
        total_mov += mov

    # divide total_mov by the number of experiments
    total_mov = total_mov / experimentdata.get_number_of_datapoints()

    return total_mov


def _create_one_recommendation_result(index: int, value_input: dict, value_output: dict, files_path: str,
                                      strategies: List[pd.Series], performance_metric: Performance) -> pd.DataFrame:

    # load_netcdf but convert str to Paths

    data: xr.DataSet = load_netcdf(Path(files_path) / Path('raw') / Path(value_output['status'] + '.cd'))

    # data: xr.Dataset = load_netcdf(f"{files_path}/raw/{value_output['status']}.cd")
    mov: pd.DataFrame = performance_metric(data)

    for strategy in strategies:
        mov[strategy.name] = mov[strategy.iloc[index]]

    logging.debug(f"Created mov for {value_output['status']}")
    return mov
