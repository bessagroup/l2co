import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import List, Union

import f3dasm
import pandas as pd
import xarray as xr

from l2co._logging import logger
from l2co.performance import margin_of_victory
from l2co.saving import load_netcdf

#                                                                Protocol class
# =============================================================================


class Strategy:
    label: str = "strategy"

    def __init__(self):
        ...

    def __call__(self, data: xr.Dataset) -> str:
        ...

#                                                                 Stategy Manager
# ===============================================================================


class StrategyManager:
    def __init__(self, raw_data_path: Path, strategy_list: List[Strategy],
                 output_directory: Path, mode: str = 'sequential'):
        self.strategy_list = strategy_list
        self.raw_data_path = raw_data_path
        self.mode = mode
        self.df_rollout = 0
        self.n_tasks = 0
        self.output_directory = output_directory

        # If output_directory does not exists, create it
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def mean_margin_of_victory(self):
        return self.df_rollout / self.n_tasks

    def _add_margin_of_victory(self, df_mov: pd.DataFrame, filename: Path):
        df_rollout_temp = deepcopy(self.df_rollout)
        df_rollout_temp += df_mov

        # Some checks
        # Check if NaN values are present
        if df_rollout_temp.isnull().values.any():
            logger.debug(f"NaN values are present in df_mov from {filename.stem}")
            return

        # Overwrite df_rollout with temporary and increment n_tasks
        self.df_rollout = df_rollout_temp
        self.n_tasks += 1

    def save_mean_margin_of_victory(self, filename: Path):
        self.mean_margin_of_victory.to_csv(self.output_directory / filename.with_suffix('.csv'))

    def save_margin_of_victory(self, df_mov: pd.DataFrame, filename: Path):
        df_mov.to_csv(self.output_directory / filename.with_suffix('.csv'))

    def post_processing(self, save_mov: Union[None, str], df_mov: pd.DataFrame, filename: Path) -> None:
        if self.mode == 'sequential':
            self._add_margin_of_victory(df_mov, filename)

            # if save_move is not None, save the margin of victory
            if save_mov is not None:
                self.save_mean_margin_of_victory(save_mov)

        if self.mode == 'cluster':
            self.save_margin_of_victory(df_mov, filename)

    def __call__(self, design: f3dasm.Design, save_mov: Union[None, str] = 'margin_of_victory') -> f3dasm.Design:
        t_start = perf_counter()

        filename = Path(design._dict_output['raw_path'])
        raw_data: xr.Dataset = load_netcdf(filename)

        df_mov = margin_of_victory(raw_data, budget=int(design['budget']))

        for strategy in self.strategy_list:
            recommendation: str = strategy(raw_data, int(design['budget']))
            design[strategy.label] = recommendation

            # Copy column of recommendation to new column named strategy.label
            df_mov[strategy.label] = df_mov[recommendation]

        self.post_processing(save_mov, df_mov, filename)

        logger.info(f"Roll-out {filename.stem} ({perf_counter() - t_start:0.4f} s)")

        return design

#                                                              Strategy classes
# =============================================================================


class BestMoVStrategy(Strategy):
    label: str = 'best_mov_strategy'

    def __call__(self, data: xr.Dataset) -> str:
        mov = margin_of_victory(data)
        return mov.iloc[-1].idxmax()


class RandomStrategy(Strategy):
    label: str = 'random_strategy'

    def __call__(self, data: xr.Dataset) -> str:
        return random.choice(list(data.optimizer.values))
