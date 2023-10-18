from __future__ import annotations

from typing import Any, Dict, List, Type

import numpy as np
import xarray as xr
from f3dasm import ExperimentData

#                                                                         Types
# =============================================================================

PERFORMANCE_DATASET_LABEL = 'movxarr_path'
RAW_DATASET_LABEL = 'path_raw'  # 'raw_path'
PerformanceDataset = Type[xr.Dataset]
StrategyDataArray = Type[xr.DataArray]
StrategyDataset = Type[xr.Dataset]

#                                                                          Load
# =============================================================================


def open_combined_dataset(experimentdata: ExperimentData, column: str) -> PerformanceDataset:
    if column == 'movxarr':

        return xr.open_mfdataset(experimentdata.output_data.to_dataframe()[PERFORMANCE_DATASET_LABEL])

    elif column == 'raw':
        return xr.open_mfdataset(experimentdata.output_data.to_dataframe()[RAW_DATASET_LABEL])


def open_one_dataset(experimentdata: ExperimentData, column: str, index: int) -> PerformanceDataset:
    if column == 'raw':
        return xr.open_dataset(experimentdata.output_data.to_dataframe()[RAW_DATASET_LABEL].loc[index])

    elif column == 'movxarr':
        return xr.open_dataset(experimentdata.output_data.to_dataframe()[PERFORMANCE_DATASET_LABEL].loc[index])

#                                                                Protocol class
# =============================================================================


class Strategy:
    label: str = "strategy"

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        """Create a prediction of the optimizer to use given a combine dataset

        Parameters
        ----------
        data
            xr.Dataset, combined_data

        Returns
        -------

        StrategyDataArray
            xr.DataArray with value 'optimizer' and coordinate 'itemID'
        """
        ...


class CustomStrategy(Strategy):
    label: str = "custom_strategy"

    @staticmethod
    def predict(dim: int, budget: int, noise: int, convex: int,
                separable: int, differentiable: int, multimodal: int,
                randomized_term: int, parametric: int) -> str:
        ...

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        predictions = [self.predict(**create_features_dict(xarr, id)) for id in xarr.itemID]
        return xr.DataArray(predictions, dims=['itemID'], coords={'itemID': xarr['itemID']})


#                                                              Strategy classes
# =============================================================================


class RandomStrategy(Strategy):
    label: str = 'random_strategy'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xr.DataArray(np.random.choice(xarr['optimizer'],
                                             size=xarr['itemID'].shape),
                            dims=['itemID'], coords={'itemID': xarr['itemID']})


# class BestMoVStrategy(Strategy):
#     label: str = 'best_mov_strategy'

#     def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
#         return xarr['mov'].sel(fractional_budget=1., method='nearest').idxmax('optimizer').drop('fractional_budget')


# class WorstMovStrategy(Strategy):
#     label: str = 'worst_mov_strategy'

#     def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
#         return xarr['mov'].sel(fractional_budget=1., method='nearest').idxmin('optimizer').drop('fractional_budget')


class WorstPerfProfile(Strategy):
    label: str = 'worst_perf_profile'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xarr['perf_profile'].mean('realization').idxmax('optimizer').sel(output_dim='y').drop('output_dim')


class BestPerfProfile(Strategy):
    label: str = 'best_perf_profile'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xarr['perf_profile'].mean('realization').idxmin('optimizer').sel(output_dim='y').drop('output_dim')

# =============================================================================


def create_strategy_xarr(combined_data: PerformanceDataset) -> StrategyDataset:
    return xr.Dataset({strategy.name: strategy for strategy in [
        xr.DataArray(opt, dims=['itemID'],
                     coords={'itemID': combined_data['itemID']},
                     name=opt) for opt in combined_data['optimizer'].values]})


def create_features_dict(combined_data: PerformanceDataset, itemID: int) -> Dict[str, Any]:
    ds = combined_data.drop_vars(['mov', 'perf_profile'])

    features = {var: ds.sel(itemID=itemID)[var].values for var in ds.data_vars}

    for feature, value in features.items():
        if isinstance(value, np.ndarray):
            if value.shape == ():
                features[feature] = value.item()

    return features

# =============================================================================


class StrategyManager:
    def __init__(self, experimentdata: ExperimentData, strategy_list: List[Strategy] = None):
        # Configure the strategies
        self.combined_data = open_combined_dataset(experimentdata)
        self.strategies = create_strategy_xarr(self.combined_data)

        if strategy_list is None:
            strategy_list = []

        strategy_list.extend([RandomStrategy(), BestPerfProfile(), WorstPerfProfile()])
        self._add_strategies(strategy_list)

    def _add_strategies(self, strategy_list):
        for strategy in strategy_list:
            self.strategies[strategy.label] = strategy(self.combined_data)

    def plot_performance_profile(self):
        perf_profile = xr.concat([self.combined_data.sel(itemID=self.strategies['itemID'],
                                                         optimizer=self.strategies[s])
                                  for s in self.strategies.data_vars],
                                 dim=xr.DataArray(self.strategies.data_vars, dims='strategy'))

        pp = perf_profile.stack({'problem': ['itemID', 'realization']})['perf_profile']

        # fraction of runs that are within a factor f of the best run
        f_values = np.linspace(1, 5, 300)

        fractions = []
        for f in f_values:
            fractions.append((pp <= f).mean(dim='problem'))

        fractions_xr = xr.DataArray(fractions, dims=('f', 'strategy', 'output_dim'), coords={
                                    'f': f_values, 'strategy': pp.strategy, 'output_dim': pp.output_dim})
        fractions_xr.plot(hue='strategy')
