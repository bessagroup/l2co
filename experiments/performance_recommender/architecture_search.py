import xarray as xr
from f3dasm import Design

from l2co.recommender import LearnedRecommenderStrategy2
from l2co.strategy import StrategyManager


def execute(design: Design, xarr: xr.Dataset) -> Design:
    recommender = LearnedRecommenderStrategy2(combined_data=xarr, **design.input_data)
    strategy_list = [recommender]

    strategy_manager = StrategyManager(combined_data=xarr, strategy_list=strategy_list)

    design['mov'] = float(strategy_manager.rollout.sel(strategy='strategy', fractional_budget=1.).values)
    return design
