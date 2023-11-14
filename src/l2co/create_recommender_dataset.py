#                                                                       Modules
# =============================================================================

# Standard
import datetime
from typing import List

# Third-party
import pandas as pd
import xarray as xr
from f3dasm import OptimizationResult
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from recommenders.utils.constants import DEFAULT_HEADER
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Local
from ._logging import logger
from .performance import Performance, margin_of_victory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class RecommenderDatasetCreator:
    def __init__(self, results: List[OptimizationResult], jobnumber: int):
        self.results = results
        self.jobnumber = jobnumber

    def create_recommender_data_features_optimizers_csv(self) -> pd.DataFrame:
        data = []

        for index, result in enumerate(self.results):
            f = ",".join(result.optimizer.get_info())
            data.append([index+1, f])

        df = pd.DataFrame(data, columns=['userID', 'information'])
        return df

    def _create_recommender_data_features_functions(self) -> List[str]:
        f = _datagenerator_factory(
            self.results[0].data_generator, self.results[0].data[0].domain)
        categories_list = []

        if f.noise is None:
            f.noise = 0.0

        # Append features
        if f.noise == 0.0:
            categories_list.append('noiseless')
        if f.noise > 0.0:
            categories_list.append('noisy')
        if (f.dimensionality <= 5):
            categories_list.append('low_dimensional')
        if (5 < f.dimensionality < 10):
            categories_list.append('medium_dimensional')
        if (f.dimensionality >= 10):
            categories_list.append('high_dimensional')
        if f.convex:
            categories_list.append('convex')
        if f.separable:
            categories_list.append('seperable')
        if f.differentiable:
            categories_list.append('differentiable')
        if f.multimodal:
            categories_list.append('multimodal')
        if f.randomized_term:
            categories_list.append('randomized_term')
        if f.parametric:
            categories_list.append('parametric')

        categories = ",".join(categories_list)
        return categories

    def create_recommender_data_features_functions_csv(self, budget: int) -> pd.DataFrame:
        columns = ('itemID', 'dim', 'budget', 'noise',
                   'convex', 'separable', 'multimodal')

        if isinstance(self.results, list):
            f = _datagenerator_factory(
                self.results[0].data_generator, self.results[0].data[0].domain)
        elif isinstance(self.results, xr.Dataset):
            f = _datagenerator_factory(
                self.results.function_name, self.results.function_dimensionality)

        if self.results.function_noise == 0.0:
            noise = 0
        else:
            noise = 1

        # create a new row of dg
        features = [[self.jobnumber, self.results.function_dimensionality, budget, noise, f.convex,
                    f.separable, f.multimodal]]

        df_features = pd.DataFrame(features, columns=columns)
        # cast all booleans in dg to ints
        df_features = df_features.astype(int)

        return df_features

    def legacy_create_recommender_data_features_functions_csv(self, budget: int) -> pd.DataFrame:
        categories = self._create_recommender_data_features_functions()
        data = [[self.jobnumber, categories]]
        df = pd.DataFrame(data, columns=['itemID', 'genres'])

        return df

    def create_recommender_data_interactions_csv(self,
                                                 performance_metric: Performance = margin_of_victory,
                                                 scaler: str = 'MinMax', budget: int = -1) -> pd.DataFrame:

        df_mov = performance_metric(self.results)

        ts: float = datetime.datetime.now().timestamp()

        # Take last margin of victory values
        scores = df_mov.iloc[budget].values

        # # Minmax scale
        if scaler == 'MinMax':
            scaler_ = MinMaxScaler(feature_range=(1, 5))

        elif scaler == 'Standard':
            scaler_ = StandardScaler()

        scaler_.fit(scores.reshape(-1, 1))

        rating_array = scaler_.transform(scores.reshape(-1, 1)).reshape(-1)

        data = [[i+1, self.jobnumber, f"{rating:.5f}", f"{ts:.0f}"]
                for i, rating in enumerate(rating_array)]

        df_stars = pd.DataFrame(data, columns=DEFAULT_HEADER)

        logger.debug("Created ratings matrix from margin of victory!")

        return df_stars
