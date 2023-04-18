#                                                                       Modules
# =============================================================================

# Standard
import datetime
import logging
from typing import List

# Third-party
import numpy as np
import pandas as pd
from f3dasm.run_optimization import OptimizationResult
from recommenders.utils.constants import DEFAULT_HEADER

# Local
from .margin_of_victory import margin_of_victory

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
        f = self.results[0].function

        categories_list = []

        # Append features
        if f.noise == 0.0:
            categories_list.append('noiseless')
        if f.noise > 0.0:
            categories_list.append('noisy')
        if (f.dimensionality <= 5):
            categories_list.append('low_dimensional')
        if (5 < f.dimensionality <= 10):
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

    def create_recommender_data_features_functions_csv(self) -> pd.DataFrame:
        categories = self._create_recommender_data_features_functions()
        data = [[self.jobnumber, categories]]
        df = pd.DataFrame(data, columns=['itemID', 'genres'])

        return df

    def create_recommender_data_interactions_csv(self) -> pd.DataFrame:

        df_mov = margin_of_victory(self.results)

        ts: float = datetime.datetime.now().timestamp()

        # Take last margin of victory values
        scores = df_mov.iloc[-1].values

        min_scores = np.min(scores)
        max_scores = np.max(scores)

        L = (scores - min_scores)/(max_scores - min_scores)

        # Make range from 0 to 5 'stars'
        rating_array = (4*L)+1

        data = [[i+1, self.jobnumber, f"{rating:.5f}", f"{ts:.0f}"] for i, rating in enumerate(rating_array)]

        df_stars = pd.DataFrame(data, columns=DEFAULT_HEADER)

        logging.info("Created ratings matrix from margin of victory!")

        return df_stars


# def combine_all_ratings(query: str) -> pd.DataFrame:
#     for index, file in enumerate(glob.glob(query)):
#         f = pd.read_csv(file)
#         if index == 0:
#             total = f

#         else:
#             total = pd.concat([total, f], ignore_index=True)

#     return total
