import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Union

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

        filename = Path(design._dict_output['status'])
        raw_data: xr.Dataset = load_netcdf(self.raw_data_path / filename.with_suffix('.cd'))

        df_mov = margin_of_victory(raw_data)

        for strategy in self.strategy_list:
            recommendation: str = strategy(raw_data)
            design.set(strategy.label, recommendation)

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


class ContentBasedRecommenderStategy(Strategy):
    label: str = 'content_based_recommender_strategy'

    def __init__(self, features: pd.DataFrame, interactions: pd.DataFrame, threshold: float = 0.8):
        self.features = features
        self.interactions = interactions
        self.threshold = threshold
        self.min_max = self._min_max_columns()

    def _min_max_columns(self) -> Dict[str, List[float]]:
        # find the lowest and highest value of each column and save in a dict
        min_max = {}
        for col in self.features.columns:
            min_max[col] = [self.features[col].min(), self.features[col].max()]

        return min_max

    def _similarity(self, row_i: pd.Series, row_j: pd.Series, p: int = 2) -> float:
        similarity = 0
        for feature_index, _ in enumerate(row_i):
            n, m = self.min_max[row_i.keys()[feature_index]]
            s = (1-(abs(row_i[feature_index] - row_j[feature_index])/(m - n)))**p
            similarity += s

        # divide by the number of features
        similarity /= len(row_i)

        return similarity

    def _calculate_similarity_matrix(self, new_entry: pd.DataFrame) -> pd.DataFrame:

        # calculate the similarity of the first row with all other rows and put them in a dataframe
        similarity_df = pd.DataFrame(columns=['index', 'similarity'])
        for j in self.features.index:

            similarity_df.loc[j] = [j, self._similarity(row_i=new_entry, row_j=self.features.loc[j])]

            # cast index column to int
        similarity_df['index'] = similarity_df['index'].astype(int)
        similarity_df.sort_values(by=['similarity'], ascending=False, inplace=True)
        return similarity_df

    def _calculate_majority_vote(self, new_entry: pd.Series) -> float:

        similarity_df = self._calculate_similarity_matrix(new_entry)
        # add a new column to similarity_df called rating
        similarity_df['rating'] = -2

        # for each itemID in similarity_df, find the rating in highest_ratings and save it to similarity_df
        for itemID in similarity_df['index']:
            similarity_df.loc[itemID, 'rating'] = self.interactions[self.interactions['itemID']
                                                                    == itemID]['rating'].argmax()

        # take all the rows of similarity_df where the rating is not -1
        similarity_df = similarity_df[similarity_df['rating'] != -1]

        # Imbalanced dataset
        # from similarity_df, from each unique rating, sample 258 rows
        least_entries = min(similarity_df['rating'].value_counts())
        similarity_df_balanced = similarity_df.groupby('rating').apply(
            lambda x: x.sample(least_entries)).reset_index(drop=True)

        # take all the rows where similarity > threshold
        similarity_df_filtered = similarity_df_balanced[similarity_df_balanced['similarity'] >= self.threshold]

        # weighted sum of each of the unique ratings with weight similarity
        weighted_ratings = {i: 0 for i in similarity_df_filtered['rating'].unique()}
        for index, row in similarity_df_filtered.iterrows():
            weighted_ratings[row['rating']] += row['similarity']

        # take argmax of weighted_ratings
        return max(weighted_ratings, key=weighted_ratings.get)

    def __call__(self, data: xr.Dataset) -> str:
        optimizer_mapping = {i: opt for i, opt in enumerate(data.optimizer.values)}
        features = pd.DataFrame(columns=['dim', 'noise', 'convex', 'separable',
                                'differentiable', 'multimodal', 'randomized_term', 'parametric'])

        f = f3dasm.datageneration.functions.find_function(data.function_name)

        if data.function_noise == 0.0:
            noise = 0
        else:
            noise = 1

        features.loc[0] = [data.function_dimensionality, noise, f.convex, f.separable,
                           f.differentiable, f.multimodal, f.randomized_term, f.parametric]
        features = features.astype(int)

        optimizer_number = self._calculate_majority_vote(new_entry=features.loc[0])
        return optimizer_mapping[int(optimizer_number)]
