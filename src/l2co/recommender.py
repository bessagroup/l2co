from pathlib import Path
from typing import Dict, List, Tuple

import f3dasm
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from l2co._logging import logger
from l2co.strategy import Strategy


class Features:
    ...


def load_features(features_file: Path) -> pd.DataFrame:
    # Check if features_file exists
    if not features_file.exists():
        raise FileNotFoundError(f"File {features_file} not found!")

    df = pd.read_csv(features_file.with_suffix('.csv'), index_col=0)
    df.set_index('itemID', inplace=True)
    logger.info(f"Loaded features from {features_file}")
    return df


def load_interactions(interactions_file: Path) -> pd.DataFrame:
    # Check if interactions_file exists
    if not interactions_file.exists():
        raise FileNotFoundError(f"File {interactions_file} not found!")

    df = pd.read_csv(interactions_file.with_suffix('.csv'), index_col=0)
    logger.info(f"Loaded interactions from {interactions_file}")
    return df


class LearnedRecommenderStrategy(Strategy):
    label: str = 'learned_recommender_strategy'

    def __init__(self, features: pd.DataFrame, interactions: pd.DataFrame):
        self.features = features.sort_index()
        self.interactions = interactions.sort_values(by=['itemID', 'userID'])

        self.n_optimizers = len(self.interactions['userID'].unique())
        self.n_features = len(self.features.columns)
        self.X, self.y = self._to_tensors()

        self.model = self._build_model()
        self._train_model()

    def _to_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        X = self.features.to_numpy()
        y = self.interactions['rating'].to_numpy().reshape(-1, self.n_optimizers)
        return X, y

    def _build_model(self) -> tf.keras.models.Sequential:
        model = Sequential()
        # Add the input layer with 9 units
        model.add(Dense(units=16, activation='relu', input_dim=self.n_features))

        # Add the first hidden layer with 16 units
        model.add(Dense(units=16, activation='relu'))

        # Add the second hidden layer with 16 units
        model.add(Dense(units=16, activation='relu'))

        # Add the third hidden layer with 16 units
        model.add(Dense(units=16, activation='relu'))

        # Add the output layer with 5 units for regression
        model.add(Dense(units=self.n_optimizers, activation='linear'))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        return model

    def _train_model(self):
        self.model.fit(self.X, self.y, epochs=100, batch_size=64, validation_split=0.2)

    def __call__(self, data: xr.Dataset, budget: int) -> str:
        optimizer_mapping = {i: opt for i, opt in enumerate(data.optimizer.values)}
        features = pd.DataFrame(columns=['dim', 'budget', 'noise', 'convex', 'separable',
                                'differentiable', 'multimodal', 'randomized_term', 'parametric'])

        f = f3dasm.datageneration.functions.find_function(data.function_name)

        if data.function_noise == 0.0:
            noise = 0
        else:
            noise = 1

        features.loc[0] = [data.function_dimensionality, budget, noise, f.convex, f.separable,
                           f.differentiable, f.multimodal, f.randomized_term, f.parametric]
        features = features.astype(int)

        optimizer_number = self._calculate_majority_vote(new_entry=features.loc[0])
        return optimizer_mapping[int(optimizer_number)]


class ContentBasedRecommenderStategy(Strategy):
    label: str = 'content_based_recommender_strategy'

    def __init__(self, features: pd.DataFrame, interactions: pd.DataFrame,
                 threshold: float = 0.8, weights: np.ndarray = None):
        self.features = features
        self.interactions = interactions
        self.threshold = threshold

        if weights is None:
            weights = np.ones(len(features.columns))

        self.weights = weights

        self.min_max = self._min_max_columns()

    def _min_max_columns(self) -> Dict[str, List[float]]:
        # find the lowest and highest value of each column and save in a dict
        min_max = {}
        for col in self.features.columns:
            min_max[col] = [self.features[col].min(), self.features[col].max()]

        return min_max

    def _similarity(self, row_i: pd.Series, row_j: pd.Series, p: int = 2) -> float:
        feat_sim: np.ndarray = np.zeros(len(row_i))
        for feature_index, _ in enumerate(row_i):
            n, m = self.min_max[row_i.keys()[feature_index]]

            if n == m:
                # similarity is always 1!
                s = 1.0
            else:
                s = (1-(abs(row_i[feature_index] - row_j[feature_index])/(m - n)))**p
            feat_sim[feature_index] = s

        # multiply each feature similarity with its weight
        feat_sim *= self.weights

        # take average of feat_sim
        return np.mean(feat_sim)

    def _calculate_similarity_matrix(self, new_entry: pd.DataFrame) -> pd.DataFrame:
        similarity_df = pd.DataFrame([self._similarity(row_i=new_entry, row_j=self.features.loc[j])
                                     for j in self.features.index], columns=['similarity'], index=self.features.index)
        similarity_df.sort_values(by=['similarity'], ascending=False, inplace=True)
        return similarity_df

    def _calculate_majority_vote(self, new_entry: pd.Series) -> float:

        similarity_df = self._calculate_similarity_matrix(new_entry)

        # add the rating column to similarity_df
        similarity_df['rating'] = pd.Series([self.interactions[self.interactions['itemID'] == itemID]['rating'].argmax(
        ) for itemID in similarity_df.index], index=similarity_df.index)

        # take all the rows of similarity_df where the rating is not -1
        similarity_df = similarity_df[similarity_df['rating'] != -1]

        # Imbalanced dataset
        # from similarity_df, from each unique rating, sample 258 rows
        least_entries = min(similarity_df['rating'].value_counts())
        similarity_df_balanced = similarity_df.groupby('rating').apply(
            lambda x: x.sample(least_entries)).reset_index(drop=True)

        # take all the rows where similarity > threshold
        similarity_df_filtered = similarity_df_balanced[similarity_df_balanced['similarity'] >= self.threshold]

        weighted_ratings = {i: similarity_df_filtered[similarity_df_filtered['rating'] == i]['similarity'].sum(
        ) for i in similarity_df_filtered['rating'].unique()}

        # take argmax of weighted_ratings
        return max(weighted_ratings, key=weighted_ratings.get)

    def __call__(self, data: xr.Dataset, budget: int) -> str:
        optimizer_mapping = {i: opt for i, opt in enumerate(data.optimizer.values)}
        features = pd.DataFrame(columns=['dim', 'budget', 'noise', 'convex', 'separable',
                                'differentiable', 'multimodal', 'randomized_term', 'parametric'])

        f = f3dasm.datageneration.functions.find_function(data.function_name)

        if data.function_noise == 0.0:
            noise = 0
        else:
            noise = 1

        features.loc[0] = [data.function_dimensionality, budget, noise, f.convex, f.separable,
                           f.differentiable, f.multimodal, f.randomized_term, f.parametric]
        features = features.astype(int)

        optimizer_number = self._calculate_majority_vote(new_entry=features.loc[0])
        return optimizer_mapping[int(optimizer_number)]
