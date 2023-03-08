import datetime
import glob
import logging
import random
from typing import List

import numpy as np
import pandas as pd
from f3dasm.run_optimization import OptimizationResult, margin_of_victory
from recommenders.utils.constants import DEFAULT_HEADER


def create_recommender_data_features_csv(results: List[OptimizationResult]) -> pd.DataFrame:
    f = results[0].function
    hash: int = f.seed

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

    data = [[hash, categories]]
    df = pd.DataFrame(data, columns=['itemID', 'genres'])

    return df

def create_recommender_data_interactions_csv(results: List[OptimizationResult]) -> pd.DataFrame:

    df_mov = margin_of_victory(results)

    # hash: int = random.getrandbits(32)
    hash: int = results[0].function.seed
    ts: float = datetime.datetime.now().timestamp()

    # df = pd.read_csv ('./filename.csv', index_col=0)

    # Take last margin of victory values
    scores = df_mov.iloc[-1].values

    min_scores = np.min(scores)
    max_scores = np.max(scores)

    L = (scores - min_scores)/(max_scores - min_scores)

    # Make range from 0 to 5 'stars'
    rating_array = (4*L)+1

    data = [[i+1, hash, f"{rating:.5f}", f"{ts:.0f}"] for i, rating in enumerate(rating_array)]


    df_stars = pd.DataFrame(data, columns=DEFAULT_HEADER)

    logging.info("Created ratings matrix from margin of victory!")

    return df_stars
    df_mov.to_csv('g.csv', index=False)

def combine_all_ratings(query: str) -> pd.DataFrame:
    for index, file in enumerate(glob.glob(query)):
        f = pd.read_csv(file)
        if index == 0:
            total = f

        else:
            total = pd.concat([total, f], ignore_index=True)

    return total
