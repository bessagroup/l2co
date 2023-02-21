import datetime
import glob
import logging
import random

import numpy as np
import pandas as pd
from recommenders.utils.constants import DEFAULT_HEADER


def create_recommender_data_csv(df_mov: pd.DataFrame) -> pd.DataFrame:

    hash: int = random.getrandbits(32)
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
