from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from f3dasm import ExperimentData

plt.style.use('science')


def select_data_from_features(query: List[str], features_functions: pd.DataFrame,
                              experimentdata: ExperimentData) -> ExperimentData:
    for q in query:
        features_functions = features_functions[features_functions['genres'].str.contains(q)]

    matched_indices = features_functions['itemID'].tolist()
    return experimentdata.select(matched_indices)


def plot(df_mov: pd.DataFrame, title: str = None):
    fig, ax = plt.figure(figsize=[10,8]), plt.axes()
    ax.plot(df_mov, label=df_mov.columns)
    ax.set_title(title)
    ax.set_xlabel('iterations')
    ax.set_ylabel('mean margin of victory')
    fig.legend()
    return fig, ax
