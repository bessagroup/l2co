from typing import List

import f3dasm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # NOQA
import xarray as xr
from f3dasm import ExperimentData

from l2co.create_recommender_dataset import RecommenderDatasetCreator
from l2co.performance import (_calculate_mean_std, _normalize,
                              area_under_curve, margin_of_victory)

plt.style.use(['science', 'no-latex'])


def select_data_from_features(query: List[str], features_functions: pd.DataFrame,
                              experimentdata: ExperimentData) -> ExperimentData:
    for q in query:
        features_functions = features_functions[features_functions['genres'].str.contains(q)]

    matched_indices = features_functions['itemID'].tolist()
    return experimentdata.select(matched_indices)


def plot(df_mov: pd.DataFrame, title: str = None):
    fig, ax = plt.figure(figsize=[10, 8]), plt.axes()
    ax.plot(df_mov, label=df_mov.columns)
    ax.set_title(title)
    ax.set_xlabel('iterations')
    ax.set_ylabel('mean margin of victory')
    fig.legend()
    return fig, ax


# plot_mean_std function but give a ax
def plot_auc(results: xr.Dataset, title: str, ax: plt.Axes):
    df = area_under_curve(results)
    optimizer_names = results.optimizer.values
    ax.plot(df, label=optimizer_names)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Negative area under curve')
    ax.legend()
    ax.set_title(title)
    return ax

# plot_mean_std function but give a ax


def plot_normalized_objective(results: xr.Dataset, title: str, ax: plt.Axes) -> plt.Axes:
    df, optimizer_names = _calculate_mean_std(results)
    df = _normalize(df, optimizer_names)
    ax.plot(df, label=optimizer_names)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best normalized objective value')
    ax.legend()
    ax.set_title(title)
    return ax

# plot_mean_std function but give a ax


def plot_margin_of_victory(results: xr.Dataset, title: str, ax: plt.Axes) -> plt.Axes:
    df = margin_of_victory(results)
    optimizer_names = results.optimizer.values
    ax.plot(df, label=optimizer_names)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Margin of victory')
    ax.legend()
    ax.set_title(title)
    return ax


def plot_loss_landscape_3d(results: xr.Dataset, title: str, ax: plt.Axes) -> plt.Axes:
    function_class = f3dasm.functions.find_function(results.function_name)
    f = function_class(dimensionality=2, scale_bounds=np.array(
        [[0., 1.], [0., 1.]]), noise=results.function_noise, offset=False)
    _, ax = f.plot(show=False, px=100, ax=ax, orientation='3D')
    ax.set_title(title)
    return ax


def plot_loss_landscape_2d(results: xr.Dataset, title: str, ax: plt.Axes) -> plt.Axes:
    function_class = f3dasm.functions.find_function(results.function_name)
    f = function_class(dimensionality=2, scale_bounds=np.array(
        [[0., 1.], [0., 1.]]), noise=results.function_noise, offset=False)
    _, ax = f.plot(show=False, px=300, ax=ax, orientation='2D')
    ax.set_title(title)
    return ax


def rating_table(results: xr.Dataset, title: str, ax: plt.Axes) -> plt.Axes:
    ax.axis('off')
    ax.axis('tight')

    rdc = RecommenderDatasetCreator(results, jobnumber=0)
    df_rdc = rdc.create_recommender_data_interactions_csv(scaler='Standard')

    # create dict mapping of optimizer_names
    optimizer_names = results.optimizer.values
    optimizer_names_dict = {i+1: optimizer_names[i] for i, _ in enumerate(optimizer_names)}
    optimizer_names_dict

    # in the column userID, map the values to optimizer_names
    df_rdc['userID'] = df_rdc['userID'].map(optimizer_names_dict)

    ax.set_title(results.function_features)

    ax.table(cellText=df_rdc.values, colLabels=df_rdc.columns, loc='center')
    return ax


def plot_four_figures(results: xr.Dataset) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10))
    title = f"{results.function_name} - noise: {results.function_noise} - {results.function_dimensionality}D"
    fig.suptitle(f'Results of the optimization: {title}', fontsize=16)

    axs = []
    axs.append(fig.add_subplot(2, 3, 1))
    axs.append(fig.add_subplot(2, 3, 2))
    axs.append(fig.add_subplot(2, 3, 3))
    axs.append(fig.add_subplot(2, 3, 4, projection='3d'))
    axs.append(fig.add_subplot(2, 3, 5))
    axs.append(fig.add_subplot(2, 3, 6))
    # axs = axs.flatten()
    plot_normalized_objective(results, 'Absolute objective value', ax=axs[0])
    plot_auc(results, 'Area under curve', ax=axs[1])
    plot_margin_of_victory(results, 'Margin of victory', ax=axs[2])
    plot_loss_landscape_3d(results, '3D Loss landscape', ax=axs[3])
    rating_table(results, 'Rating table', ax=axs[4])
    plot_loss_landscape_2d(results, '2D Loss landscape', ax=axs[5])

    return fig
