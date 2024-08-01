from pathlib import Path
from typing import Iterable, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # NOQA
from cycler import cycler


def concat_df(csv_files_list: List[Path]) -> pd.DataFrame:
    df_all = None
    for file in csv_files_list:
        # read the csv file into a pandas dataframe
        df = pd.read_csv(file)

        if df_all is None:
            df_all = df
            continue

        unique_column = df.columns.difference(df_all.columns)
        df_all[unique_column] = df[unique_column]

    # remove iteration column
    if 'iterations' in df_all.columns:
        df_all = df_all.drop(columns=['iterations'])

    if df_all.index.name != 'f':
        df_all.set_index('f', inplace=True)

    return df_all


def group_optimizers_startswith(optimizers: Iterable[str]) -> List[List[str]]:
    unique_optimizer = set(o.split('+')[0] for o in optimizers)

    # filter optimizers in seperate lists based if they
    # start with any in unique_optimizers
    grouped_optimizers = [
        [o for o in optimizers if o.startswith(u)] for u in unique_optimizer]

    return grouped_optimizers


def group_optimizers_endswith(optimizers: Iterable[str]) -> List[List[str]]:
    unique_optimizer = set(o.split('+')[0] for o in optimizers)

    # filter optimizers in seperate lists based if they
    # start with any in unique_optimizers
    grouped_optimizers = [
        [o for o in optimizers if o.endswith(u)] for u in unique_optimizer]

    return grouped_optimizers


def plot_per_starting_optimizer(df_all: pd.DataFrame) -> plt.Figure:
    optimizers = df_all.columns.drop(list(df_all.filter(regex='_')))

    grouped_optimizers = group_optimizers_startswith(optimizers)

    # find the product of two numbers that make len(grouped_optimizers)
    # this is used to create the subplots
    nrows = 2
    ncols = 2
    colours = ['r', 'g', 'b', 'k']

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    axs = axs.ravel()

    for c, ax, group in zip(colours, axs, grouped_optimizers):
        for opt in group:
            ax.plot(df_all.index, df_all[opt], label=opt, color=c)

        ax.set_title(group[0].split('+')[0])

        ax.legend()

    fig.suptitle('Performance profile per starting optimizer')

    return fig


def plot_per_ending_optimizer(df_all: pd.DataFrame) -> plt.Figure:
    optimizers = df_all.columns.drop(list(df_all.filter(regex='_')))

    grouped_optimizers = group_optimizers_endswith(optimizers)

    # find the product of two numbers that make len(grouped_optimizers)
    # this is used to create the subplots
    nrows = 2
    ncols = 2
    colours = ['r', 'g', 'b', 'k']

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    axs = axs.ravel()

    for c, ax, group in zip(colours, axs, grouped_optimizers):
        for opt in group:
            ax.plot(df_all.index, df_all[opt], label=opt, color=c)

        ax.set_title(group[0].split('+')[-1])

        ax.legend()

    fig.suptitle('Performance profile per ending optimizer')

    return fig


def plot_all(df_all: pd.DataFrame) -> plt.Figure:

    colours = plt.cm.Set1(np.linspace(0, 1, 9))

    default_cycler = (
        cycler(color=colours) * cycler(linestyle=['-', '--', ':']))

    plt.rc('axes', prop_cycle=default_cycler)

    fig, ax = plt.subplots(figsize=(10, 7))

    # remove iteration column
    if 'iterations' in df_all.columns:
        df_all = df_all.drop(columns=['iterations'])

    if df_all.index.name != 'f':
        df_all.set_index('f', inplace=True)

    ax.set_xscale('log')
    ax.set_xlabel('$f$')
    ax.set_ylabel('fraction of problems solved')

    for name in df_all.columns:

        # Combination optimizer
        if '+' in name:
            ax.plot(df_all.index, df_all[name], label=name, linewidth=1)

        # Strategy
        elif '_' in name:
            ax.plot(df_all.index, df_all[name], label=name, linewidth=2)

        else:
            ax.plot(df_all.index, df_all[name], label=name, linewidth=1)

    # put the legend under the figure and in a table like structure
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=4)
    return fig


@hydra.main(config_path="configs", config_name="post_process")
def main(config):
    root_path = Path(config.experiment_dir)

    # find all the csv files in subdirectories called 'performance
    csv_files = root_path.glob('**/performance_profile_csv/*.csv')
    df_all = concat_df(csv_files)
    fig_all = plot_all(df_all)
    fig_per_starting_optimizer = plot_per_starting_optimizer(df_all)
    fig_per_ending_optimizer = plot_per_ending_optimizer(df_all)

    # save
    fig_all.savefig('performance_profile.png', dpi=300, bbox_inches='tight')
    fig_per_starting_optimizer.savefig(
        'performance_profile_per_optimizer_starting.png',
        dpi=300, bbox_inches='tight')

    fig_per_ending_optimizer.savefig(
        'performance_profile_per_optimizer_ending.png',
        dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    plt.ioff()
    # plt.style.use('science')
    main()
