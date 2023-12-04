"""
Module containing function that use collaborative-filtering recommender system
techniques to fill in the missing values for the performance of optimizers.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import pandas as pd
import surprise
import xarray as xr
from f3dasm import ExperimentData
from recommenders.models.surprise.surprise_utils import \
    compute_ranking_predictions

# Local
from .baselines import get_metric

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
# =============================================================================


def create_interaction_matrix_from_full_data(
        combined: xr.Dataset, budget: int) -> pd.DataFrame:

    combined = combined.drop('output_dim')

    # change itemID column to problem
    combined = combined.rename({'itemID': 'problem'})

    df = combined.sel(iterations=budget-1,
                      drop=True).to_dataframe(name='rating')
    # remove missing values in df
    df = df.dropna()
    df = df.reset_index()
    return df


def create_interaction_matrix(
        data: ExperimentData, budget: int,
        baseline: str = 'randomsearch') -> pd.DataFrame:

    combined = xr.concat(
        [get_metric(data, i, budget, baseline) for i in data.index],
        dim=xr.DataArray(data.index, dims='itemID'))

    combined = combined.drop('output_dim')

    # change itemID column to problem
    combined = combined.rename({'itemID': 'problem'})

    df = combined.sel(iterations=budget-1,
                      drop=True).to_dataframe(name='rating')
    # remove missing values in df
    df = df.dropna()
    df = df.reset_index()
    return df


def collaborative_filtering(df_interaction: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the collaborative filtering predictions for the performance of
    optimizers on benchmark functions.

    Parameters
    ----------
    df_interaction : pandas.DataFrame
        DataFrame containing the sparse matrix performances
        between optimizers and optimization problems.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing a full matrix of the predicted performance of
        optimizers on benchmark functions.

    Notes
    -----
    The collaborative filtering predictions are computed using the
    `surprise.SVD` algorithm.
    """
    reader = surprise.Reader(rating_scale=(
        df_interaction['rating'].min(), df_interaction['rating'].max()))

    train_set = surprise.Dataset.load_from_df(
        df_interaction, reader=reader).build_full_trainset()

    svd = surprise.SVD(random_state=0, n_factors=200,
                       n_epochs=30, verbose=False)
    svd.fit(train_set)

    df_predictions = compute_ranking_predictions(
        svd, df_interaction, usercol='problem',
        itemcol='optimizer', remove_seen=True)
    df_predictions.columns = df_interaction.columns
    return pd.concat([df_interaction, df_predictions], ignore_index=True)


def for_the_first_window(df_interaction: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the performances of optimizers that start with the same optimizer
    in the first window.

    Parameters
    ----------
    df_interaction : pandas.DataFrame
        DataFrame containing the sparse matrix performances
        between optimizers and optimization problems.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the performances of optimizers where the first
        window is combined.
    """
    startswith = []
    unique_optimizers = df_interaction['optimizer'].str.split(
        '+').str[0].unique()
    for opt in unique_optimizers:
        df_partial = df_interaction[df_interaction['optimizer'].str.startswith(
            opt)]

        # give all the values of optimizer the value Adam and collapse the rows
        df_partial['optimizer'] = opt

        # only select unique values of problem
        df_partial = df_partial.drop_duplicates(subset=['problem'])
        startswith.append(df_partial)

    df_startswith = pd.concat(startswith, ignore_index=True)

    # sort on problem column
    df_startswith = df_startswith.sort_values(by=['problem'])
    return df_startswith
