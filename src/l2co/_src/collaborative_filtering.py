"""
Module containing functions that use collaborative-filtering recommender system
techniques to fill in the missing values for the performance of optimizers.

Functions
---------
create_interaction_matrix_from_full_data
    Create a sparse matrix of the performances of optimizers
    on benchmark functions.
collaborative_filtering
    Compute the collaborative filtering predictions for the
    performance of optimizers on benchmark functions.
for_the_first_window
    Combine the performances of optimizers that start with
    the same optimizer in the first window.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
import pandas as pd
import surprise
import xarray as xr
from recommenders.models.surprise.surprise_utils import \
    compute_ranking_predictions

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def create_interaction_matrix_from_full_data(
        combined: xr.Dataset, budget: int) -> pd.DataFrame:
    """
    Create a sparse matrix of the performances of optimizers on benchmark
    functions.

    Parameters
    ----------
    combined : xarray.Dataset
        xarray Dataset object containing the raw data files for which the
        optimizer given is the first window
    budget : int
        number of function evaluations

    Returns
    -------
    pandas.DataFrame
        DataFrame containing a sparse matrix of the performances of optimizers
        on benchmark functions.

    Notes
    -----
    The sparse matrix is created by selecting the last iteration of the
    optimization process for each optimizer and optimization problem.
    """

    combined = combined.drop('output_dim')

    # change itemID column to problem
    combined = combined.rename({'itemID': 'problem'})

    df = combined.sel(iterations=budget-1,
                      drop=True).to_dataframe(name='rating')
    # remove missing values in df
    df = df.dropna()
    df = df.reset_index()
    return df


# def collaborative_filtering(df_interaction: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute the collaborative filtering predictions for the performance of
#     optimizers on benchmark functions.

#     Parameters
#     ----------
#     df_interaction : pandas.DataFrame
#         DataFrame containing the sparse matrix performances
#         between optimizers and optimization problems.

#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame containing a full matrix of the predicted performance of
#         optimizers on benchmark functions.

#     Notes
#     -----
#     The collaborative filtering predictions are computed using the
#     `surprise.SVD` algorithm.
#     """
#     reader = surprise.Reader(rating_scale=(
#         df_interaction['rating'].min(), df_interaction['rating'].max()))

#     train_set = surprise.Dataset.load_from_df(
#         df_interaction, reader=reader).build_full_trainset()

#     svd = surprise.SVD(random_state=0, n_factors=200,
#                        n_epochs=30, verbose=False)
#     svd.fit(train_set)

#     df_predictions = compute_ranking_predictions(
#         svd, df_interaction, usercol='problem',
#         itemcol='optimizer', remove_seen=True)
#     df_predictions.columns = df_interaction.columns
#     return pd.concat([df_interaction, df_predictions], ignore_index=True)

# def collaborative_filtering(performance: xr.DataArray):
#     """
#     Compute the collaborative filtering predictions for the performance of
#     optimizers on benchmark functions.

#     Parameters
#     ----------
#     performance : xarray.DataArray
#         DataArray containing the sparse matrix performances
#         between optimizers and optimization problems.

#     Returns
#     -------
#     xarray.DataArray
#         DataArray containing a full matrix of the predicted performance of
#         optimizers on benchmark functions.

#     Notes
#     -----
#     The collaborative filtering predictions are computed using the
#     `surprise.SVD` algorithm.
#     """
#     df = performance.sel(realization=0, drop=True).to_dataframe().dropna(
#         axis=0).reset_index()
#     n_realizations = np.unique(performance.realization).size

#     reader = surprise.Reader(rating_scale=(
#         df['performance'].min(), df['performance'].max()))

#     train_set = surprise.Dataset.load_from_df(
#         df, reader=reader).build_full_trainset()

#     svd = surprise.SVD(random_state=0, n_factors=200,
#                        n_epochs=30, verbose=False)
#     svd.fit(train_set)

#     df_predictions = compute_ranking_predictions(
#         svd, df, usercol='optimizer',
#         itemcol='itemID', remove_seen=True)
#     df_predictions.columns = df.columns

#     performance_full = xr.DataArray(
#         pd.concat([df, df_predictions], ignore_index=False).set_index(
#             ['optimizer', 'itemID']).unstack().droplevel(level=None, axis=1))
#     performance_full = performance_full.expand_dims({'realization': np.arange(
#         n_realizations)}).stack(problem=('realization', 'itemID'))
#     return performance_full


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


def remove_performance_values(
        ds_performance: xr.DataArray,
        ratio_to_remove: float) -> xr.DataArray:

    perf_mean = ds_performance.unstack().mean('realization')

    # Define the percentage of values to mask
    # percentage_to_mask = 20  # Change this as needed

    # Calculate the number of elements to mask
    total_elements = np.prod(perf_mean.shape)
    num_elements_to_mask = int(ratio_to_remove * total_elements)

    # Generate random indices to mask
    random_indices = np.random.choice(
        total_elements, size=num_elements_to_mask, replace=False)

    # Reshape the random indices to match the shape of the DataArray
    random_indices = np.unravel_index(random_indices, perf_mean.shape)

    # Create a mask array with False everywhere except at the random indices
    mask = perf_mean.isnull().values

    mask[random_indices] = True

    # Don't touch the first and last index to preserve the boundaries
    mask[:, 0] = False
    mask[:, -1] = False

    # Ensure that at least one value of every row is False
    for i in range(mask.shape[1]):
        if not np.any(~mask[:, i]):
            # Randomly select one index to set to False
            index_to_set_false = index_to_set_false = np.random.choice(
                np.where(perf_mean.isnull().values[:, i] == 0)[0])  # == False
            mask[index_to_set_false, i] = False

    # Apply the mask to the DataArray
    performance = perf_mean.where(~mask, drop=False)

    return performance
