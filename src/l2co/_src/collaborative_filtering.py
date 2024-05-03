"""
Module containing functions that use collaborative-filtering recommender system
techniques to fill in the missing values for the performance of optimizers.

Functions
---------
collaborative_filtering
    Compute the collaborative filtering predictions for the
    performance of optimizers on benchmark functions.
remove_performance_values
    Remove a fraction of the performance values from the performance DataArray.
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
from sklearn.preprocessing import LabelEncoder

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def collaborative_filtering(da_performance: xr.DataArray) -> xr.DataArray:
    """
    Fill in missing values in the performance DataFrame using
    collaborative filtering.

    Parameters
    ----------
    sparse_ds : xarray.DataArray
        The performance DataArray with missing values

    Returns
    -------
    xarray.DataArray
        The performance DataFrame without missing values
    """

    df_performance = da_performance.to_dataframe(
    )['performance'].dropna().reset_index()
    combined_columns = df_performance['realization'].astype(
        str) + '+' + df_performance['itemID'].astype(str)

    encoder = LabelEncoder()
    encoded_columns = encoder.fit_transform(combined_columns)
    df_performance['encoding'] = encoded_columns

    df_performance = df_performance[['optimizer', 'encoding', 'performance']]

    reader = surprise.Reader(rating_scale=(
        df_performance['performance'].min(),
        df_performance['performance'].max()))

    train_set = surprise.Dataset.load_from_df(
        df_performance, reader=reader).build_full_trainset()

    svd = surprise.SVD(random_state=0, n_factors=200,
                       n_epochs=30, verbose=False)
    svd.fit(train_set)

    df_predictions = compute_ranking_predictions(
        svd, df_performance, usercol='optimizer',
        itemcol='encoding', remove_seen=True)

    df_predictions = df_predictions.rename(
        columns={'prediction': 'performance'})

    complete_df = pd.concat(
        [df_performance, df_predictions], ignore_index=True)

    complete_df['decoding'] = encoder.inverse_transform(
        complete_df['encoding'])

    complete_df['itemID'] = complete_df['decoding'].str.split(
        '+').str[1].astype(int)
    complete_df['realization'] = complete_df['decoding'].str.split(
        '+').str[0].astype(int)
    complete_df = complete_df[['optimizer',
                               'itemID', 'realization', 'performance']]

    return complete_df.set_index(
        ['itemID', 'realization', 'optimizer']
    ).to_xarray()['performance'].T.stack(
        problem=('realization', 'itemID')).sel(problem=da_performance.problem)

    # Check if there is anything to matrix complete
    # if not da_performance.isnull().values.any():
    #     return da_performance

    # df_performance = da_performance.to_dataframe().reset_index().dropna()

    # reader = surprise.Reader(rating_scale=(
    #     df_performance['performance'].min(),
    #     df_performance['performance'].max()))

    # train_set = surprise.Dataset.load_from_df(
    #     df_performance, reader=reader).build_full_trainset()

    # svd = surprise.SVD(random_state=0, n_factors=200,
    #                    n_epochs=30, verbose=False)
    # svd.fit(train_set)

    # df_predictions = compute_ranking_predictions(
    #     svd, df_performance, usercol='optimizer',
    #     itemcol='itemID', remove_seen=True)

    # df_predictions = df_predictions.rename(
    #     columns={'prediction': 'performance'})

    # complete_df = pd.concat(
    #     [df_performance, df_predictions], ignore_index=True)

    # # cast to xr.DataArray
    # return complete_df.set_index(
    #     ['itemID', 'optimizer']).to_xarray()['performance'].T


def remove_performance_values(
        da_performance: xr.DataArray,
        ratio_to_remove: float) -> xr.DataArray:

    perf_mean = da_performance.unstack()  # .mean('realization')

    # Calculate the number of elements to mask
    total_elements = np.prod(perf_mean.shape)
    num_elements_to_mask = int(ratio_to_remove * total_elements)

    # Generate random indices to mask
    random_indices = np.random.choice(
        total_elements, size=num_elements_to_mask, replace=False)

    # Reshape the random indices to match the shape of the DataArray
    random_indices = np.unravel_index(random_indices, perf_mean.shape)

    np_masked_entries = np.zeros(shape=perf_mean.shape, dtype=bool)
    np_masked_entries[random_indices] = True

    masked_entries = xr.DataArray(
        np_masked_entries, dims=perf_mean.dims, coords=perf_mean.coords)
    # masked_entries[random_indices] = True

    # Create a mask array with False everywhere except at the random indices
    mask = perf_mean.isnull()
    combined_mask = (mask | masked_entries)

    def check_mask(arr):
        if not arr.all():
            return arr

        new_arr = arr.copy()
        # set one random index to False
        random_index = np.random.randint(0, new_arr.size)
        new_arr[random_index] = False
        return new_arr

    checked_mask = xr.apply_ufunc(
        check_mask,
        combined_mask,
        input_core_dims=[['optimizer']],
        output_core_dims=[['optimizer']],
        dask='parallelized',
        vectorize=True,
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    # Apply the mask to the DataArray
    perf_sparse = perf_mean.where(~checked_mask, drop=False)

    return perf_sparse.stack(
        problem=('realization', 'itemID')).sel(problem=da_performance.problem)
