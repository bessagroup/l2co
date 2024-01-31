"""
Module containing functions that compute the performance of optimizers

Functions
---------

get_performance
    Generate a dataset that captures the performance of an optimizer
    on a benchmark function as function of the number of function evaluations.
query_t
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.
is_sparse
    Check if the performance DataArray is sparse.
collaborative_filtering
    Fill in missing values in the performance DataFrame using
    collaborative filtering.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from time import perf_counter

# Third-party
import numpy as np
import pandas as pd
import surprise
import xarray as xr
from f3dasm import ExperimentData
from recommenders.models.surprise.surprise_utils import \
    compute_ranking_predictions

# Local
from ._logging import logger
from .io import _validate_path

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def output_cummin(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute the cumulative minimum of the output values.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset object containing the output values of the
        optimization process.

    Returns
    -------
    xarray.DataArray
        xarray DataArray object containing the cumulative minimum of the
        output values.

    Note
    ----
    The cumulative minimum is computed along the 'iterations' dimension
    with the `numpy.minimum.accumulate` function.
    """
    return xr.DataArray(
        np.minimum.accumulate(ds.output.values, axis=ds.output.get_axis_num(
            'iterations')), coords=ds.output.coords, dims=ds.output.dims)


def randomsearch_baseline(
        rs: xr.Dataset, size: int, shuffles: int = 99) -> np.ndarray:
    """
    Compute the random search baseline for the performance of an optimizer
    on a benchmark function.

    Parameters
    ----------
    rs : xarray.Dataset
        xarray Dataset object containing the output values of the
        random search process.
    size : int
        number of function evaluations of the dataset
    shuffles : int
        number of shuffles to compute the baseline

    Returns
    -------
    numpy.ndarray
        numpy ndarray object containing the baseline for the performance of
        an optimizer on a benchmark function.

    Note
    ----
    The baseline is computed by shuffling the output values of the random
    search process and taking the median of the cumulative minimum of the
    shuffled output values.
    """

    rs_data = np.random.choice(rs['output'].values.ravel(), size=size)
    shuffled_arrays = np.minimum.accumulate(rs_data).reshape(1, -1)

    for _ in range(shuffles):
        # Shuffle the datapoints
        np.random.shuffle(rs_data)

        # Concatenate the accumulated minimum of the shuffled data
        shuffled_arrays = np.concatenate(
            [shuffled_arrays, np.minimum.accumulate(rs_data).reshape(1, -1)])

    # take the median over shuffled_arrays first dimension;
    return np.median(shuffled_arrays, axis=0)


def _calculate_metric(
        ds: xr.Dataset, baseline: np.ndarray) -> xr.DataArray:
    """
    Compute the performance of an optimizer on a benchmark function.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset object containing the output values of the
        optimization process.
    baseline : numpy.ndarray
        numpy ndarray object containing the baseline for the performance of
        an optimizer on a benchmark function.

    Returns
    -------
    xarray.DataArray
        xarray DataArray object containing the performance of the optimizer
        on a benchmark function.

    Note
    ----
    The performance is computed by taking the logarithm of the ratio of the
    cumulative minimum of the output values of the optimization process and
    the baseline.
    """
    ds['output_cummin'] = output_cummin(ds)

    # Find the minimum value of the output
    y_min = ds.output.min()

    # Select the cumulative minimum and subtract the minimum value
    # Add one to make sure we don't take the log of zero
    a = ds.output_cummin.sel(
        output_dim='y').median('realization') - y_min + 1.0
    b = baseline - y_min.values + 1.0

    return np.log(a/b)


def get_metric(
        data: ExperimentData, index: int):
    """
    Compute the performance of an optimizer on a benchmark function.

    Parameters
    ----------
    data : f3dasm.ExperimentData
        ExperimentData object containing the parameterization of
        the benchmark functions and the results of the optimization
        process for a selection of optimizers.
    index : int
        index of the experiment sample

    Returns
    -------
    xarray.DataArray
        xarray DataArray object containing the performance of the optimizer
        on a benchmark function.
    """

    # Retrieve the ExperimentSample of the given index
    experiment_sample = data.get_experiment_sample(index)

    # Load the optimization dynamics data
    ds = xr.open_mfdataset(
        [data.project_dir / path for key, path in
         experiment_sample.output_data_with_references.items()
         if _validate_path(
             path, data.project_dir) and key != 'random_samples'])

    # Load the random search data
    rs = xr.open_dataset(
        data.project_dir / experiment_sample.
        output_data_with_references['random_samples'])

    # Establish the baseline
    baseline = randomsearch_baseline(rs, size=len(ds.iterations))

    # Return the calculated performance metric
    return _calculate_metric(ds, baseline)


def get_metric_preprocessed(
        ds: xr.Dataset, ds_samples: xr.Dataset, index: int) -> xr.DataArray:
    """
    Compute the performance of an optimizer on a benchmark function.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset object containing the output values of the
        optimization process.
    ds_samples : xarray.Dataset
        xarray Dataset object containing the output values of the
        random search process.
    index : int
        index of the experiment sample

    Returns
    -------
    xarray.DataArray
        xarray DataArray object containing the performance of the optimizer
    """
    return _calculate_metric(
        ds.sel(itemID=index),
        randomsearch_baseline(ds_samples.sel(itemID=index),
                              size=len(ds.iterations)))


def get_performance_preprocessed(ds_all: xr.Dataset,
                                 ds_random: xr.Dataset) -> xr.DataArray:
    """
    Get the performance of the optimizers on the benchmark functions from
    preprocessed files

    Parameters
    ----------
    ds_all : xarray.Dataset
        xarray Dataset object containing the output values of the
        optimization process.
    ds_random : xarray.Dataset
        xarray Dataset object containing the output values of the
        random search process.

    Returns
    -------
    xarray.DataArray
        xarray DataArray object containing the performance of the optimizers
        on the benchmark functions.
    """
    t_start = perf_counter()
    combined = xr.concat([get_metric_preprocessed(ds_all, ds_random, i)
                         for i in ds_all.itemID], dim=ds_all.itemID)

    combined = combined.drop('output_dim')
    combined = combined.rename('performance')

    logger.info(
        f"Performance dataset preprocessed dataset! "
        f"(Took {perf_counter() - t_start:.2f} s)")
    return combined


def get_performance(experimentdata: ExperimentData) -> xr.DataArray:
    """
    Generate a dataset that captures the performance of an optimizer
    on a benchmark function as function of the number of function evaluations.

    Parameters
    ----------
    experimentdata : f3dasm.ExperimentData
        ExperimentData object containing the parameterization of
        the benchmark functions and the results of the optimization
        process for a selection of optimizers.

    Returns
    -------
    xarray.DataArray
        xarray DataArray object containing a floating point value for
        the performance of the optimizers on benchmark functions as a function
        of the number of function evaluations.

    Note
    ----
    The experimentdata object must contain the following attributes:
        - an 'random_samples' output parameter that contains a reference to
        a xarray Dataset object containing random samples. This needs to be
        specified for all the experiments.

    The dataarray has three coordinates:
        - optimizer: the name of the first and second optimizer (str)
        - itemID: an identifier for the benchmark function (int)
        - iterations: the number of function evaluations (int)

    """
    t_start = perf_counter()

    combined = xr.concat(
        [get_metric(experimentdata, i) for i in experimentdata.index],
        dim=xr.DataArray(experimentdata.index, dims='itemID'))

    # Drop the output dimension and rename the DataArray to 'performance'
    combined = combined.drop('output_dim')
    combined = combined.rename('performance')

    logger.info(
        f"Performance dataset created of : {experimentdata.project_dir}! "
        f"(Took {perf_counter() - t_start:.2f} s)")
    return combined


def query_t(performance_dataset: xr.Dataset, t: int) -> xr.DataArray:
    """
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.

    Parameters
    ----------

    performance_dataset : xarray.Dataset
        xarray Dataset object containing a floating point value for
        the performance of the optimizers on benchmark functions as a function
        of the number of function evaluations.
    t : int
        number of function evaluations

    Returns
    -------
    xarray.DataArray
        xarray DataArray containing the performance of the optimizers
        after a given number of function evaluations.
    """
    if t > performance_dataset.iterations.max():
        raise ValueError(
            f'Number of iterations {t} is larger than the maximum number of '
            f'iterations in the performance dataset '
            f'{performance_dataset.iterations.max().values}')

    ds_performance: xr.DataArray = performance_dataset.sel(
        iterations=t, drop=True)

    logger.info(f"Query t={t} from performance dataset")

    return ds_performance


def is_sparse(ds: xr.DataArray) -> bool:
    """
    Check if the performance DataArray is sparse.

    Parameters
    ----------
    ds : xarray.DataArray
        The performance DataArray

    Returns
    -------
    bool
        True if there are missing values (NaN), False otherwise.
    """
    return ds.isnull().values.any()


def collaborative_filtering(sparse_ds: xr.DataArray) -> xr.DataArray:
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
    # Check if there is anything to matrix complete
    if not is_sparse(sparse_ds):
        return sparse_ds

    df_performance = sparse_ds.to_dataframe().reset_index().dropna()

    reader = surprise.Reader(rating_scale=(
        df_performance['performance'].min(),
        df_performance['performance'].max()))

    train_set = surprise.Dataset.load_from_df(
        df_performance, reader=reader).build_full_trainset()

    svd = surprise.SVD(random_state=0, n_factors=200,
                       n_epochs=30, verbose=False)
    svd.fit(train_set)

    df_predictions = compute_ranking_predictions(
        svd, df_performance, usercol='itemID',
        itemcol='optimizer', remove_seen=True)
    df_predictions.columns = df_performance.columns

    complete_df = pd.concat(
        [df_performance, df_predictions], ignore_index=True)

    # cast to xr.DataArray
    return complete_df.set_index(
        ['itemID', 'optimizer']).to_xarray()['performance']
