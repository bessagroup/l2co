
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import surprise
import xarray as xr
from f3dasm import ExperimentData
from recommenders.models.surprise.surprise_utils import \
    compute_ranking_predictions


def _output_cummin(ds: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        np.minimum.accumulate(ds.output.values, axis=ds.output.get_axis_num(
            'iterations')), coords=ds.output.coords, dims=ds.output.dims)


def _randomsearch_baseline(
        rs: xr.Dataset, size: int, shuffles: int = 99) -> np.ndarray:

    rs_data = np.random.choice(rs['output'].values.ravel(), size=size)
    shuffled_arrays = np.minimum.accumulate(rs_data).reshape(1, -1)

    for _ in range(shuffles):
        np.random.shuffle(rs_data)
        shuffled_arrays = np.concatenate(
            [shuffled_arrays, np.minimum.accumulate(rs_data).reshape(1, -1)])

    # take the median over shuffled_arrays first dimension;

    return np.median(shuffled_arrays, axis=0)


def _calculate_metric(
        ds: xr.Dataset, baseline: np.ndarray) -> xr.DataArray:
    ds['output_cummin'] = _output_cummin(ds)
    y_min = ds.output.min()

    a = ds.output_cummin.sel(
        output_dim='y').median('realization') - y_min + 1.0
    b = baseline - y_min.values + 1.0

    return np.log(a/b)


def _validate_path(path: str | float, project_dir: Path) -> bool:
    if isinstance(path, str):
        return Path(project_dir / path).exists()

    return False


def _get_metric(
        data: ExperimentData, index: int):

    experiment_sample = data.get_experiment_sample(index)

    ds = xr.open_mfdataset(
        [data.project_dir / path for key, path in
         experiment_sample.output_data_with_references.items()
         if _validate_path(
             path, data.project_dir) and key != 'random_samples'])

    rs = xr.open_dataset(
        data.project_dir / experiment_sample.
        output_data_with_references['random_samples'])

    baseline = _randomsearch_baseline(rs, size=len(ds.iterations))

    return _calculate_metric(ds, baseline)


def get_performance(experimentdata: ExperimentData) -> xr.Dataset:
    """
    Generate a dataset that captures the performance of an optimizer
    on a benchmark function as function of the number of function evaluations.

    Parameters
    ----------
    experimentdata : ExperimentData
        ExperimentData object containing the parameterization of
        the benchmark functions and the results of the optimization
        process for a selection of optimizers.

    Returns
    -------
    xr.Dataset
        xarray Dataset object containing a floating point value for
        the performance of the optimizers on benchmark functions as a function
        of the number of function evaluations.

    Note
    ----
    The experimentdata object must contain the following attributes:
        - an 'random_samples' output parameter that contains a reference to
        a xarray Dataset object containing random samples. This needs to be
        specified for all the experiments.

    The dataset has three coordinates:
        - optimizer: the name of the first and second optimizer (str)
        - itemID: an identifier for the benchmark function (int)
        - iterations: the number of function evaluations (int)
    """
    combined = xr.concat(
        [_get_metric(experimentdata, i) for i in experimentdata.index],
        dim=xr.DataArray(experimentdata.index, dims='itemID'))

    combined = combined.drop('output_dim')

    return combined.rename('performance')


def query_t(performance_dataset: xr.Dataset, t: int) -> xr.DataArray:
    """
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.

    Parameters
    ----------

    performance_dataset : xr.Dataset
        xarray Dataset object containing a floating point value for
        the performance of the optimizers on benchmark functions as a function
        of the number of function evaluations.
    t : int
        number of function evaluations

    Returns
    -------
    xr.DataArray
        xarray DataArray containing the performance of the optimizers
        after a given number of function evaluations.
    """
    if t > performance_dataset.iterations.max():
        raise ValueError(
            f'Number of iterations {t} is larger than the maximum number of '
            f'iterations in the performance dataset '
            f'{performance_dataset.iterations.max()}')

    ds_performance: xr.DataArray = performance_dataset.sel(
        iterations=t, drop=True)
    # ds_performance.reset_index(inplace=True)
    # df_performance = df_performance.pivot(
    #     index='itemID', columns='optimizer', values='performance')

    return ds_performance


def is_sparse(ds: xr.DataArray) -> bool:
    """
    Check if the performance DataArray is sparse.

    Parameters
    ----------
    ds : ds.DataArray
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
    sparse_ds : xr.DataArray
        The performance DataArray with missing values

    Returns
    -------
    xr.DataArray
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
