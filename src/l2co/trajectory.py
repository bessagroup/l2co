"""
Module containing functions that compute the trajectory similarity of
optimizers roll-outs.
"""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
# =============================================================================


def filter_dataset(experimentdata: ExperimentData,
                   optimizer: str) -> xr.Dataset:
    """
    Filter the experimentdata object for a given optimizer and number of
    function evaluations.

    Parameters
    ----------
    experimentdata : ExperimentData
        ExperimentData object containing the parameterization of
        the benchmark functions and the results of the optimization
        process for a selection of optimizers.
    optimizer : str
        name of the optimizer

    Returns
    -------
    xr.Dataset
        xarray Dataset object containing the raw data files for which the
        optimizer given is the first window
    """
    # Convert the output of the ExperimentData object to a pd.DataFrame
    _, df_output = experimentdata.to_pandas()

    # Filter the columns that contain the optimizer name
    df_output = df_output[[
        col for col in df_output.columns if col.startswith(optimizer)]]

    # Remove rows where all values are nan
    df_output = df_output.dropna(how='all')

    # Remove rows with the value 'ERROR'
    df_output = df_output[df_output != 'ERROR'].dropna(how='all')

    # For each row, create a new series
    # with the value of the first non NaN value
    raw_paths: pd.Series = df_output.apply(
        lambda row: experimentdata.project_dir / Path(
            row.first_valid_index()) / f"{row.name}.nc", axis=1)

    ds_similarity = xr.open_mfdataset(
        raw_paths, combine='nested',
        concat_dim=xr.DataArray([int(f.stem) for f in raw_paths],
                                dims='itemID'))

    # Change the coordinate optimizer to the optimizer
    ds_similarity.coords['optimizer'] = optimizer

    return ds_similarity


def get_trajectory(experimentdata: ExperimentData,
                   optimizer: str,
                   t_switch: int, t_start: int = 0) -> xr.DataArray:
    """
    Calculate a trajectoy similarity measure for the output of the optimizer.

    Parameters
    ----------
    experimentdata : ExperimentData
        ExperimentData object containing the parameterization of
        the benchmark functions and the results of the optimization
        process for a selection of optimizers.
    optimizer : str
        name of the optimizer
    t_switch : int
        number of function evaluations after which the optimizer
        switches to a different function
    t_start : int, optional
        number of function evaluations after which the optimizer
        starts to optimize, by default 0

    Returns
    -------
    xr.Dataset
        xarray Dataset object containing the similarity of the output of the
        optimizer
    """

    ds_similarity = filter_dataset(
        experimentdata, optimizer).sel(iterations=range(t_start, t_switch))

    ds_scaled = xr.apply_ufunc(
        np.log, (
            ds_similarity - (ds_similarity.output.min('iterations') - 1.0)).
        output, dask='parallelized')

    # Initialize the scaler
    scaler = StandardScaler()

    # for every realization and itemID apply the StandardScaler.fit_transform
    ds_scaled = xr.apply_ufunc(
        scaler.fit_transform, ds_scaled, input_core_dims=[
            ['iterations', 'output_dim']],
        output_core_dims=[['iterations', 'output_dim']], vectorize=True,
        dask='parallelized')

    return ds_scaled


def compute_cosine_similarity(single_vec, training_set):
    """
    Function to calculate the cosine similarity between a single vector and a
    training set of vectors.

    Parameters
    ----------
    single_vec : np.ndarray
        vector of shape (n_features, 1)
    training_set : np.ndarray
        matrix of shape (n_features, n_samples)

    Returns
    -------
    np.ndarray
        vector of shape (n_samples, 1)
    """
    return cosine_similarity(single_vec.T, training_set.T).flatten()


def calculate_trajectory_similarity(
        ds_single: xr.DataArray, ds_similarity: xr.Dataset) -> xr.DataArray:
    """
    Calculate the cosine similarity between a single vector and a training set
    of vectors.

    Parameters
    ----------
    ds_single : xr.DataArray
        xarray DataArray object containing the single vector
    ds_similarity : xr.Dataset
        xarray Dataset object containing the training set of vectors

    Returns
    -------
    xr.DataArray
        xarray DataArray object containing the cosine similarity between the
        single vector and the training set of vectors.
    """
    # combine the itemID and realization dimensions into one
    ds_similarity = ds_similarity.stack(problem=['itemID', 'realization'])

    # Apply function to calculate cosine similarity using apply_ufunc
    cosine_similarities: xr.DataArray = xr.apply_ufunc(
        compute_cosine_similarity,
        ds_single,
        ds_similarity.sel(output_dim='y', drop=True),
        input_core_dims=[['iterations', 'output_dim'],
                         ['iterations', 'problem']],
        output_core_dims=[['problem']],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True,
                            },  # Use allow_rechunk to handle chunking
        output_dtypes=[float]  # dtype of the output
    )

    # unstack cosine_similarities to match the original shape of ds_similarity
    cosine_similarities = cosine_similarities.unstack('problem')

    return cosine_similarities.rename('similarity')
