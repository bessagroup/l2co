"""
Module containing functions that choose an optimizer based on content-based
features.

Functions
---------

get_labels
    Get labels for each itemID in the experimentdata.
cosine_sim
    Compute the cosine similarity between a single vector and a training set.
calculate_similarity_features
    Calculate the cosine similarity between a single vector and a training set.
predict_zero
    Predict the optimizer before considering evaluating the function. This
    is done by calculating the cosine similarity between the features of the
    function to be evaluated and the features of the problems in the
    training set.
"""

#                                                                       Modules
# =============================================================================

# Standard
from time import perf_counter
from typing import Any, Dict

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from ._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def _get_labels(experimentdata: ExperimentData, index: int) -> str:
    """
    Create a string with the labels for a given itemID.

    Parameters
    ----------
    experimentdata : f3dasm.ExperimentData
        ExperimentData object.
    index : int
        index of the experiment sample.

    Returns
    -------
    str
        string which contains comma separeted labels of the benchmark function.
    """
    experiment_sample = experimentdata.get_experiment_sample(index)
    function_name = experiment_sample.get('function_name')
    dim = experiment_sample.get('dimensionality')

    f = _datagenerator_factory(function_name, domain=dim)

    categories_list = []

    # if f.noise is None:
    #     f.noise = 0.0

    # Append features
    # if f.noise == 0.0:
    #     categories_list.append('noiseless')
    # if f.noise > 0.0:
    #     categories_list.append('noisy')
    if f.convex:
        categories_list.append('convex')
    if f.separable:
        categories_list.append('seperable')
    if f.multimodal:
        categories_list.append('multimodal')

    # categories_list.append(f"dim={dim}")
    # check if categories_list is empty

    return ",".join(categories_list)


def _get_not_encoded_labels(
        experimentdata: ExperimentData, index: int) -> Dict[str, Any]:
    experiment_sample = experimentdata.get_experiment_sample(index)
    function_name = experiment_sample.get('function_name')
    dim = experiment_sample.get('dimensionality')

    f = _datagenerator_factory(function_name, domain=dim)

    labels = {}

    if f.noise is None:
        labels['noise'] = 0.0
    else:
        labels['noise'] = f.noise

    labels['dim'] = dim
    # labels['noise'] = f.noise

    return labels


def get_labels(experimentdata: ExperimentData, n_realizations: int) -> xr.DataArray:
    """
    Get labels for each itemID in the experimentdata.

    Parameters
    ----------
    experimentdata : f3dasm.ExperimentData
        The experimentdata object.
    n_realizations: int
        The number of realizations.

    Returns
    -------
    xarray.DataArray
        A dataframe with the labels for each itemID.
    """
    t_start = perf_counter()

    df = pd.DataFrame([_get_labels(experimentdata, i)
                       for i in experimentdata.index], columns=['labels'],
                      index=experimentdata.index)

    df.index.name = 'itemID'

    # Encode the unique words:
    unique_words = set(','.join(df['labels']).split(','))
    unique_words.remove('')

    for word in unique_words:
        df[word] = df['labels'].apply(lambda x: 1 if word in x else 0)

    df.drop('labels', axis=1, inplace=True)

    df_non_encoded = pd.DataFrame(
        [_get_not_encoded_labels(experimentdata, i)
         for i in experimentdata.index],
        index=experimentdata.index)

    # combine df and df_non_encoded
    df = pd.concat([df, df_non_encoded], axis=1)

    xarr = xr.DataArray(
        df.values,
        coords={'itemID': df.index, 'features': df.columns},
        dims=['itemID', 'features'], name='labels')

    logger.info(
        f"Labels dataset created of : {experimentdata.project_dir}! "
        f"(Took {perf_counter() - t_start:.2f} s)")

    return xarr.expand_dims(
        dim={'realization': np.arange(n_realizations)}
    ).stack(problem=['realization', 'itemID']).T


def cosine_sim(single_vec, training_set) -> np.ndarray:
    """
    Compute the cosine similarity between a single vector and a training set.

    Parameters
    ----------
    single_vec : numpy.ndarray
        A single vector.
    training_set : numpy.ndarray
        A training set.

    Returns
    -------
    numpy.ndarray
        The cosine similarity between the single vector and the training set.
    """
    return cosine_similarity(single_vec.reshape(1, -1), training_set.T).ravel()


def calculate_similarity_features(
        ds_single: xr.DataArray, ds_features: xr.DataArray) -> xr.DataArray:
    """
    Calculate the cosine similarity between a single vector and a training set.

    Parameters
    ----------
    ds_single : xarray.DataArray
        A single vector.
    ds_features : xarray.DataArray
        A training set.

    Returns
    -------
    xarray.DataArray
        The cosine similarity between the single vector and the training set.
    """
    cosine_similarities: xr.DataArray = xr.apply_ufunc(
        cosine_sim,
        ds_single,
        ds_features,
        input_core_dims=[['features'],
                         ['features', 'itemID']],
        output_core_dims=[['itemID']],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True,
                            },  # Use allow_rechunk to handle chunking
        output_dtypes=[float]  # dtype of the output
    )

    cosine_similarities = cosine_similarities.rename('features')
    logger.info("Similarity features calculated!")

    return cosine_similarities


def predict_zero(
        ds_performance: xr.DataArray, ds_features: xr.DataArray) -> str:
    """
    Predict the optimizer before considering evaluating the function. This
    is done by calculating the cosine similarity between the features of the
    function to be evaluated and the features of the problems in the
    training set.

    Parameters
    ----------
    ds_performance : xarray.DataArray
        The performance of the optimizers on the training set.
    ds_features : xarray.DataArray
        The features of the training set.

    Returns
    -------
    str
        The name of the optimizer that is predicted to perform best.
    """

    ds_comb = xr.merge([ds_performance, ds_features])

    # only display the itemIDs where features > 0.6
    ds_comb = ds_comb.where(ds_comb.features > 0.6, drop=True)

    scaler = MinMaxScaler()

    scaled = xr.apply_ufunc(
        scaler.fit_transform, ds_comb.performance,
        input_core_dims=[['optimizer', 'itemID']],
        output_core_dims=[['optimizer', 'itemID']],
        vectorize=True, dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True,
                            },  # Use allow_rechunk to handle chunking
    )

    prediction = ds_comb.optimizer[int(
        (ds_comb.features * scaled).sum(dim='itemID').argmin())].item()
    return prediction
