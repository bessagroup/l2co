"""
Module containing functions that choose an optimizer based on content-based
features.

Functions
---------

get_labels
    Get labels for each itemID in the experimentdata.
get_labels_one
    Create a string with the labels for a given itemID.
get_not_encoded_labels
    Get the labels that are not encoded.
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

# Local
from l2co import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def get_labels_one(experimentdata: ExperimentData, index: int) -> str:
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

    if f.convex:
        categories_list.append('convex')
    if f.separable:
        categories_list.append('seperable')
    if f.multimodal:
        categories_list.append('multimodal')
    if f.differentiable:
        categories_list.append('differentiable')

    return ",".join(categories_list)


def get_not_encoded_labels(
        experimentdata: ExperimentData, index: int) -> Dict[str, Any]:
    """
    Get the labels that are not encoded.

    Parameters
    ----------
    experimentdata : f3dasm.ExperimentData
        ExperimentData object.
    index : int
        index of the experiment sample.

    Returns
    -------
    dict
        dictionary with the labels that are not encoded.
    """
    experiment_sample = experimentdata.get_experiment_sample(index)
    noise = experiment_sample.get('noise')
    dim = experiment_sample.get('dimensionality')

    labels = {}

    if noise is None:
        labels['noise'] = 0.0
    else:
        labels['noise'] = noise

    labels['dim'] = dim

    return labels


def get_labels(experimentdata: ExperimentData,
               n_realizations: int) -> xr.DataArray:
    """
    Get labels for each optimization problem in the experimentdata.

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

    df = pd.DataFrame([get_labels_one(experimentdata, i)
                       for i in experimentdata.index], columns=['labels'],
                      index=experimentdata.index)

    df.index.name = 'itemID'

    # Encode the unique words:
    unique_words = set(','.join(df['labels']).split(','))

    # check if empty string is in unique_words
    if '' in unique_words:
        unique_words.remove('')

    for word in unique_words:
        df[word] = df['labels'].apply(lambda x: 1 if word in x else 0)

    df.drop('labels', axis=1, inplace=True)

    df_non_encoded = pd.DataFrame(
        [get_not_encoded_labels(experimentdata, i)
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


def get_labels_arbitrary(
        experimentdata: ExperimentData, n_realizations: int) -> xr.DataArray:
    X = experimentdata.to_xarray().input.rename(
        {'iterations': 'itemID', 'input_dim': 'features'})
    return X.expand_dims(
        dim={'realization': np.arange(n_realizations)}
    ).stack(problem=['realization', 'itemID']).T.rename('labels')
