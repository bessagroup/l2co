"""
Module containing functions that choose an optimizer based on content-based
features.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
# =============================================================================


def _get_labels(experimentdata: ExperimentData, index: int) -> str:
    """
    Create a string with the labels for a given itemID.

    Parameters
    ----------
    experimentdata : ExperimentData
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

    if f.noise is None:
        f.noise = 0.0

    # Append features
    if f.noise == 0.0:
        categories_list.append('noiseless')
    if f.noise > 0.0:
        categories_list.append('noisy')
    if f.convex:
        categories_list.append('convex')
    if f.separable:
        categories_list.append('seperable')
    if f.multimodal:
        categories_list.append('multimodal')

    categories_list.append(f"dim={dim}")

    return ",".join(categories_list)


def get_labels(experimentdata: ExperimentData) -> xr.DataArray:
    """
    Get labels for each itemID in the experimentdata.

    Parameters
    ----------
    experimentdata : ExperimentData
        The experimentdata object.

    Returns
    -------
    xr.DataArray
        A dataframe with the labels for each itemID.
    """
    df = pd.DataFrame([_get_labels(experimentdata, i)
                       for i in experimentdata.index], columns=['labels'],
                      index=experimentdata.index)

    df.index.name = 'itemID'

    # Encode the unique words:
    unique_words = set(','.join(df['labels']).split(','))

    for word in unique_words:
        df[word] = df['labels'].apply(lambda x: 1 if word in x else 0)

    df.drop('labels', axis=1, inplace=True)
    return xr.DataArray(
        df.values,
        coords={'itemID': df.index, 'features': df.columns},
        dims=['itemID', 'features'], name='labels')


def cosine_sim(single_vec, training_set):
    return cosine_similarity(single_vec.reshape(1, -1), training_set.T).ravel()


def calculate_similarity_features(
        ds_single: xr.DataArray, ds_features: xr.DataArray) -> xr.DataArray:

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

    return cosine_similarities.rename('features')


def predict_zero(ds_performance, ds_features) -> str:
    ds_comb = xr.merge([ds_performance, ds_features])

    # only display the itemIDs where features > 0.6
    ds_comb = ds_comb.where(ds_comb.features > 0.6, drop=True)

    scaler = MinMaxScaler()

    scaled = xr.apply_ufunc(
        scaler.fit_transform, ds_comb.performance,
        input_core_dims=[['optimizer', 'itemID']],
        output_core_dims=[['optimizer', 'itemID']],
        vectorize=True, dask='parallelized')

    prediction = ds_comb.optimizer[int(
        (ds_comb.features * scaled).sum(dim='itemID').argmin())].item()
    return prediction
