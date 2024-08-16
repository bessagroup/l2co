"""
Module containing functions that choose an optimizer based on content-based
features.

Functions
---------

"""

#                                                                       Modules
# =============================================================================

# Third-party
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData

# Local
from . import FUNCTION_MAPPING

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def encoded_labels(x: pd.Series):
    categories_list = []
    f = FUNCTION_MAPPING[x['function_name'].lower()]

    if f.convex:
        categories_list.append('convex')
    if f.separable:
        categories_list.append('seperable')
    if f.multimodal:
        categories_list.append('multimodal')
    if f.differentiable:
        categories_list.append('differentiable')

    return ",".join(categories_list)


def non_encoded_labels(x: pd.Series):
    labels = {}

    labels['noise'] = x['noise'] if x['noise']\
        is not None else 0.0
    labels['dim'] = x['dimensionality']

    return pd.Series(labels)


def get_labels(experimentdata: ExperimentData,
               data: xr.Dataset) -> xr.DataArray:

    data.itemID.load()

    e_data = experimentdata.select(data.itemID)
    e_data._reset_index()

    df_input, df_output = e_data.to_pandas()

    df = pd.DataFrame(df_input.apply(encoded_labels, axis=1),
                      columns=['labels'], index=data.itemID)

    df.index.name = 'itemID'

    # Encode the unique words:
    unique_words = set(','.join(df['labels']).split(','))

    # check if empty string is in unique_words
    if '' in unique_words:
        unique_words.remove('')

    for word in unique_words:
        df[word] = df['labels'].apply(lambda x: 1 if word in x else 0)

    df.drop('labels', axis=1, inplace=True)

    df_non_encoded = df_input.apply(
        non_encoded_labels, axis=1).set_index(data.itemID.values)

    # combine df and df_non_encoded
    df = pd.concat([df, df_non_encoded], axis=1)

    return xr.DataArray(
        df.values,
        coords={'problem': data.problem,
                'features': df.columns},
        dims=['problem', 'features'],
        name='labels').assign_coords({'itemID': data.itemID})
