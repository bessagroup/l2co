"""
Module for preprocessing the categorical
input features of the benchmark functions

Functions
---------
ordinal_encoder_factory
    Returns an OrdinalEncoder
standard_scaler_factory
    Returns a StandardScaler
transformer_factory
    Factory function that returns a Pipeline for preprocessing the data.
preprocessing_factory
    Factory function that returns a ColumnTransformer for preprocessing
    the data.
"""
#                                                                       Modules
# =============================================================================

# Third-party
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def ordinal_encoder_factory() -> OrdinalEncoder:
    """
    Returns an OrdinalEncoder

    Returns
    -------
    OrdinalEncoder
        An ordinal encoder
    """
    return OrdinalEncoder(handle_unknown='use_encoded_value',
                          unknown_value=-1)


def standard_scaler_factory() -> StandardScaler:
    """
    Returns a standard scaler

    Returns
    -------
    StandardScaler
        A standard scaler
    """
    return StandardScaler()


# =============================================================================


FEATURE_TYPES = {'convex': 'categorical', 'multimodal': 'categorical',
                 'seperable': 'categorical',
                 'noise': 'categorical', 'dim': 'ordinal'}

# =============================================================================


def transformer_factory(dtype: str) -> Pipeline:
    """
    Factory function that returns a Pipeline for preprocessing the data.

    Parameters
    ----------
    dtype : str
        The type of the feature.

    Returns
    -------
    Pipeline
        A pipeline for preprocessing the data.
    """
    if dtype == 'categorical':
        return Pipeline(steps=[("encoder", ordinal_encoder_factory())])

    elif dtype == 'ordinal':
        return Pipeline(steps=[("encoder", ordinal_encoder_factory())])

    elif dtype == 'scaler':
        return Pipeline(steps=[("scaler", standard_scaler_factory())])

    else:
        raise ValueError(f'Unknown dtype {dtype}')

# =============================================================================


def preprocessing_factory(feature_types: dict) -> ColumnTransformer:
    """
    Factory function that returns a ColumnTransformer
    for preprocessing the data.

    Parameters
    ----------
    feature_types : dict
        A dictionary containing the feature names and their types.

    Returns
    -------
    ColumnTransformer
        A column transformer for preprocessing the data.
    """
    return make_column_transformer(
        *[(transformer_factory(feature_type), [feature_name])
          for feature_name, feature_type in feature_types.items()],
        remainder='passthrough')
