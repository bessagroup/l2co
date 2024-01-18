
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
__status__ = 'Planning'
# =============================================================================


def ordinal_encoder_factory():
    return OrdinalEncoder(handle_unknown='use_encoded_value',
                          unknown_value=-1)


def standard_scaler_factory():
    return StandardScaler()


# =============================================================================


FEATURE_TYPES = {'convex': 'categorical', 'multimodal': 'categorical',
                 'seperable': 'categorical',
                 'noise': 'categorical', 'dim': 'ordinal'}

# =============================================================================


def transformer_factory(dtype: str) -> Pipeline:
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
    return make_column_transformer(
        *[(transformer_factory(feature_type), [feature_name])
          for feature_name, feature_type in feature_types.items()],
        remainder='passthrough')
