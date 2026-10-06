"""
Module for preprocessing the categorical
input features of the benchmark functions

Functions
---------
transformer_factory
    Factory function that returns a Pipeline for preprocessing the data.
preprocessing_factory
    Factory function that returns a ColumnTransformer for preprocessing
    the data.
"""
#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

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
    if dtype == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

    elif dtype == "scaler":
        return StandardScaler()

    else:
        raise ValueError(f"Unknown dtype {dtype}")


# =============================================================================


class EnsureColumnsTransformer(BaseEstimator, TransformerMixin):
    """Transformer that ensures a DataFrame has expected columns.

    Missing columns are filled with NaN and extra columns
    are dropped so the output always matches the column set
    seen during ``fit``.
    """

    def __init__(self, trained_columns=None):
        # Columns the model was trained on
        self.trained_columns = trained_columns

    def fit(self, X, y=None):
        """Learn the expected column names from X.

        Parameters
        ----------
        X : DataFrame
            Training data.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        EnsureColumnsTransformer
            Fitted transformer.
        """
        if self.trained_columns is None:
            self.trained_columns = list(X.columns)
        return self

    def transform(self, X):
        """Align columns of X to the trained column set.

        Parameters
        ----------
        X : array-like or DataFrame
            Data to transform.

        Returns
        -------
        DataFrame
            DataFrame with columns matching the trained set.
        """
        # Ensure DataFrame format
        X = pd.DataFrame(X)

        # Add missing columns with NaN
        for col in self.trained_columns:
            if col not in X.columns:
                X[col] = np.nan

        # Reorder columns to match the trained set
        return X[self.trained_columns]


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    """ColumnTransformer that preserves DataFrame format and column names."""

    def __init__(self, feature_types: dict[str, str]):
        self.feature_types = feature_types
        self.column_transformer = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the ColumnTransformer and store feature names."""

        filtered_feature_types = {
            k: v for k, v in self.feature_types.items() if k in set(X.columns)
        }

        transformers = [
            (feature_name, transformer_factory(feature_type), [feature_name])
            for feature_name, feature_type in filtered_feature_types.items()
        ]

        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            force_int_remainder_cols=False,
        )
        self.column_transformer.fit(X, y)
        self.feature_names_out_ = self.get_feature_names_out(X)

        return self

    def transform(self, X):
        """Transform and return a DataFrame with column names preserved."""
        transformed_array = self.column_transformer.transform(X)
        return pd.DataFrame(
            transformed_array, columns=self.feature_names_out_, index=X.index
        )

    def get_feature_names_out(self, X):
        """Extracts feature names from the transformers."""
        feature_names = []
        for _, transformer, cols in self.column_transformer.transformers_:
            if transformer == "passthrough":
                feature_names.extend(cols)
            elif hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out())
        return feature_names


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
    return DataFrameColumnTransformer(
        [
            (feature_name, transformer_factory(feature_type), [feature_name])
            for feature_name, feature_type in feature_types.items()
        ],
        remainder="drop",
    )
