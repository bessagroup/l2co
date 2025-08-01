"""
Module for creating the L2CO model.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import jax.numpy as jnp
import pandas as pd
from sklearn.pipeline import Pipeline

from ._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def train_stage0_model(model: Pipeline, x: pd.DataFrame, y: jnp.ndarray
                       ) -> Pipeline:
    """
    Train a stage 0 model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The model pipeline to train.
    x : pandas.DataFrame
        Input features.
    y : jax.numpy.ndarray
        Target values.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The trained model pipeline.
    """
    logger.debug(f"Training stage0 model {model}")
    model.fit(x, y)
    return model


def predict_stage0_model(model: Pipeline, x: pd.DataFrame) -> jnp.ndarray:
    """
    Predict using a stage 0 model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The model pipeline to use for prediction.
    x : pandas.DataFrame
        Input features.

    Returns
    -------
    jax.numpy.ndarray
        Predicted values.
    """
    logger.debug(f"Predicting with stage0 model {model}")
    y_hat = model.predict(x)

    # Select prediction with lowest performance value
    return jnp.argmin(y_hat, axis=1)
