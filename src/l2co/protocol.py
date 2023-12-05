"""
Module containing protocol classes for the data-driven process of the L2CO
experiment.
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import Protocol

# Third-party
import xarray as xr

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
# =============================================================================


class CollaborativeFilteringAlgorithm(Protocol):
    """
    Protocol class for collaborative filtering algorithms.
    """

    def predict(self) -> xr.DataArray:
        """
        Predict the missing values in the performance matrix.

        Returns
        -------
        xr.DataArray
            Data array with fully definited performance matrix.
        """
        ...


class FeaturesSimilarity(Protocol):
    """
    Protocol class for finding similarity between features of optimization
    problems.
    """

    def compare(self, instance: xr.DataArray) -> xr.DataArray:
        """
        Compare the features of an optimization problem with the features of
        the optimization problems in the training set.

        Parameters
        ----------
        instance : xr.DataArray
            Data array with features of an optimization problem.

        Returns
        -------
        xr.DataArray
            Data array with similarity between features of optimization
        """
        ...


class HistorySimilarity(Protocol):
    """
    Protocol class for finding similarity between history of optimization
    problems.
    """

    def compare(self, instance: xr.DataArray) -> xr.DataArray:
        """
        Compare the history of an optimization problem with the history of the
        optimization problems in the training set.

        Parameters
        ----------
        instance : xr.DataArray
            Data array with history of an optimization problem.

        Returns
        -------
        xr.DataArray
            Data array with similarity between history of optimization
        """
        ...
