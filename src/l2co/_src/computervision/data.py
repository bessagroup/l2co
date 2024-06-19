"""
Data for classification tasks
"""

#
#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from typing import Generator, Tuple

# Third-party
import numpy as np
from jax.random import KeyArray, permutation
from mnist1d.data import get_dataset_args, make_dataset
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Dataset:
    def __init__(self,
                 x_train: np.ndarray, x_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray):
        """
        Initialize the dataset

        Parameters
        ----------
        x_train : np.ndarray
            Training data
        x_test : np.ndarray
            Testing data
        y_train : np.ndarray
            Training labels
        y_test : np.ndarray
            Testing labels

        Note
        ----
        The data should be in the shape (n_samples, height, width, n_channels)

        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    #                                                      Dataset constructors
    # =========================================================================

    @classmethod
    def from_mnist(cls):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return cls(
            np.expand_dims(x_train, axis=-1),
            np.expand_dims(x_test, axis=-1), y_train, y_test)

    @classmethod
    def from_fashion_mnist(cls):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return cls(
            np.expand_dims(x_train, axis=-1),
            np.expand_dims(x_test, axis=-1), y_train, y_test)

    @classmethod
    def from_cifar10(cls):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return cls(x_train, x_test, y_train, y_test)

    @classmethod
    def from_mnist1d(cls):
        defaults = get_dataset_args()
        data = make_dataset(defaults)
        return cls(x_train=data['x'].reshape(-1, 40, 1, 1),
                   x_test=data['x_test'].reshape(-1, 40, 1, 1),
                   y_train=data['y'],
                   y_test=data['y_test'])

    #                                                   Shuffeling and batching
    # =========================================================================

    def get_batch(self, batch_size) -> Generator[
            Tuple[np.ndarray, np.ndarray], None, None]:
        for i in range(0, len(self.x_train), batch_size):
            yield (self.x_train[i:i + batch_size],
                   self.y_train[i:i + batch_size])

    def permute(self, rng: KeyArray) -> Dataset:
        _permutation = permutation(rng, len(self.x_train))
        return Dataset(x_train=self.x_train[_permutation],
                       y_train=self.y_train[_permutation],
                       x_test=self.x_test, y_test=self.y_test)

    def kfolds(
            self, n_folds: int, seed: int) -> Generator[Dataset, None, None]:
        _kfold = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

        for train_indices, test_indices in _kfold.split(
                self.x_train, self.y_train):
            yield Dataset(
                x_train=self.x_train[train_indices],
                y_train=self.y_train[train_indices],
                x_test=self.x_train[test_indices],
                y_test=self.y_train[test_indices])
