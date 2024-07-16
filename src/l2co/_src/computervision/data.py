"""
Data for classification tasks
"""

#
#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from typing import Generator, Iterable, Optional, Tuple

# Third-party
import numpy as np
from jax.random import KeyArray, PRNGKey, permutation
from mnist1d.data import get_dataset_args, make_dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import \
    train_test_split as sklearn_train_test_split
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist

from ..data_loader import DataLoader

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

    def __add__(self, other: Dataset) -> Dataset:
        return Dataset(
            np.concatenate([self.x_train, other.x_train]),
            np.concatenate([self.x_test, other.x_test]),
            np.concatenate([self.y_train, other.y_train]),
            np.concatenate([self.y_test, other.y_test]))

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
    def from_mnist1d(cls, labels: Optional[Iterable[int]] = None):
        defaults = get_dataset_args()
        data = make_dataset(defaults)

        _dataset = cls(x_train=data['x'].reshape(-1, 40, 1, 1),
                       x_test=data['x_test'].reshape(-1, 40, 1, 1),
                       y_train=data['y'],
                       y_test=data['y_test'])

        if labels is None:
            return _dataset

        else:
            _datasets_with_label = [
                _dataset.with_label(label) for label in labels]
            return sum(_datasets_with_label[1:], _datasets_with_label[0])

    @classmethod
    def from_optimization_trajectory(
            cls, d: DataLoader, optimizer: str, t: int, t_start: int = 0,
            seed: Optional[int] = None, train_test_split: bool = True):
        X_traj = d.select_optimizer(optimizer).select_t(
            t=t, t_start=t_start).X_trajectory(t, t_start=t_start)
        y_traj = d.query_optimizer(optimizer).performance.isel(
            iterations=-1)

        X = np.expand_dims(X_traj.values.transpose(2, 1, 0), axis=-1)
        Y = y_traj.values.T

        # train test split
        if train_test_split:
            X_train, X_test, y_train, y_test = sklearn_train_test_split(
                X, Y, test_size=0.2, random_state=seed)
        else:
            X_test = X
            y_test = Y
            X_train = None
            y_train = None

        return cls(X_train, X_test, y_train, y_test)

    #                                                   Shuffeling and batching
    # =========================================================================

    def get_batch(
        self, batch_size, num_epochs: int = 1000,
            seed: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        if seed is None:
            rng = PRNGKey(np.random.randint(0, 2**32 - 1))
        else:
            rng = PRNGKey(seed)

        for epoch in range(num_epochs):
            self._permute(rng)

            for i_batch in range(0, len(self.x_train), batch_size):
                batch_x = self.x_train[i_batch:i_batch + batch_size]
                batch_y = self.y_train[i_batch:i_batch + batch_size]

                # Check if the batch has fewer than batch_size items
                if len(batch_x) < batch_size:
                    # Reshuffle data
                    self._permute(rng)

                    # Calculate how many items are missing
                    missing_count = batch_size - len(batch_x)

                    # Append missing items from the beginning of the
                    # reshuffled data
                    batch_x = np.concatenate(
                        (batch_x, self.x_train[:missing_count]))
                    batch_y = np.concatenate(
                        (batch_y, self.y_train[:missing_count]))

                yield (batch_x, batch_y)

    def _permute(self, rng: KeyArray):
        _permutation = permutation(rng, len(self.x_train))
        self.x_train = self.x_train[_permutation]
        self.y_train = self.y_train[_permutation]

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

    def with_label(self, label: int) -> Dataset:
        return Dataset(
            x_train=self.x_train[self.y_train == label],
            y_train=self.y_train[self.y_train == label],
            x_test=self.x_test[self.y_test == label],
            y_test=self.y_test[self.y_test == label])
