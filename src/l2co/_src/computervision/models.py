"""
Neural network models for classification
"""

#
#                                                                       Modules
# =============================================================================

# Third-party
import flax.linen as nn
import jax

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def normal_initializer(rng, shape, dtype):
    return jax.random.normal(rng, shape, dtype) * 0.01

# =============================================================================


class SimpleMLP(nn.Module):
    n_features: int
    n_layers: int
    n_labels: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input

        for _ in range(self.n_layers):
            x = nn.Dense(features=self.n_features,
                         kernel_init=normal_initializer,
                         bias_init=normal_initializer)(x)
            x = nn.relu(x)

        x = nn.Dense(features=self.n_labels, kernel_init=normal_initializer,
                     bias_init=normal_initializer)(x)
        return nn.log_softmax(x)


class CNN(nn.Module):
    """A simple CNN model."""
    n_features: int
    n_layers: int
    n_labels: int

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 4  # (batch_size, height, width, channels)

        x = nn.Conv(features=28, kernel_size=(3, 3),
                    kernel_init=normal_initializer,
                    bias_init=normal_initializer)(x)  # 32
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=48, kernel_size=(3, 3),
                    kernel_init=normal_initializer,
                    bias_init=normal_initializer)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=256,
                     kernel_init=normal_initializer,
                     bias_init=normal_initializer)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_labels,
                     kernel_init=normal_initializer,
                     bias_init=normal_initializer)(x)
        return nn.log_softmax(x)
