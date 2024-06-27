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

        for _ in range(self.n_layers):
            x = nn.Conv(features=self.n_features, kernel_size=(3, 3),
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


# =============================================================================


class CNN_MNIST1D(nn.Module):
    n_labels: int
    channels: int = 25
    linear_in: int = 125

    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, 1, x.shape[-1]))
        x = nn.relu(nn.Conv(features=self.channels, kernel_size=(5,),
                            strides=(2,), padding='same',
                            kernel_init=normal_initializer,
                            bias_init=normal_initializer)(x))
        x = nn.relu(nn.Conv(features=self.channels, kernel_size=(3,),
                            strides=(2,), padding='same',
                            kernel_init=normal_initializer,
                            bias_init=normal_initializer)(x))
        x = nn.relu(nn.Conv(features=self.channels, kernel_size=(3,),
                            strides=(2,), padding='same',
                            kernel_init=normal_initializer,
                            bias_init=normal_initializer)(x))

        x = x.reshape((x.shape[0], -1))  # flatten the conv features
        x = nn.Dense(
            features=self.n_labels, kernel_init=normal_initializer,
            bias_init=normal_initializer)(x)
        return nn.log_softmax(x)


# =============================================================================


class ResidualBlock(nn.Module):
    n_features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.n_features, kernel_size=(3, 3),
                    kernel_init=normal_initializer,
                    bias_init=normal_initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.n_features, kernel_size=(3, 3),
                    kernel_init=normal_initializer,
                    bias_init=normal_initializer)(x)
        return nn.relu(x + residual)


class ResNet(nn.Module):
    n_features: int
    n_layers: int
    n_labels: int

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 4  # (batch_size, height, width, channels)

        x = nn.Conv(features=self.n_features, kernel_size=(7, 7),
                    strides=(2, 2),
                    kernel_init=normal_initializer,
                    bias_init=normal_initializer)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        for _ in range(self.n_layers):
            x = ResidualBlock(n_features=self.n_features)(x)

        # Assuming input size to be at least 32x32
        x = nn.avg_pool(x, window_shape=(4, 4), strides=(4, 4))

        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=256,
                     kernel_init=normal_initializer,
                     bias_init=normal_initializer)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_labels,
                     kernel_init=normal_initializer,
                     bias_init=normal_initializer)(x)
        return nn.log_softmax(x)

# Example usage
# model = ResNet(n_features=64, n_blocks=3, n_labels=10)
