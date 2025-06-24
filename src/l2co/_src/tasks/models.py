
from typing import Any, Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd

# =============================================================================

TagDict = Dict[str, Any]

# =============================================================================


class RNN(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array
    in_size: int = eqx.field(static=True)

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrd.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(
            hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)
        self.in_size = in_size

    def __call__(self, x):
        x = x.reshape(-1, self.in_size)
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, x)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)


def rnn(in_size: int, out_size: int, hidden_size: int, *, key
        ) -> Tuple[eqx.Module, Dict[str, Any]]:
    return (RNN(in_size=in_size,
                out_size=out_size, hidden_size=hidden_size, key=key),
            {'model': 'rnn',
             'in_size': in_size,
             'out_size': out_size,
             'hidden_size': hidden_size})

# =============================================================================


class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    hidden_activation: Callable[[jnp.ndarray],
                                jnp.ndarray] = eqx.field(static=True)
    output_activation: Callable[[jnp.ndarray],
                                jnp.ndarray] = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, hidden_size: int,
                 num_layers: int,
                 hidden_activation: Callable = jax.nn.tanh,
                 output_activation: Callable = lambda x: x,
                 *, key):
        keys = jrd.split(key, num_layers + 1)  # Separate keys for each layer
        self.layers = []
        self.layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[0]))

        for i in range(num_layers - 1):
            self.layers.append(eqx.nn.Linear(
                hidden_size, hidden_size, key=keys[i + 1]))

        self.layers.append(eqx.nn.Linear(hidden_size, out_size, key=keys[-1]))

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, x):
        for layer in self.layers[:-1]:  # Apply activation for hidden layers
            x = self.hidden_activation(layer(x))
        # Apply output activation
        return self.output_activation(self.layers[-1](x))


def mlp(in_size: int, out_size: int, hidden_size: int, num_layers: int,
        hidden_activation: Callable = jax.nn.tanh,
        output_activation: Callable = lambda x: x,  # Identity function
        *, key) -> Tuple[eqx.Module, Dict[str, Any]]:
    return (MLP(in_size, out_size, hidden_size, num_layers,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                key=key),
            {'model': 'mlp',
             'in_size': in_size,
             'out_size': out_size,
             'hidden_size': hidden_size,
             'num_layers': num_layers,
             'hidden_activation': hidden_activation.__name__,
             'output_activation': output_activation.__name__ if
             output_activation != (lambda x: x) else 'identity'})


# =============================================================================

class CNN(eqx.Module):
    conv_layers: list
    width: int
    in_channels: int

    def __init__(self, in_size, out_size, hidden_channels, num_layers,
                 kernel_size, width, in_channels, *, key):
        keys = jrd.split(key, num_layers + 1)

        self.width = width
        self.in_channels = in_channels

        self.conv_layers = []

        # Define convolutional layers
        for i in range(num_layers):
            self.conv_layers.extend([
                eqx.nn.Conv1d(in_channels, hidden_channels,
                              kernel_size, padding="SAME", key=keys[i]),
                eqx.nn.MaxPool1d(kernel_size=2),
                jax.nn.relu,  # Apply ReLU activation
            ])
            in_channels = hidden_channels  # Set in_channels for the next layer

        # Final layer: Flatten the result and apply a fully connected layer
        self.conv_layers.extend([
            jnp.ravel,  # Flatten the output
            eqx.nn.Linear(hidden_channels*(width-num_layers), out_size,
                          key=keys[-1]),  # Linear layer
            jax.nn.log_softmax,  # Log softmax for output probabilities
        ])

    def __call__(self, x):
        """
        x: (features,) -> Reshaped to (sequence_length, channels)
        """
        # Reshape (features,) to (sequence_length, channels)
        x = x.reshape(
            self.in_channels, self.width)

        # Pass through all convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        return x


def cnn(in_size: int, out_size: int, hidden_channels: int,
        num_layers: int, kernel_size: int,
        height: int, width: int,
        in_channels: int, *, key
        ) -> Tuple[eqx.Module, Dict[str, Any]]:
    return (CNN(in_size=in_size,
                out_size=out_size,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                kernel_size=kernel_size,
                width=width,
                in_channels=in_channels,
                key=key),
            {'model': 'cnn',
             'in_size': in_size,
             'out_size': out_size,
             'hidden_channels': hidden_channels,
             'num_layers': num_layers,
             'kernel_size': kernel_size,
             'width': width,
             'in_channels': in_channels})
