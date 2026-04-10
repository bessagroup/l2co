"""
Neural network model architectures (RNN, MLP, CNN) for optimization tasks.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd

# =============================================================================

TagDict = dict[str, Any]

# =============================================================================


class RNN(eqx.Module):
    """Recurrent neural network with GRU cell for sequence tasks.

    Parameters
    ----------
    in_size : int
        Input feature size per time step.
    out_size : int
        Output size.
    hidden_size : int
        Hidden state size for the GRU cell.
    key : jax.random.PRNGKey
        Random key for parameter initialization.
    """

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
            hidden_size, out_size, use_bias=False, key=lkey
        )
        self.bias = jnp.zeros(out_size)
        self.in_size = in_size

    def __call__(self, x):
        """Forward pass through the RNN.

        Parameters
        ----------
        x : jax.Array
            Input tensor, reshaped to ``(-1, in_size)``.

        Returns
        -------
        jax.Array
            Sigmoid-activated output for binary classification.
        """
        x = x.reshape(-1, self.in_size)
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            """Scan body: advance GRU one step."""
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, x)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)


def rnn(
    in_size: int, out_size: int, hidden_size: int, *, key
) -> tuple[eqx.Module, dict[str, Any]]:
    """
    Create a recurrent neural network (RNN) model with GRU cells.

    Parameters
    ----------
    in_size : int
        Input size for the RNN.
    out_size : int
        Output size for the RNN.
    hidden_size : int
        Hidden state size for the GRU cell.
    key : jax.random.PRNGKey
        Random key for initialization.

    Returns
    -------
    tuple[eqx.Module, dict[str, Any]]
        The RNN model and its metadata dictionary.

    Notes
    -----
    Uses GRU cells for recurrent processing and applies sigmoid
    activation to output for binary classification.
    """
    return (
        RNN(
            in_size=in_size,
            out_size=out_size,
            hidden_size=hidden_size,
            key=key,
        ),
        {
            "model": "rnn",
            "in_size": in_size,
            "out_size": out_size,
            "hidden_size": hidden_size,
        },
    )


# =============================================================================


class MLP(eqx.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Parameters
    ----------
    in_size : int
        Input size of the MLP.
    out_size : int
        Output size of the MLP.
    hidden_size : int
        Size of the hidden layers.
    num_layers : int
        Number of layers in the MLP.
    hidden_activation : Callable, optional
        Activation function for hidden layers, by default jax.nn.tanh.
    output_activation : Callable, optional
        Activation function for the output layer, by default identity.

    Methods
    -------
    __call__(x)
        Forward pass of the MLP.
    """

    layers: list[eqx.nn.Linear]
    hidden_activation: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(
        static=True
    )
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(
        static=True
    )

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        hidden_activation: Callable = jax.nn.tanh,
        output_activation: Callable = lambda x: x,
        *,
        key,
    ):
        # Separate keys for each layer
        keys = jrd.split(key, num_layers + 1)
        self.layers = []
        self.layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[0]))

        for i in range(num_layers - 1):
            self.layers.append(
                eqx.nn.Linear(hidden_size, hidden_size, key=keys[i + 1])
            )

        self.layers.append(eqx.nn.Linear(hidden_size, out_size, key=keys[-1]))

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, x):
        """Forward pass through the MLP.

        Parameters
        ----------
        x : jax.Array
            Input tensor.

        Returns
        -------
        jax.Array
            Output after hidden and output activations.
        """
        # Apply activation for hidden layers
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        # Apply output activation
        return self.output_activation(self.layers[-1](x))


def mlp(
    in_size: int,
    out_size: int,
    hidden_size: int,
    num_layers: int,
    hidden_activation: Callable = jax.nn.tanh,
    output_activation: Callable = lambda x: x,
    *,
    key,
) -> tuple[eqx.Module, dict[str, Any]]:
    """
    Create an MLP model.

    Parameters
    ----------
    in_size : int
        Input size of the MLP.
    out_size : int
        Output size of the MLP.
    hidden_size : int
        Size of the hidden layers.
    num_layers : int
        Number of layers in the MLP.
    hidden_activation : Callable, optional
        Activation function for hidden layers, by default jax.nn.tanh.
    output_activation : Callable, optional
        Activation function for the output layer, by default identity.
    key : jax.random.PRNGKey
        Random key for initializing the model.

    Returns
    -------
    Tuple[eqx.Module, Dict[str, Any]]
        The MLP model and its metadata.
    """
    return (
        MLP(
            in_size,
            out_size,
            hidden_size,
            num_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            key=key,
        ),
        {
            "model": "mlp",
            "in_size": in_size,
            "out_size": out_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "hidden_activation": hidden_activation.__name__,
            "output_activation": output_activation.__name__
            if output_activation != (lambda x: x)
            else "identity",
        },
    )


# =============================================================================


class CNN(eqx.Module):
    """1-D convolutional neural network for sequence classification.

    Architecture: stacked Conv1d layers with max pooling and ReLU,
    followed by a fully-connected layer with log-softmax output.

    Parameters
    ----------
    in_size : int
        Total input feature count.
    out_size : int
        Number of output classes.
    hidden_channels : int
        Channels in each convolutional layer.
    num_layers : int
        Number of convolutional layers.
    kernel_size : int
        Convolution kernel size.
    width : int
        Sequence length of the input.
    in_channels : int
        Number of input channels.
    key : jax.random.PRNGKey
        Random key for parameter initialization.
    """

    conv_layers: list
    width: int
    in_channels: int

    def __init__(
        self,
        in_size,
        out_size,
        hidden_channels,
        num_layers,
        kernel_size,
        width,
        in_channels,
        *,
        key,
    ):
        keys = jrd.split(key, num_layers + 1)

        self.width = width
        self.in_channels = in_channels

        self.conv_layers = []

        # Define convolutional layers
        for i in range(num_layers):
            self.conv_layers.extend(
                [
                    eqx.nn.Conv1d(
                        in_channels,
                        hidden_channels,
                        kernel_size,
                        padding="SAME",
                        key=keys[i],
                    ),
                    eqx.nn.MaxPool1d(kernel_size=2),
                    # Apply ReLU activation
                    jax.nn.relu,
                ]
            )
            # Set in_channels for the next layer
            in_channels = hidden_channels

        # Final layer: Flatten the result and apply a fully connected layer
        self.conv_layers.extend(
            [
                # Flatten the output
                jnp.ravel,
                # Linear layer
                eqx.nn.Linear(
                    hidden_channels * (width - num_layers),
                    out_size,
                    key=keys[-1],
                ),
                # Log softmax for output probabilities
                jax.nn.log_softmax,
            ]
        )

    def __call__(self, x):
        """Forward pass through the CNN.

        Parameters
        ----------
        x : jax.Array
            Flat input with shape ``(features,)``, reshaped
            internally to ``(in_channels, width)``.

        Returns
        -------
        jax.Array
            Log-softmax class probabilities.
        """
        # Reshape (features,) to (sequence_length, channels)
        x = x.reshape(self.in_channels, self.width)

        # Pass through all convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        return x


def cnn(
    in_size: int,
    out_size: int,
    hidden_channels: int,
    num_layers: int,
    kernel_size: int,
    height: int,
    width: int,
    in_channels: int,
    *,
    key,
) -> tuple[eqx.Module, dict[str, Any]]:
    """
    Create a convolutional neural network (CNN) model.

    Parameters
    ----------
    in_size : int
        Input size (total number of features).
    out_size : int
        Output size (number of classes).
    hidden_channels : int
        Number of channels in convolutional layers.
    num_layers : int
        Number of convolutional layers.
    kernel_size : int
        Size of the convolutional kernel.
    height : int
        Height of the input (unused in current implementation).
    width : int
        Width/sequence length of the input.
    in_channels : int
        Number of input channels.
    key : jax.random.PRNGKey
        Random key for initialization.

    Returns
    -------
    tuple[eqx.Module, dict[str, Any]]
        The CNN model and its metadata dictionary.

    Notes
    -----
    Architecture includes Conv1d layers with max pooling and ReLU
    activation, followed by flattening and a fully connected layer
    with log-softmax output.
    """
    return (
        CNN(
            in_size=in_size,
            out_size=out_size,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            width=width,
            in_channels=in_channels,
            key=key,
        ),
        {
            "model": "cnn",
            "in_size": in_size,
            "out_size": out_size,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "kernel_size": kernel_size,
            "width": width,
            "in_channels": in_channels,
        },
    )
