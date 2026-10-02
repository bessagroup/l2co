"""Reusable neural-network building blocks (``MLP``, ``LSTM_Encoder``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from l2co._src.models.base import AbstractModel

# =============================================================================


class LSTM_Encoder(AbstractModel):
    """
    LSTM Encoder module.

    Parameters
    ----------
    in_size : int
        Input size of the LSTM encoder.
    hidden_size : int
        Hidden size of the LSTM encoder.
    key : jax.Array
        Random key for initializing the LSTM cell.

    Attributes
    ----------
    hidden_size : int
        Hidden size of the LSTM encoder.
    cell : eqx.nn.LSTMCell
        LSTM cell used in the encoder.
    linear : eqx.nn.Linear
        Linear layer applied to the final hidden state.
    """

    hidden_size: int
    in_size: int
    out_size: int
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, *, key):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.in_size = in_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=key)

    @classmethod
    def init(self, in_size: int, hidden_size: int, out_size: int, *, key):
        return LSTM_Encoder(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            key=key,
        )

    def __call__(self, x):
        """
        Encode sequence using LSTM.

        Parameters
        ----------
        x : array-like
            Input sequence to encode.
        init_state : array-like
            Initial hidden state for LSTM.

        Returns
        -------
        array-like
            Final hidden state after encoding sequence.
        """

        def scan_fn(state, input):
            return (self.cell(input, state), None)

        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

        (h, c), _ = jax.lax.scan(scan_fn, init_state, x)
        h = self.linear(h)
        return h

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize LSTM encoder configuration.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary.
        """
        return {
            "in_size": self.in_size,
            "hidden_size": self.hidden_size,
            "out_size": self.out_size,
        }

    @classmethod
    def reconstruct(
        cls,
        in_size: int,
        hidden_size: int,
        out_size: int,
    ) -> LSTM_Encoder:
        return cls(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            key=jr.key(0),
        )


# =============================================================================


class MLP(AbstractModel):
    """Multi-layer perceptron wrapper around ``eqx.nn.MLP``.

    Provides ``init``, ``to_dict``, and ``reconstruct`` for
    convenient construction and serialization.

    Attributes
    ----------
    model : eqx.nn.MLP
        Underlying Equinox MLP module.
    """

    model: eqx.nn.MLP

    def __init__(
        self,
        model: eqx.Module,
    ):
        self.model = model

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        *,
        key,
    ):
        """Create an MLP with the given architecture.

        Parameters
        ----------
        in_size : int
            Input feature dimension.
        out_size : int
            Output dimension (number of classes/logits).
        width_size : int
            Width of each hidden layer.
        depth : int
            Number of hidden layers.
        activation : Callable, optional
            Activation function. Default is ``jax.nn.relu``.
        key : jax.Array
            PRNG key for weight initialization.

        Returns
        -------
        MLP
            Initialized MLP instance.
        """
        model = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )
        return cls(model=model)

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        return self.model(x)

    def to_dict(self) -> dict[str, Any]:
        """Serialize MLP configuration to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``in_size``, ``out_size``,
            ``width_size``, ``depth``, and ``activation``
            (as a string name).
        """
        match self.model.activation:
            case jax.nn.relu:
                activation_str = "relu"
            case jax.nn.tanh:
                activation_str = "tanh"
        return {
            "in_size": self.model.in_size,
            "out_size": self.model.out_size,
            "width_size": self.model.width_size,
            "depth": self.model.depth,
            "activation": activation_str,
        }

    @classmethod
    def reconstruct(
        cls,
        activation: Literal["relu", "tanh"],
        **kwargs,
    ) -> MLP:
        """Reconstruct an MLP from serialized config.

        Weights are initialized with a dummy PRNG key and
        should be overwritten via
        ``eqx.tree_deserialise_leaves`` when loading.

        Parameters
        ----------
        activation : {"relu", "tanh"}
            Activation function name.
        **kwargs
            Remaining MLP config (``in_size``, ``out_size``,
            ``width_size``, ``depth``).

        Returns
        -------
        MLP
            Reconstructed MLP instance.
        """
        match activation:
            case "relu":
                activation_fn = jax.nn.relu
            case "tanh":
                activation_fn = jax.nn.tanh
        model = eqx.nn.MLP(
            **kwargs,
            activation=activation_fn,
            key=jr.key(0),
        )

        return cls(
            model=model,
        )
