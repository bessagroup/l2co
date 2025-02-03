
from typing import Any, Dict, Tuple

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

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrd.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(
            hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)


def rnn(in_size: int, out_size: int, hidden_size: int, *, key
        ) -> Tuple[eqx.Module, Dict[str, Any]]:
    return (RNN(in_size, out_size, hidden_size, key=key),
            {'model': 'rnn',
             'in_size': in_size,
             'out_size': out_size,
             'hidden_size': hidden_size})
