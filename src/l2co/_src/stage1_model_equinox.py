""""""

#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm import tqdm

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

# =============================================================================


def batch_iterator(data: tuple, batch_size: int, rng=None):
    data_size = data[0].shape[0]
    indices = jnp.arange(data_size)
    if rng is not None:
        indices = jax.random.permutation(rng, indices)
    for start_idx in range(0, data_size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(d[batch_indices] for d in data)


# =============================================================================


# =============================================================================


class LSTMAndMLP(eqx.Module):
    encoder: eqx.Module
    linear_layer: eqx.Module
    mlp: eqx.Module

    def __init__(self, mlp_layers, mlp_neurons_per_layer, lstm_hidden_size,
                 lstm_latent_dim, trajectory_size, features_size, n_labels,
                 key):
        # Split three random keys
        key1, key2, key3 = jax.random.split(key, num=3)

        # Create the LSTM encoder
        self.encoder = LSTMEncoder(latent_dim=lstm_latent_dim,
                                   hidden_size=lstm_hidden_size,
                                   input_size=trajectory_size,
                                   key=key1)

        # Create a linear layer to combine the LSTM output with the features
        self.linear_layer = eqx.nn.Linear(
            in_features=features_size+lstm_latent_dim,
            out_features=mlp_neurons_per_layer, key=key2)

        # Create an MLP to predict the optimizer
        self.mlp = SimpleMLP(n_features=mlp_neurons_per_layer,
                             n_layers=mlp_layers,
                             n_labels=n_labels,
                             key=key3)

    def __call__(self, x_trajectory, x_features, train: bool = True):
        x_latent_trajectory = self.encoder(x_trajectory, train)

        x_combined = jnp.concatenate(
            [x_latent_trajectory, x_features], axis=-1)

        x = self.linear_layer(x_combined)

        y = self.mlp(x, train)
        return y


# =============================================================================


class LSTMEncoder(eqx.Module):
    lstm_cell: eqx.Module
    latent_dense: eqx.Module

    def __init__(self, latent_dim, hidden_size, input_size, key):
        key1, key2 = jax.random.split(key)
        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, key=key1)
        self.latent_dense = eqx.nn.Linear(hidden_size, latent_dim, key=key2)

    def __call__(self, x, train: bool = True):
        def scan_fn(state, input):
            return (self.lstm_cell(input, state), None)

        init_state = (jnp.zeros(self.lstm_cell.hidden_size),
                      jnp.zeros(self.lstm_cell.hidden_size))

        (h, c), _ = jax.lax.scan(scan_fn, init_state, x)

        h = self.latent_dense(h)

        return h


# =============================================================================


class SimpleMLP(eqx.Module):
    layers: List[eqx.Module]

    def __init__(self, n_features, n_layers, n_labels, key):
        keys = jax.random.split(key, n_layers+1)

        self.layers = [eqx.nn.Linear(in_features=n_features, out_features=n_features,
                                     key=key)
                       for key in keys[:-1]]
        self.layers.append(eqx.nn.Linear(in_features=n_features, out_features=n_labels,
                                         key=keys[-1]))

    def __call__(self, x, train: bool = True):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.sigmoid(self.layers[-1](x))

# =============================================================================


@eqx.filter_value_and_grad
def compute_mse_loss(model, X_trajectory, X_features, Y):
    y_pred = jax.vmap(model)(X_trajectory, X_features)
    return jnp.mean((y_pred - Y) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, X_trajectory, X_features, Y, optimizer):
    loss, grads = compute_mse_loss(model, X_trajectory, X_features, Y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def eval_step(model, X_trajectory, X_features, Y):
    y_pred = jax.vmap(model)(X_trajectory, X_features)
    return jnp.mean((y_pred - Y) ** 2)

# =============================================================================


def train_loop(model, X_trajectory, X_features, Y, optimizer, n_steps: int,
               verbose: bool = True):
    # Create optimizer and initialize state
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    if verbose:
        loop_iterator = tqdm(range(n_steps), desc="Loss: N/A", leave=True)
    else:
        loop_iterator = range(n_steps)

    try:
        for step in loop_iterator:
            loss, model, opt_state = train_step(
                model, opt_state, X_trajectory, X_features, Y, optimizer
            )

            if verbose:
                # Update tqdm description with the current loss
                loop_iterator.set_description(f"Loss: {loss:.4f}")
    finally:
        if verbose:
            loop_iterator.close()

    return model
