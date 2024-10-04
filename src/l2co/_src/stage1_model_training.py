# Standard
from typing import Any, Dict

# Third-party
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state


class TrainState(train_state.TrainState):
    """
    Custom TrainState class
    """
    batch_stats: Any
    key: jax.Array


def create_initial_params(rng,
                          model: nn.Module,
                          **input_shapes) -> Dict[str, Any]:
    """
    Create initial parameters for the model

    Parameters
    ----------
    rng : jax.random.KeyArray
        Random key
    model : nn.Module
        The model
    input_shapes : Dict[str, Tuple[int]]
        The shapes of the input tensors
    """

    # Create dummy input tensors for each shape provided in input_shapes
    dummy_inputs = [jnp.ones(shape, jnp.float32)
                    for shape in input_shapes.values()]

    # Initialize model parameters using the dummy inputs
    variables = model.init({'params': rng}, *dummy_inputs, train=False)

    return variables


@jax.jit
def train_step(state: train_state.TrainState, X_trajectory, X_features, Y):
    def mse(params):
        def loss_fn(x_trajectory, x_features, y):
            y_pred = state.apply_fn(
                {'params': params},
                x_trajectory, x_features, train=True,
            )
            return (y_pred - y) ** 2

        # Vectorize loss_fn over the batch dimension
        batched_loss_fn = jax.vmap(loss_fn, in_axes=(0, 0, 0))
        loss = batched_loss_fn(X_trajectory, X_features, Y)
        return jnp.mean(loss)  # Mean across the batch

    # Compute the loss and gradients
    loss, grads = jax.value_and_grad(mse)(state.params)

    # Update the state with gradients
    state = state.apply_gradients(grads=grads)

    # Collect and return metrics
    metrics = {'loss': loss}
    return state, metrics

# =============================================================================


@jax.jit
def eval_step(state: train_state.TrainState, X_trajectory, X_features, Y):
    def mse(params):
        def loss_fn(x_trajectory, x_features, y):
            y_pred = state.apply_fn(
                {'params': params},
                x_trajectory, x_features, train=False,
            )
            return (y_pred - y) ** 2

        # Vectorize loss_fn over the batch dimension
        batched_loss_fn = jax.vmap(loss_fn, in_axes=(0, 0, 0))
        loss = batched_loss_fn(X_trajectory, X_features, Y)
        return jnp.mean(loss)  # Mean across the batch

    loss = mse(state.params)
    metrics = {'loss': loss}
    return metrics

# =============================================================================


def batch_iterator(data: tuple, batch_size: int, rng=None):
    # Assume all data arrays have the same first dimension (batch size)
    data_size = data[0].shape[0]

    # Generate indices
    indices = jnp.arange(data_size)

    # Shuffle indices if requested
    if rng is not None:
        indices = jax.random.permutation(rng, indices)

    # Yield batches
    for start_idx in range(0, data_size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(d[batch_indices] for d in data)
