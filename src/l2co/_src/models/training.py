"""Training helpers (``mse_loss``, ``train_equinox``)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax_tqdm import scan_tqdm
from tqdm import tqdm

from .base import AbstractModel

# =============================================================================


def mse_loss(model: AbstractModel, x, y) -> jax.Array:
    """Mean squared error between model predictions and targets."""
    preds = eqx.filter_vmap(model)(x)
    return jnp.mean((preds - y) ** 2)


def train_equinox(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    train_data: tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array],
    val_data: tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]
    | None = None,
    num_steps: int = 100,
    verbose: bool = False,
    scan: bool = True,
) -> tuple[eqx.Module, dict[str, jax.Array]]:
    """Train an Equinox model with MSE loss via optax.

    Partitions the model into trainable parameters and static
    structure, then runs a training loop using either
    ``jax.lax.scan`` (JIT-compiled) or a Python for-loop.

    Parameters
    ----------
    model : eqx.Module
        Equinox model to train.
    optimizer : optax.GradientTransformation
        Optax optimizer instance.
    train_data : tuple
        Tuple of ``(x_train, y_train)`` arrays.
    val_data : tuple or None, optional
        Tuple of ``(x_val, y_val)`` arrays. If None,
        validation loss is reported as NaN. Default is None.
    num_steps : int, optional
        Number of gradient steps. Default is 100.
    verbose : bool, optional
        Show progress bar. Default is False.
    scan : bool, optional
        Use ``jax.lax.scan`` (True) or a Python for-loop
        (False). Default is True.

    Returns
    -------
    tuple[eqx.Module, dict[str, jax.Array]]
        Trained model and metrics dict with
        ``train_loss`` and ``val_loss`` arrays of shape
        ``(num_steps,)``.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    loss_fn = mse_loss
    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def make_step(carry, _):
        params, opt_state = carry
        model = eqx.combine(params, static)

        # Training Branch
        train_loss, grads = loss_and_grad_fn(model, *train_data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)

        # Validation Branch (Conditional)
        val_loss = (
            loss_fn(model, *val_data) if val_data is not None else jnp.nan
        )

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        return (new_params, opt_state), metrics

    # 2. Execution Logic (Scan vs Loop)
    if scan:
        step_fn = scan_tqdm(num_steps)(make_step) if verbose else make_step
        (params, opt_state), metrics = jax.lax.scan(
            step_fn,
            (params, opt_state),
            xs=jnp.arange(num_steps),
            length=num_steps,
        )
    else:
        metrics_list = []
        for _ in tqdm(range(num_steps), disable=not verbose):
            (params, opt_state), step_metrics = make_step(
                (params, opt_state), None
            )
            metrics_list.append(step_metrics)
        metrics = {
            k: jnp.stack([m[k] for m in metrics_list]) for k in metrics_list[0]
        }

    return eqx.combine(params, static), metrics
