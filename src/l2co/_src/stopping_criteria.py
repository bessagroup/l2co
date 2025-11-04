import jax
import jax.numpy as jnp
import optax

from ._typing import OptHistoryType, OptStates


def gradient_based_stopping_criteria(
    history: OptHistoryType,
    opt_states: OptStates,
    index: int,
    grad_tol: float = 1e-3,
) -> bool:
    grad_norms = jax.vmap(optax.global_norm)(history.grads)

    return jnp.nanmean(grad_norms) < grad_tol


def loss_based_stopping_criteria(
    history: OptHistoryType,
    opt_states: OptStates,
    index: int,
    loss_tol: float = 1e-6,
) -> bool:
    loss_improvement = jnp.abs(history.loss[-1] - history.loss[0])
    return loss_improvement < loss_tol


def combined_stopping_criteria(
    history: OptHistoryType,
    opt_states: OptStates,
    index: int,
    grad_tol: float = 1e-3,
    loss_tol: float = 1e-6,
) -> bool:
    grad_stop = gradient_based_stopping_criteria(
        history, opt_states, index, grad_tol=grad_tol
    )
    loss_stop = loss_based_stopping_criteria(
        history, opt_states, index, loss_tol=loss_tol
    )
    return jnp.any(jnp.logical_or(grad_stop, loss_stop))


def covariance_matrix_stopping_criteria(
    history: OptHistoryType,
    opt_states: OptStates,
    index: int,
    tol: float = 1e-12,
) -> bool:
    cov_matrix = opt_states[index].C
    cov_matrix = jnp.diag(cov_matrix) if cov_matrix.ndim == 1 else cov_matrix

    has_nan = jnp.any(jnp.isnan(cov_matrix))
    max_eigenvalue = jnp.nanmax(jnp.linalg.eigvalsh(cov_matrix))
    return jnp.any(jnp.logical_or(has_nan, max_eigenvalue < tol))
