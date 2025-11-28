"""
Module for stopping criteria
"""
#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

# Third-party
from jaxtyping import PyTree

# Local
from .typing import OptHistoryType

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def gradient_based_stopping_criteria(
    history: OptHistoryType,
    opt_state: PyTree,
    grad_tol: float = 1e-3,
) -> bool:
    """Check if optimization should stop based on gradient norms.

    Parameters
    ----------
    history : OptHistoryType
        Optimization history containing gradients.
    opt_state : PyTree
        Current optimizer state (unused).
    grad_tol : float, optional
        Gradient tolerance threshold, by default 1e-3.

    Returns
    -------
    bool
        True if mean gradient norm is below tolerance, False otherwise.
    """
    grad_norms = jax.vmap(optax.global_norm)(history.grads)

    return jnp.nanmean(grad_norms) < grad_tol


def loss_based_stopping_criteria(
    history: OptHistoryType,
    opt_state: PyTree,
    loss_tol: float = 1e-6,
) -> bool:
    """Check if optimization should stop based on loss improvement.

    Parameters
    ----------
    history : OptHistoryType
        Optimization history containing loss values.
    opt_state : PyTree
        Current optimizer state (unused).
    loss_tol : float, optional
        Loss improvement tolerance threshold, by default 1e-6.

    Returns
    -------
    bool
        True if loss improvement is below tolerance, False otherwise.
    """
    loss_improvement = jnp.abs(history.loss[-1] - history.loss[0])
    return loss_improvement < loss_tol


def combined_stopping_criteria(
    history: OptHistoryType,
    opt_state: PyTree,
    grad_tol: float = 1e-3,
    loss_tol: float = 1e-6,
) -> bool:
    """Check if optimization should stop based on gradient or loss criteria.

    Parameters
    ----------
    history : OptHistoryType
        Optimization history containing gradients and loss values.
    opt_state : PyTree
        Current optimizer state.
    grad_tol : float, optional
        Gradient tolerance threshold, by default 1e-3.
    loss_tol : float, optional
        Loss improvement tolerance threshold, by default 1e-6.

    Returns
    -------
    bool
        True if either gradient or loss stopping criteria is met,
        False otherwise.
    """
    grad_stop = gradient_based_stopping_criteria(
        history, opt_state, grad_tol=grad_tol
    )
    loss_stop = loss_based_stopping_criteria(
        history, opt_state, loss_tol=loss_tol
    )
    return jnp.any(jnp.logical_or(grad_stop, loss_stop))


def covariance_matrix_stopping_criteria(
    history: OptHistoryType,
    opt_state: PyTree,
    tol: float = 1e-12,
) -> bool:
    """Check if optimization should stop based on covariance matrix condition.

    Stops if the covariance matrix contains NaN values or if the maximum
    eigenvalue is below the tolerance threshold.

    Parameters
    ----------
    history : OptHistoryType
        Optimization history (unused).
    opt_state : PyTree
        Current optimizer state containing covariance matrix C.
    tol : float, optional
        Tolerance for maximum eigenvalue, by default 1e-12.

    Returns
    -------
    bool
        True if covariance matrix contains NaN or max eigenvalue is below
        tolerance, False otherwise.
    """
    cov_matrix = opt_state.C
    cov_matrix = jnp.diag(cov_matrix) if cov_matrix.ndim == 1 else cov_matrix

    has_nan = jnp.any(jnp.isnan(cov_matrix))
    max_eigenvalue = jnp.nanmax(jnp.linalg.eigvalsh(cov_matrix))
    return jnp.any(jnp.logical_or(has_nan, max_eigenvalue < tol))


# =============================================================================

STOPPING_CRITERIA: dict[str, Callable[[OptHistoryType, PyTree, int], bool]] = {
    "covariance_matrix": covariance_matrix_stopping_criteria,
    "combined": combined_stopping_criteria,
    "gradient_based": gradient_based_stopping_criteria,
    "loss_based": loss_based_stopping_criteria,
}
