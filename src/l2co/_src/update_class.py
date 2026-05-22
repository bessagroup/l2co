"""
Module for the update class.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Local
from .batching import BatchState
from .history_state import HistoryState
from .opt_history import OptHistory, RecentHistory
from .tree_utils import vmapped_loss, vmapped_loss_with_rng
from .typing import (
    Carry,
    InitFunction,
    InputParameters,
    StepFunction,
    StopFunction,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# =============================================================================


def init_(
    update_class: UpdateClass,
    params: InputParameters,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    key: PRNGKeyArray,
) -> PyTree:
    """Initialize the optimizer state.

    Parameters
    ----------
    params : InputParameters
        Current input parameters.
    batch_state : BatchState
        Current batch state.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    PyTree
        Initialized optimizer state.
    """
    batch_idxs, _ = batch_state.next(key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
    return update_class.init_fn(params, key, **sample)


# =============================================================================


def step(
    carry: Carry,
    xs,
    update_class: UpdateClass,
    dataset: dict[str, jax.Array],
) -> tuple[Carry, OptHistory]:
    def do_step(
        carry: Carry,
    ) -> tuple[Carry, OptHistory]:
        """Execute a single optimization step.

        Parameters
        ----------
        carry : Carry
            Tuple containing (params, opt_state, key, batch_state, done,
            recent_history).

        Returns
        -------
        tuple
            Updated carry and optimization history.
        """
        (params, opt_state, key, batch_state, done, recent_history) = carry
        batch_idxs, batch_state = batch_state.next(key)
        sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
        new_carry, history = update_class.step_fn(
            (params, opt_state, key), sample=sample
        )

        new_recent_history = recent_history.update(history)

        return (
            *new_carry,
            batch_state,
            done,
            new_recent_history,
        ), history

    def skip_step(
        carry: Carry,
    ) -> tuple[Carry, OptHistory]:
        """Skip an optimization step when stopping criteria is met.

        Parameters
        ----------
        carry : Carry
            Tuple containing (params, opt_state, key, batch_state, done,
            recent_history).
        dataset : dict[str, jax.Array]
            Unused; accepted so the signature matches :meth:`do_step` for
            :func:`jax.lax.cond`.

        Returns
        -------
        tuple
            Unchanged carry and dummy history.
        """
        return carry, OptHistory.init(
            carry[0], update_class.popsize, update_class.hash
        )

    _, opt_state, _, _, done, recent_history = carry

    (new_carry, history_entry) = jax.lax.cond(
        done | update_class.stop_fn(recent_history, opt_state),
        skip_step,
        do_step,
        carry,
    )

    return new_carry, history_entry


batch_step_ = eqx.filter_vmap(
    step, in_axes=(eqx.if_array(axis=0), None, None, None)
)

# =============================================================================


def run_(
    update_class: UpdateClass,
    opt_state: PyTree,
    params: InputParameters,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    key: PRNGKeyArray,
    n_iterations: int,
    verbose: bool,
) -> tuple[
    InputParameters,
    InputParameters,
    Array[Float, ""],
    PyTree,
    BatchState,
    HistoryState,
]:
    """Perform optimization steps for specified iterations.

    Parameters
    ----------
    opt_state : PyTree
        Current optimizer state.
    params : InputParameters
        Current input parameters.
    batch_state : BatchState
        Current batch state.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    key : PRNGKeyArray
        PRNG key for random operations.
    n_iterations : int
        Number of iterations to perform.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[InputParameters, InputParameters, Array[Float, ""], PyTree,
    BatchState, HistoryState]
        Updated states after optimization steps.
    """
    # Initialize moving history buffer with NaNs
    recent_history_init = RecentHistory.init(
        params=params,
        memory_size=20,
    )

    # Initialize done flag
    init_done = jnp.array(False)

    # Run the optimization loop
    n = n_iterations + 1

    scan_step = Partial(
        step,
        update_class=update_class,
        dataset=dataset,
    )

    scan_fn = scan_tqdm(n)(scan_step) if verbose else scan_step

    (_, opt_state, _, batch_state, _, _), history = jax.lax.scan(
        f=scan_fn,
        init=(
            params,
            opt_state,
            key,
            batch_state,
            init_done,
            recent_history_init,
        ),
        xs=jnp.arange(n),
        length=n,
    )

    best_params, best_loss = history.flatten().retrieve_best()

    # Remove first element for dim=0 history
    ## TODO: If I don't do this, the history has the first element
    # Is this a problem? Because for switching it is useful to have it for
    # the first choice...
    history: OptHistory = jax.tree.map(lambda x: x[1:], history)

    params = jax.tree.map(lambda x: x[-1], history.params)

    # Create history state
    new_history_state = HistoryState.from_history(history)

    return (
        params,
        best_params,
        best_loss,
        opt_state,
        batch_state,
        new_history_state,
    )


def batch_run_(
    update_class: UpdateClass,
    opt_state: PyTree,
    params: InputParameters,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    key: PRNGKeyArray,
    n_iterations: int,
    verbose: bool,
) -> tuple[
    InputParameters,
    InputParameters,
    Array[Float, ""],
    PyTree,
    BatchState,
    HistoryState,
]:
    """Run ``n_realizations`` optimization trajectories in one fused scan.

    A single :func:`jax.lax.scan` walks the time axis, and each scan
    iteration invokes :data:`batch_step_` -- :func:`step` vmapped over a
    leading ``n_realizations`` axis. This produces one compiled scan
    body that does all realizations' time steps together, rather than
    ``n_realizations`` independent scans.

    Inputs ``opt_state`` and ``params`` are expected to carry a leading
    realization axis (size ``n_realizations``); the auxiliary scan
    carry leaves (``key``, ``batch_state``, ``init_done``,
    ``recent_history_init``) are broadcast to that same axis here so
    every leaf of the carry is mappable on axis 0.

    Parameters
    ----------
    update_class : UpdateClass
        Optimizer configuration (static across realizations).
    opt_state : PyTree
        Per-realization optimizer state with shape
        ``(n_realizations, ...)`` on every array leaf.
    params : InputParameters
        Per-realization input parameters with shape
        ``(n_realizations, popsize, ...)``.
    batch_state : BatchState
        Single (un-batched) batch state; broadcast across realizations
        internally.
    dataset : dict[str, jax.Array]
        Dataset for evaluation (shared across realizations).
    key : PRNGKeyArray
        Single PRNG key; broadcast across realizations to match the
        ``in_axes=None`` semantics of the run_state-level
        ``batch_run`` wrapper.
    n_iterations : int
        Number of iterations to perform.
    verbose : bool
        Whether to wrap the scan body with :func:`jax_tqdm.scan_tqdm`.

    Returns
    -------
    tuple[InputParameters, InputParameters, Array[Float, ""], PyTree,
    BatchState, HistoryState]
        ``(params, best_params, best_loss, opt_state, batch_state,
        history_state)`` -- each output carries a leading
        ``n_realizations`` axis.
    """
    n_realizations = jax.tree.leaves(params)[0].shape[0]

    recent_history_init = eqx.filter_vmap(
        RecentHistory.init,
        in_axes=(0, None),
    )(params, 20)

    # Per-realization done flag.
    init_done = jnp.zeros((n_realizations,), dtype=jnp.bool_)

    batch_state = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_realizations, *x.shape)),
        batch_state,
    )

    n = n_iterations + 1

    def scan_step(carry: Carry, xs):
        return batch_step_(carry, xs, update_class, dataset)

    scan_fn = scan_tqdm(n)(scan_step) if verbose else scan_step

    (_, opt_state, _, batch_state, _, _), history = jax.lax.scan(
        f=scan_fn,
        init=(
            params,
            opt_state,
            key,
            batch_state,
            init_done,
            recent_history_init,
        ),
        xs=jnp.arange(n),
        length=n,
    )

    # ``history`` leaves come out with shape (n_iter, n_realizations, ...);
    # swap so realization is leading and per-realization helpers see the
    # canonical (n_iter, ...) layout they were written against.
    history: OptHistory = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), history
    )

    def _finalize(
        h: OptHistory,
    ) -> tuple[
        InputParameters, InputParameters, Array[Float, ""], HistoryState
    ]:
        best_params, best_loss = h.flatten().retrieve_best()
        # Drop the leading sentinel iteration from ``n_iterations + 1``.
        h = jax.tree.map(lambda x: x[1:], h)
        final_params = jax.tree.map(lambda x: x[-1], h.params)
        history_state = HistoryState.from_history(h)
        return final_params, best_params, best_loss, history_state

    params, best_params, best_loss, new_history_state = eqx.filter_vmap(
        _finalize
    )(history)

    return (
        params,
        best_params,
        best_loss,
        opt_state,
        batch_state,
        new_history_state,
    )


# =============================================================================


def run_randomsearch(
    update_class: UpdateClass,
    opt_state: PyTree,
    params: InputParameters,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    key: PRNGKeyArray,
    n_iterations: int,
    verbose: bool,
) -> tuple[
    InputParameters,
    InputParameters,
    Array[Float, ""],
    PyTree,
    BatchState,
    HistoryState,
]:
    """Run random search by evaluating all candidates in one batched call.

    The entire ``n_iterations * popsize`` candidate set is drawn from
    ``update_class.sampling_fn``, clipped to ``update_class.bounded``,
    and evaluated in parallel via :func:`vmapped_loss` (or
    :func:`vmapped_loss_with_rng` for stochastic losses). The flat loss
    vector is reshaped into ``(n_iterations, popsize, ...)`` so the
    downstream :class:`HistoryState` reduction yields one
    ``output_min/mean/std`` value per logical iteration. The scan-loop
    machinery used by :func:`run_` is bypassed -- random-search
    candidates are independent and gain nothing from sequential
    iteration.

    Parameters
    ----------
    update_class : UpdateClass
        Must be a ``RandomSearchUpdateClass`` instance, providing
        ``sampling_fn``, ``static_model``, ``loss_fn``, ``pass_rng``,
        ``bounded``, ``popsize``, and ``hash`` attributes.
    opt_state : PyTree
        Ignored (random search keeps no state); returned unchanged.
    params : InputParameters
        Ignored (each iteration re-samples from scratch).
    batch_state : BatchState
        Used to draw one batch; advanced once.
    dataset : dict[str, jax.Array]
        Task dataset.
    key : PRNGKeyArray
        PRNG key, split into batch / sampling / per-candidate-loss
        sub-keys.
    n_iterations : int
        Number of logical iterations to emit in the trajectory.
    verbose : bool
        Ignored.

    Returns
    -------
    tuple[InputParameters, InputParameters, Array[Float, ""], PyTree,
    BatchState, HistoryState]
        ``(final_params, best_params, best_loss, opt_state,
        batch_state, history_state)`` matching :func:`run_`.
    """
    del params, verbose
    total = n_iterations * update_class.popsize

    batch_key, sample_key, loss_key = jr.split(key, 3)
    loss_keys = jr.split(loss_key, total)

    # One batch reused for all evaluations.
    batch_idxs, batch_state = batch_state.next(batch_key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)

    # Flat candidate set.
    candidates = update_class.sampling_fn(sample_key, total)
    candidates = jax.tree.map(
        lambda p: jnp.clip(p, *update_class.bounded), candidates
    )

    # Parallel evaluation.
    combined = eqx.combine(candidates, update_class.static_model)
    if update_class.pass_rng:
        losses = vmapped_loss_with_rng(
            combined, update_class.loss_fn, sample, loss_keys
        )
    else:
        losses = vmapped_loss(combined, update_class.loss_fn, sample)

    # Reshape to (n_iter, popsize, ...) for OptHistory.
    losses_2d = losses.reshape(n_iterations, update_class.popsize)
    params_2d = jax.tree.map(
        lambda x: x.reshape(n_iterations, update_class.popsize, *x.shape[1:]),
        candidates,
    )
    grads_2d = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), params_2d)

    history = OptHistory(
        loss=losses_2d,
        params=params_2d,
        grads=grads_2d,
        update_step=jnp.full((n_iterations,), update_class.hash, dtype=int),
        fevals=jnp.full((n_iterations,), update_class.popsize, dtype=int),
        iterations=jnp.ones((n_iterations,), dtype=int),
    )
    history_state = HistoryState.from_history(history)

    # Global best across the full flat candidate set.
    best_idx = jnp.argmin(losses)
    best_params = jax.tree.map(lambda x: x[best_idx], candidates)
    best_loss = losses[best_idx]

    # "Current population" returned to RunState -- last iteration's
    # popsize candidates, matching run_'s contract.
    final_params = jax.tree.map(lambda x: x[-1], params_2d)

    return (
        final_params,
        best_params,
        best_loss,
        opt_state,
        batch_state,
        history_state,
    )


def batch_run_randomsearch(
    update_class: UpdateClass,
    opt_state: PyTree,
    params: InputParameters,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    key: PRNGKeyArray,
    n_iterations: int,
    verbose: bool,
) -> tuple[
    InputParameters,
    InputParameters,
    Array[Float, ""],
    PyTree,
    BatchState,
    HistoryState,
]:
    """Run ``n_realizations`` independent random-search trajectories.

    Vmaps :func:`run_randomsearch` over a leading ``n_realizations``
    axis so each realization draws its own
    ``n_iterations * popsize`` candidates from an independently split
    PRNG key. The role of this function for random search mirrors what
    :func:`batch_run_` does for the scan-based optimizers, and the
    output shape contract matches.

    Inputs ``opt_state``, ``params``, and ``key`` already carry the
    realization axis (``key`` is pre-split by the caller, see
    :func:`batch_evaluate`); ``batch_state`` is broadcast to that same
    axis here so every leaf of the vmap input is mappable on axis 0.

    Parameters
    ----------
    update_class : UpdateClass
        Must be a ``RandomSearchUpdateClass`` instance.
    opt_state : PyTree
        Per-realization scalar placeholder, shape ``(n_realizations,)``.
        Returned unchanged.
    params : InputParameters
        Per-realization parameters with shape
        ``(n_realizations, popsize, ...)``. Unused (each iteration
        re-samples from scratch) but the realization axis is read to
        infer ``n_realizations``.
    batch_state : BatchState
        Single (un-batched) batch state; broadcast across realizations
        internally so each realization advances its own copy.
    dataset : dict[str, jax.Array]
        Dataset for evaluation (shared across realizations).
    key : PRNGKeyArray
        Per-realization PRNG keys, shape ``(n_realizations,)`` --
        already split by the caller.
    n_iterations : int
        Number of logical iterations to emit per realization.
    verbose : bool
        Ignored.

    Returns
    -------
    tuple[InputParameters, InputParameters, Array[Float, ""], PyTree,
    BatchState, HistoryState]
        ``(params, best_params, best_loss, opt_state, batch_state,
        history_state)`` -- each output carries a leading
        ``n_realizations`` axis.
    """
    del verbose
    n_realizations = jax.tree.leaves(params)[0].shape[0]

    batch_state = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_realizations, *x.shape)),
        batch_state,
    )

    def _single(opt_state, params, batch_state, key):
        return run_randomsearch(
            update_class=update_class,
            opt_state=opt_state,
            params=params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=False,
        )

    return eqx.filter_vmap(_single, in_axes=(0, 0, 0, 0))(
        opt_state, params, batch_state, key
    )


# =============================================================================


def _default_stop_fn(*_, **__) -> bool:
    return False


class UpdateClass(eqx.Module):
    """Class for managing optimizer update operations.

    This class encapsulates the initialization, stepping, and stopping logic
    for optimization algorithms.

    Attributes
    ----------
    init_fn : InitFunction
        Function to initialize optimizer state.
    step_fn : StepFunction
        Function to perform optimization step.
    popsize : int
        Population size for the optimizer.
    hash : int
        Hash identifier for the optimizer configuration.
    stop_fn : StopFunction
        Function to check stopping criteria. Defaults to a function that
        never stops.
    """

    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)
    stop_fn: StopFunction = eqx.field(static=True, default=_default_stop_fn)


# =============================================================================


def _evaluate_one(
    model: PyTree,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    pass_rng: bool,
    batch_key: PRNGKeyArray,
    loss_key: PRNGKeyArray,
) -> jax.Array:
    """Evaluate a single model on a batch of data.

    Parameters
    ----------
    model : PyTree
        Model to evaluate.
    batch_state : BatchState
        Batch state for data sampling.
    dataset : dict[str, jax.Array]
        Full dataset.
    loss_fn : Callable
        Loss function.
    pass_rng : bool
        Whether to pass RNG key to the loss function.
    batch_key : PRNGKeyArray
        PRNG key for batch sampling.
    loss_key : PRNGKeyArray
        PRNG key for the loss function.

    Returns
    -------
    jax.Array
        Loss value.
    """
    batch_idxs, _ = batch_state.next(batch_key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
    if pass_rng:
        return loss_fn(model, key=loss_key, **sample)
    else:
        return loss_fn(model, **sample)


evaluate = eqx.filter_vmap(
    _evaluate_one, in_axes=(0, None, None, None, None, None, 0)
)
