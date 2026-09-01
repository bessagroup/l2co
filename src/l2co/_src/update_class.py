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
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Local
from .batching import BatchState
from .history_state import HistoryState
from .opt_history import OptHistory, RecentHistory
from .optimizers.update_class import UpdateClass
from .tree_utils import vmapped_loss, vmapped_loss_with_rng
from .typing import (
    Carry,
    InputParameters,
)

__all__ = [
    "UpdateClass",
    "batch_run_",
    "batch_run_randomsearch",
    "batch_run_sequential_",
    "batch_step_",
    "evaluate",
    "init_",
    "run_",
    "run_randomsearch",
    "step",
]

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


def _reduce_entry(
    entry: OptHistory,
) -> tuple[tuple[Array, ...], PyTree, Array]:
    """Reduce one per-step ``OptHistory`` to the fields a run needs.

    Returns ``(stats, step_best_params, step_best_loss)`` where ``stats``
    is ``(output_min, output_mean, output_std, update_step, fevals,
    iterations)`` for this single generation -- the same NaN-aware
    reductions :meth:`HistoryState.from_history` computes over the
    ``popsize`` axis, applied per step so the full ``(n_iter, popsize,
    dim)`` population/gradient trajectory never has to be stacked as scan
    output. ``step_best_params`` / ``step_best_loss`` are the best
    (lowest-loss) member of this generation, threaded through the scan
    carry to maintain a running best in place of a post-hoc
    ``OptHistory.retrieve_best`` over the stacked trajectory.

    Operates on an unbatched entry (``loss`` shape ``(popsize,)``); the
    batched driver vmaps it over the realization axis.
    """
    loss = entry.loss
    output_min = jnp.nanmin(loss)
    stats = (
        output_min,
        jnp.nanmean(loss),
        jnp.nanstd(loss),
        entry.update_step,
        entry.fevals,
        entry.iterations,
    )
    # NaN-safe argmin: NaN slots (padding / skipped steps) are pushed to
    # +inf so they are never selected, matching ``jnp.nanargmin`` on the
    # flattened trajectory used by ``OptHistory.retrieve_best``.
    safe = jnp.where(jnp.isnan(loss), jnp.inf, loss)
    best_idx = jnp.argmin(safe)
    step_best_params = jax.tree.map(lambda p: p[best_idx], entry.params)
    return stats, step_best_params, output_min


def _update_best(
    best_params: PyTree,
    best_loss: Array,
    cand_params: PyTree,
    cand_loss: Array,
) -> tuple[PyTree, Array]:
    """Keep the lower-loss of the running best and a new candidate.

    Strict ``<`` so the *earliest* generation wins ties, matching the
    row-major first-occurrence of ``jnp.nanargmin`` over the flattened
    trajectory. A ``NaN`` candidate loss compares ``False`` and leaves the
    running best untouched. Unbatched (scalar ``cand_loss``); the batched
    driver vmaps it over the realization axis.
    """
    take = cand_loss < best_loss
    best_loss = jnp.where(take, cand_loss, best_loss)
    best_params = jax.tree.map(
        lambda c, b: jnp.where(take, c, b), cand_params, best_params
    )
    return best_params, best_loss


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

    # Reduce each generation in place (``_reduce_entry``) instead of
    # stacking the full ``(n, popsize, dim)`` population/gradient
    # trajectory as scan output and reducing afterwards: the trajectory
    # stays out of the scan output, so peak memory no longer scales with
    # ``dim`` or ``popsize``. The running best and the last evaluated
    # population ride in the carry. ``params[0]`` seeds the running best
    # with a correctly shaped/typed member (overwritten on the first,
    # always-executed step); ``best_loss`` is +inf in the loss dtype so
    # the carry type is stable across the scan.
    float_dtype = jax.tree.leaves(params)[0].dtype
    best_params_init = jax.tree.map(lambda p: p[0], params)
    best_loss_init = jnp.array(jnp.inf, dtype=float_dtype)
    eval_params_init = jax.tree.map(
        lambda p: jnp.full_like(p, jnp.nan), params
    )

    def scan_step(carry_ext, xs):
        carry, best_params, best_loss, _eval = carry_ext
        new_carry, entry = step(carry, xs, update_class, dataset)
        stats, cand_params, cand_loss = _reduce_entry(entry)
        best_params, best_loss = _update_best(
            best_params, best_loss, cand_params, cand_loss
        )
        # ``entry.params`` is the evaluated population (``input_params``,
        # NaN on skipped steps) -- what the legacy path returned as
        # ``final_params`` via ``history.params[-1]``. Thread it forward so
        # the last step's value survives without stacking the trajectory.
        return (new_carry, best_params, best_loss, entry.params), stats

    scan_fn = scan_tqdm(n)(scan_step) if verbose else scan_step

    init_carry = (
        params,
        opt_state,
        key,
        batch_state,
        init_done,
        recent_history_init,
    )
    final_ext, stats = jax.lax.scan(
        f=scan_fn,
        init=(
            init_carry,
            best_params_init,
            best_loss_init,
            eval_params_init,
        ),
        xs=jnp.arange(n),
        length=n,
    )

    (_, opt_state, _, batch_state, _, _), best_params, best_loss, params = (
        final_ext
    )

    # Drop the leading sentinel iteration (parity with the legacy
    # ``history[1:]`` slice) before assembling the reduced HistoryState.
    stats = tuple(s[1:] for s in stats)
    new_history_state = HistoryState.from_reduced(*stats)

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

    # Per-realization running-best / last-evaluated-population seeds; see
    # ``run_`` for the reduce-in-scan rationale. ``params[:, 0]`` seeds one
    # correctly-shaped best member per realization; ``best_loss`` is +inf
    # in the loss dtype so the carry type is stable across the scan.
    float_dtype = jax.tree.leaves(params)[0].dtype
    best_params_init = jax.tree.map(lambda p: p[:, 0], params)
    best_loss_init = jnp.full((n_realizations,), jnp.inf, dtype=float_dtype)
    eval_params_init = jax.tree.map(
        lambda p: jnp.full_like(p, jnp.nan), params
    )

    reduce_batch = jax.vmap(_reduce_entry)
    update_best_batch = jax.vmap(_update_best)

    def scan_step(carry_ext, xs):
        carry, best_params, best_loss, _eval = carry_ext
        new_carry, entry = batch_step_(carry, xs, update_class, dataset)
        stats, cand_params, cand_loss = reduce_batch(entry)
        best_params, best_loss = update_best_batch(
            best_params, best_loss, cand_params, cand_loss
        )
        return (new_carry, best_params, best_loss, entry.params), stats

    scan_fn = scan_tqdm(n)(scan_step) if verbose else scan_step

    init_carry = (
        params,
        opt_state,
        key,
        batch_state,
        init_done,
        recent_history_init,
    )
    final_ext, stats = jax.lax.scan(
        f=scan_fn,
        init=(
            init_carry,
            best_params_init,
            best_loss_init,
            eval_params_init,
        ),
        xs=jnp.arange(n),
        length=n,
    )

    (_, opt_state, _, batch_state, _, _), best_params, best_loss, params = (
        final_ext
    )

    # ``stats`` leaves are (n_iter, n_realizations); drop the leading
    # sentinel iteration and move realization to the front so
    # ``from_reduced`` (vmapped over realizations) sees canonical
    # (n_iter,) fields. This transpose is on scalar-per-step arrays, not
    # the (n, popsize, dim) trajectory the legacy swapaxes copied.
    stats = tuple(jnp.swapaxes(s[1:], 0, 1) for s in stats)
    new_history_state = jax.vmap(HistoryState.from_reduced)(*stats)

    return (
        params,
        best_params,
        best_loss,
        opt_state,
        batch_state,
        new_history_state,
    )


# =============================================================================


def batch_run_sequential_(
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
    """Run ``n_realizations`` trajectories one at a time.

    Same inputs, same outputs and same per-realization arithmetic as
    :func:`batch_run_` -- it maps :func:`run_` over the realization axis
    with :func:`jax.lax.map` (sequentially) instead of vmapping
    :func:`step` inside one fused scan. The role this plays for the
    meta-optimizers mirrors what :func:`batch_run_randomsearch` does for
    random search, and the output shape contract matches both.

    Why a sequential map is ever worth having
    -----------------------------------------
    A meta-optimizer's step chooses a sub-optimizer and runs only that
    one, expressed as a :func:`jax.lax.switch` on the chosen index.
    ``vmap`` cannot do per-lane control flow: with a batched index the
    switch has to evaluate **every** branch and select from the results.
    So under :func:`batch_run_` a menu of ``(lbfgs, rprop, crfmnes,
    mr15ga)`` pays, on every single generation, a width-1 gradient
    evaluation *and* a width-28 population evaluation *and* a width-27
    one *and* L-BFGS's line search -- then throws all but one away. The
    waste is per-generation and it scales with how expensive the task's
    loss is, so it hurts most exactly where it can least be afforded.
    One realization at a time keeps the index a scalar, so only the
    chosen branch runs.

    The trade is the SIMD width across realizations, and it is a real
    trade rather than a free win. The saving grows with what a branch
    costs -- so with the task's loss and with the menu's population sizes
    -- while the lost vectorization is roughly fixed. Below the crossover
    a menu-based optimizer is genuinely better off vectorized.
    Measured on ``bbob`` ``sphere`` at 25 realizations with the
    four-entry ``(lbfgs, rprop, crfmnes, mr15ga)`` menu, sequential
    against fused:

    ==========  =============
    ``d``       speedup
    ==========  =============
    2           0.72x
    10          0.64x
    40          1.03x
    64          1.02x
    256         1.83x
    1024        2.89x
    ==========  =============

    and 8.7x on the 2241-dimensional ``convection`` PINN. So this driver
    is taken unconditionally for a menu-based optimizer: it gives back
    about 1.5x on the small-``d`` cells, which are seconds of work, and
    it is break-even to 8.7x everywhere the runtime actually is
    (``conf/budgets/based_on_dim.yaml`` records that ``d >= 64`` is 96.6%
    of BBOB runtime). A plain optimizer has no switch to collapse and
    stays on :func:`batch_run_`, which is strictly faster for it.
    :func:`l2co._src.run_state.batch_run` picks between them on
    :attr:`UpdateClass.sequential_realizations`.

    Equivalence to :func:`batch_run_`
    ---------------------------------
    Each realization sees the same ``opt_state`` / ``params`` /
    ``batch_state`` / ``key`` slice it would as a ``vmap`` lane, and
    :func:`run_` applies the same per-lane initialisation and reductions
    :func:`batch_run_` applies through its vmaps. Same inputs, same
    arithmetic, same output contract -- but not the same bits, and over
    a long run not even the same third significant figure.

    What holds exactly is the sequence of choices: ``update_step``
    agrees generation for generation, so both drivers run the same
    sub-optimizer at the same time on the same realization.

    What drifts is everything continuous, and it drifts because an
    optimizer trajectory amplifies. ``vmap``-of-``switch`` compiles a
    different (all-branches-then-select) program than a single-branch
    scan, so XLA does not pin reduction order across the two, and the
    starting disagreement is a few ulp. A line-search sub-optimizer then
    turns that into a discrete difference -- how many trials L-BFGS bills
    depends on a continuous comparison, so ``fevals`` can differ by one
    -- and the slightly different step feeds the next generation.
    Measured on the ``convection`` PINN over 120 generations: median
    relative difference in ``output_min`` 2e-15, but 3e-2 at the worst
    entry, and the divergence appears only in the realizations that got
    far enough to be in a sensitive region (the ones still far from a
    minimum agree to round-off throughout).

    So: a cell recomputed on this driver is the same experiment, not the
    same numbers as one stored from :func:`batch_run_`. rl2co's
    ``rl2co._src.rollout.unroll.map_unroll`` documents the same trade for
    the same reason on the training side.

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
        internally so each realization advances its own copy.
    dataset : dict[str, jax.Array]
        Dataset for evaluation (shared across realizations).
    key : PRNGKeyArray
        Per-realization PRNG keys, shape ``(n_realizations,)`` --
        already split by the caller (see
        :meth:`l2co.RolloutWrapper.batch_evaluate`).
    n_iterations : int
        Number of iterations to perform.
    verbose : bool
        Ignored. A per-realization :func:`jax_tqdm.scan_tqdm` bar would
        emit ``n_realizations`` interleaved progress bars from inside
        the map, which is worse than none; :func:`batch_run_randomsearch`
        drops it for the same reason.

    Returns
    -------
    tuple[InputParameters, InputParameters, Array[Float, ""], PyTree,
    BatchState, HistoryState]
        ``(params, best_params, best_loss, opt_state, batch_state,
        history_state)`` -- each output carries a leading
        ``n_realizations`` axis, matching :func:`batch_run_`.
    """
    del verbose
    n_realizations = jax.tree.leaves(params)[0].shape[0]

    batch_state = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_realizations, *x.shape)),
        batch_state,
    )

    def _single(slices):
        opt_state, params, batch_state, key = slices
        return run_(
            update_class=update_class,
            opt_state=opt_state,
            params=params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=False,
        )

    return jax.lax.map(_single, (opt_state, params, batch_state, key))


# =============================================================================


# Target number of candidates per random-search chunk. Peak memory for a
# chunk is roughly this many rows times the parameter dimension, times a
# small constant for the RNG generation. ~1e5 keeps a 1024-dim chunk well
# under 1 GiB while staying large enough that the per-chunk vectorised
# evaluation is efficient.
_RS_CHUNK_TARGET_CANDIDATES = 100_000


def _rs_chunking(n_iterations: int, popsize: int) -> tuple[int, int]:
    """Pick ``(n_chunks, chunk_iters)`` for chunked random search.

    Splits the ``n_iterations`` logical iterations into ``n_chunks``
    balanced chunks of ``chunk_iters`` iterations each such that a chunk
    holds about :data:`_RS_CHUNK_TARGET_CANDIDATES` candidates. Both
    values are static (``n_iterations``/``popsize`` are Python ints), so
    they fix shapes at trace time. ``n_chunks * chunk_iters`` may slightly
    exceed ``n_iterations`` (by fewer than ``n_chunks`` iterations); the
    caller masks/drops that padding.
    """
    target_iters = max(1, _RS_CHUNK_TARGET_CANDIDATES // max(popsize, 1))
    n_chunks = max(1, -(-n_iterations // target_iters))  # ceil div
    chunk_iters = -(-n_iterations // n_chunks)  # ceil div -> balanced
    return n_chunks, chunk_iters


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
    """Run random search over the candidate set, one chunk per scan step.

    Random search draws ``n_iterations * popsize`` independent candidates
    from ``update_class.sampling_fn``, evaluates them via
    :func:`vmapped_loss` (or :func:`vmapped_loss_with_rng` for stochastic
    losses), and reports one ``output_min/mean/std`` per logical iteration
    plus the global best. Rather than materialise and evaluate the whole
    set at once -- which peaks at several times its own size (``jax.random``
    uniform generation alone spikes to ~4x the result while producing
    threefry bits), the cause of out-of-memory failures on
    high-dimensional tasks -- the candidates are processed in chunks of
    :func:`_rs_chunking` iterations under a :func:`jax.lax.scan`. Each
    chunk's candidate array is transient (not part of the scan carry) and
    freed before the next step, so peak memory is
    ``O(chunk_iters * popsize * dim)`` independent of the total budget.
    The scan-loop machinery of :func:`run_` is still bypassed within a
    chunk: candidates are independent and evaluated in one vectorised
    call. Per-iteration reductions and the global best are identical to a
    single-shot evaluation of the same per-chunk draws.

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
    popsize = update_class.popsize
    n_chunks, chunk_iters = _rs_chunking(n_iterations, popsize)
    padded_iters = n_chunks * chunk_iters  # >= n_iterations
    chunk_size = chunk_iters * popsize

    batch_key, sample_key, loss_key = jr.split(key, 3)

    # One batch reused for all evaluations.
    batch_idxs, batch_state = batch_state.next(batch_key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)

    # Independent per-chunk sampling / loss keys.
    sample_keys = jr.split(sample_key, n_chunks)
    loss_keys = jr.split(loss_key, n_chunks)

    # ``bounded`` is a static field, so this is resolved at trace time.
    # With ``(None, None)`` -- the default for the random-search baseline
    # -- ``jnp.clip`` is a no-op that still allocates a full copy of the
    # chunk, so skip it unless a finite bound is actually set.
    lo, hi = update_class.bounded
    do_clip = lo is not None or hi is not None

    def _chunk(carry, xs):
        c_idx, s_key, l_key = xs

        candidates = update_class.sampling_fn(s_key, chunk_size)
        if do_clip:
            candidates = jax.tree.map(
                lambda p: jnp.clip(p, lo, hi), candidates
            )
        combined = eqx.combine(candidates, update_class.static_model)
        if update_class.pass_rng:
            losses = vmapped_loss_with_rng(
                combined,
                update_class.loss_fn,
                sample,
                jr.split(l_key, chunk_size),
            )
        else:
            losses = vmapped_loss(combined, update_class.loss_fn, sample)

        # Mask candidates in padded (out-of-range) iterations so they can
        # neither win the global best nor pollute the trajectory.
        valid_iter = (
            c_idx * chunk_iters + jnp.arange(chunk_iters)
        ) < n_iterations
        valid_cand = jnp.repeat(valid_iter, popsize)
        masked = jnp.where(valid_cand, losses, jnp.inf)

        # Per-chunk best (``argmin`` keeps the earliest candidate on ties).
        best_idx = jnp.argmin(masked)
        chunk_best_loss = masked[best_idx]
        chunk_best_params = jax.tree.map(lambda x: x[best_idx], candidates)

        losses_2d = losses.reshape(chunk_iters, popsize)

        # Population at this chunk's last in-range iteration. Only the
        # final chunk's value is used downstream, giving the last logical
        # iteration's ``popsize`` block (matches ``candidates[-popsize:]``
        # of the previous one-shot implementation).
        last_valid_local = jnp.clip(
            n_iterations - 1 - c_idx * chunk_iters, 0, chunk_iters - 1
        )
        last_pop = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, last_valid_local * popsize, popsize, axis=0
            ),
            candidates,
        )
        return carry, (losses_2d, last_pop, chunk_best_loss, chunk_best_params)

    _, (losses_2d_all, last_pops, chunk_best_losses, chunk_best_params) = (
        jax.lax.scan(
            _chunk,
            (),
            (jnp.arange(n_chunks), sample_keys, loss_keys),
        )
    )

    # Reduce the flat candidate losses per logical iteration directly into
    # the HistoryState (dropping the padded tail). Building an intermediate
    # ``(n_iter, popsize, dim)`` ``OptHistory`` -- and an all-NaN ``grads``
    # array of the same shape -- only to collapse it via ``from_history``
    # wastes memory scaling with ``dim``; random-search candidates carry no
    # NaN padding, so per-iteration ``nanmin/nanmean/nanstd`` match exactly.
    losses_2d = losses_2d_all.reshape(padded_iters, popsize)[:n_iterations]
    history_state = HistoryState.from_reduced(
        jnp.nanmin(losses_2d, axis=1),
        jnp.nanmean(losses_2d, axis=1),
        jnp.nanstd(losses_2d, axis=1),
        jnp.full((n_iterations,), update_class.hash, dtype=int),
        jnp.full((n_iterations,), popsize, dtype=int),
        jnp.ones((n_iterations,), dtype=int),
    )

    # Global best across all chunks (earliest chunk wins ties).
    best_chunk = jnp.argmin(chunk_best_losses)
    best_loss = chunk_best_losses[best_chunk]
    best_params = jax.tree.map(lambda x: x[best_chunk], chunk_best_params)

    # "Current population" returned to RunState -- the last logical
    # iteration's ``popsize`` candidates, matching ``run_``'s contract.
    final_params = jax.tree.map(lambda x: x[-1], last_pops)

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

    Maps :func:`run_randomsearch` over a leading ``n_realizations`` axis
    with :func:`jax.lax.map` -- i.e. *sequentially*, one realization at a
    time -- so each realization draws its own ``n_iterations * popsize``
    candidates from an independently split PRNG key. Unlike an
    :func:`eqx.filter_vmap` (which materialises every realization's
    ``(n_iterations * popsize, dim)`` candidate set concurrently, so peak
    memory scales with ``n_realizations``), ``lax.map`` keeps only one
    realization's candidate set live at a time: the transient candidates
    are not part of the scan carry and are freed each step, while the
    stacked per-realization outputs (``best_params``, ``history_state``,
    ...) are small. Results match the vmap version because the
    per-realization ``key`` slices -- and therefore the drawn candidates
    -- are unchanged; only execution order (parallel -> sequential)
    differs (up to float32 reduction-ordering noise XLA does not pin
    across the two lowerings). The role of this function for random
    search mirrors what
    :func:`batch_run_` does for the scan-based optimizers, and the output
    shape contract matches.

    Inputs ``opt_state``, ``params``, and ``key`` already carry the
    realization axis (``key`` is pre-split by the caller, see
    :func:`batch_evaluate`); ``batch_state`` is broadcast to that same
    axis here so every leaf of the mapped input is sliceable on axis 0.

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

    def _single(slices):
        opt_state, params, batch_state, key = slices
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

    # Sequential (not vmapped) over realizations: bounds peak memory to a
    # single realization's candidate set instead of all ``n_realizations``
    # at once. See the docstring for why results are identical.
    return jax.lax.map(_single, (opt_state, params, batch_state, key))


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
