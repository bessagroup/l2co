# optax integration friction

Constraints of the `optax` API encountered while implementing
optimizers against it (first case: per-eval-key L-BFGS,
`l2co._src.optimizers.lbfgs`). Recorded so future
integrations don't re-discover them. Verified against optax 0.2.8.

## `fn_kwargs` is frozen for the whole zoom linesearch

`scale_by_zoom_linesearch.update_fn` extracts the extra keyword
arguments of `value_fn` once (`optax/_src/linesearch.py`, the
`_extract_fns_kwargs` call) and binds them unchanged into the
`lax.while_loop` body. A PRNG key passed as
`solver.update(..., key=...)` therefore reaches *every* trial
evaluation of that outer step identically — there is no hook to vary
per-iteration kwargs, so per-evaluation noise realizations require
a replacement linesearch. l2co's
`scale_by_zoom_linesearch_per_eval_key` re-drives optax's own
`zoom_linesearch` triple with a `(key, state)` while-loop carry and
one `jr.split` per inner evaluation.

## `zoom_linesearch`'s triple is not public API

Only the assembled `scale_by_zoom_linesearch` (plus
`ScaleByZoomLinesearchState` / `ZoomLinesearchInfo`) is re-exported
by `optax`'s public namespace; the underlying
`(init_fn, step_fn, cond_fn)` triple must be imported from
`optax._src.linesearch`. This is the one private import in
`lbfgs.py` — re-check it on optax upgrades
(`uv.lock` pins 0.2.8).

## `value_and_grad_from_state` serves one-realization-stale caches

The utility returns the value/gradient cached at the end of the
previous step's linesearch and recomputes only when the cached value
is non-finite. For a stochastic `value_fn` this means the pair that
seeds step *t*'s descent direction and Wolfe reference
(`value_init` / `slope_init`) was computed under step *t−1*'s key,
while the trial evaluations run under step *t*'s key — every L-BFGS
step compares across two realizations no matter how keys are
threaded. This is a property of the caching utility, not of the
linesearch; it is inherited (documented, not worked around) by the
per-eval-key variant.

## No positive-curvature guard on the L-BFGS secant pairs

`scale_by_lbfgs` stores `ρ = 1/(yᵀs)` guarding only against exactly
zero (`optax/_src/transform.py`, the `weight = jnp.where(...)`
line); noisy gradients that make `yᵀs < 0` enter the memory as-is
and destroy positive-definiteness of the preconditioner (ascent
directions). Upstream is aware — the skip/cautious-update guard of
Nocedal & Wright p. 537 is requested in
[optax#1708](https://github.com/google-deepmind/optax/issues/1708) —
so severe-noise L-BFGS behavior may change on a future optax
upgrade; benchmark results depend on the pinned version.
