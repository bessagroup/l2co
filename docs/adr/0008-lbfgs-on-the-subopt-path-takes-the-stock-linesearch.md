# "lbfgs" resolves to the stock linesearch on the SubOpt path, and the SubOpt registries must cover the optimizer registry

`optstep_to_subopt` resolves an optimizer name against five
registries — `normalized_optax_normal`,
`normalized_optax_from_state`, `l2co_native_optax`,
`normalized_evosax`, `l2co_native_evosax` — which are collectively
narrower than `optimizer_mapping`'s. When `docs/adr/0006` moved
`"lbfgs"` out of `opt_names_from_state` into its own module,
`normalized_optax_from_state` became empty and `"lbfgs"` silently
stopped resolving to a `SubOpt`: it stayed fully runnable through
the registry (`RunState.init`, dataset creation, every ordinary
rollout) while any menu containing it raised `ValueError` inside both
strategy wrappers. Nothing in either repo's test suite pinned the
correspondence, so the break surfaced only as three failing
L-BFGS wrapper tests in rl2co.

We decided to make `l2co_native_optax` the standing home for
optax-style transforms l2co assembles itself — the gradient-side
mirror of `l2co_native_evosax` — with `"lbfgs"` its first entry, and
to pin the registry correspondence with a coverage test: every name
in `optimizer_mapping` except the meta-optimizers (`"l2co"`,
`"rl2co"`, which select *among* sub-optimizers) must appear in one of
the five. That test, not the wrapper tests, is the guard against the
next module move.

**The entry is `stock_lbfgs`, not `lbfgs_per_eval_key`, on every
task — including stochastic ones.** This is a deliberate exception to
`docs/adr/0006`, forced by the ask/tell contract: the `value_fn` a
line search receives on this path is the caller's plain `params ->
scalar` callable (`strategy_wrapper.tell(..., value_fn=)` and rl2co's
equivalent), with no key to split. `lbfgs_per_eval_key` requires a
`key` extra kwarg that `SubOpt.step_fn` has no channel for — its
`key` positional is documented unused and the wrappers pass only
`value` / `grad` / `value_fn`. So on a stochastic task the noise
contract of a wrapper-driven L-BFGS is whatever the caller's
`value_fn` implements, *not* the one-realization-per-evaluation
contract `docs/adr/0006` guarantees for the registry path.

## Consequences

- Wrapper-driven and registry-driven `"lbfgs"` are not
  trajectory-comparable on stochastic tasks. Deterministic tasks are
  unaffected: `lbfgs_update` already delegates to `stock_lbfgs` when
  `task.pass_rng` is false, so both paths take the identical
  transform.
- Threading a per-evaluation key through the wrappers was rejected as
  out of scope: it would mean widening the `SubOpt` gradient contract
  (and both wrappers' `tell` signatures) to carry a key into
  `value_fn`, for one optimizer on one task family. Revisit if
  noisy-task wrapper rollouts become a real workload rather than a
  latent capability.
- Raising on `task.pass_rng` instead of degrading was also rejected:
  the caller's `value_fn` is opaque here, so l2co cannot tell a
  noise-handling one from a naive one, and refusing would block the
  deterministic-`value_fn`-on-a-stochastic-task case that is
  perfectly well defined.
- `_stock_lbfgs` is renamed `stock_lbfgs` (still `_src`-level, not on
  the `l2co.optimizers` facade) now that a second module references
  it.
