# The handshake policy table is promoted to l2co; only the decision is shared

Which handshake an optimizer needs on a switch — whether its internal
state can survive an iterate jump — is a property of the *optimizer*,
not of the RL loop that switches. Yet the table encoding it
(`HANDSHAKE_MAPPING`, with `lbfgs` as the one `reset_opt_state` entry)
lived in `rl2co._src.env.handshake`, so the two deployment wrappers
that also switch optimizers could not read it: `l2co._src.
strategy_wrapper` is in l2co, which cannot depend on rl2co, and
`rl2co._src.evosax_wrapper` / `optax_wrapper` simply hardcoded
`continue_opt_state` for every optimizer. Consequence: deploy a
trained policy with `lbfgs` in the menu and its curvature and
line-search state ride through a jump to `best_solution` — exactly the
case the table exists to prevent — while `strategy_wrapper` took the
opposite bias and reset *everything*, discarding CMA-ES adaptation the
env would have kept.

The policy table, a `HandshakePolicy` record (`population`:
`best` | `best_one`; `opt_state`: `reset` | `continue`), and the
`handshake_policy_for(opt_step)` resolver therefore move to
`l2co._src.optimizers.handshake_policy` and are exposed on the
`l2co.optimizers` authoring facade (ADR `0007`), following the same
promotion path as the `SubOpt` adapter (ADR `0003`). The resolved
`opt_state` half is carried on `SubOpt` as a static field, defaulted
to `'continue'` so existing `grad_sub_optimizer` / `pop_sub_optimizer`
callers are unaffected; `optstep_to_subopt` fills it from the
resolver, which is the only point on the wrapper path where the
optimizer *name* is still in scope (`SubOpt` otherwise keeps only
`name_hash`).

**Only the decision is shared, not the implementation.** The four
concrete handshake functions stay in rl2co, because they come in two
incompatible data models: the env's operate on `RunState` /
`UpdateClass`, the wrappers' on `SubOpt` and their own `State`, and
the two population halves have genuinely divergent layouts (the env's
`best_one` puts the best candidate in the *last* slot and emits
exactly `popsize` entries; both wrappers put it in slot 0 and NaN-pad
to the static `population_size`). `HANDSHAKE_MAPPING` becomes a
derivation over the l2co table rather than a second literal, so the
two cannot drift. Rejected: moving `HandshakeClass` itself to l2co
(the wrappers have no `RunState` to hand it, and it would force the
population-layout divergence to be resolved as a precondition);
keeping a duplicated table in each repo (guaranteed drift, the same
argument that settled ADR `0003`).

**`strategy_wrapper`'s unconditional reset is gated, not fixed.**
`_reset_active_substate` reproduces the published `reset_from_stage0`
semantics that ADR `0004`'s feval accounting is written against;
switching it to the per-optimizer table would change L2CO paper
results, which is a research decision rather than a bug fix. The
per-optimizer behaviour is therefore opt-in via an
`opt_state_handshake` argument on `l2co_population_strategy`
(`'legacy_reset'` default, `'per_optimizer'` opt-in). If
`'per_optimizer'` later becomes the default, that supersedes this
paragraph and needs its own ADR.
