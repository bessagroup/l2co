# L2CO strategy runs through an evosax-style wrapper + UpdateClass bridge as the canonical execution path

The L2CO strategy (stage-0 + stage-1, `L2COModel`) used to execute
only through `RunOptimization.run_optimization_l2co` — an eager,
Python-level two-phase orchestration special-cased on
`optimizer == "l2co"`. We decided to replace that path entirely:
`L2COModel` is exposed as a duck-typed evosax strategy
(`l2co_population_strategy`, mirroring rl2co's
`rl2co_population_strategy`), bridged into an `l2co.UpdateClass` via
a `UpdateClass.l2co_update` classmethod and registered as `"l2co"`
in the optimizer registry — so `OptimizationStep(optimizer="l2co")`
runs through the ordinary jitted/vmapped `init_`/`step`/`run_`/
`batch_run_` machinery like any plain optimizer, and the special
case is deleted.

The wrapper is an **exact single-fire port**, not a generalization:
stage-0 fires at construction time (task features exist before the
run; keeps `ScikitStage0Model` usable since nothing sklearn enters a
jitted region), and stage-1 fires exactly once, in-graph via
`lax.cond`, at the static switch generation `round(t × budget)`.
A re-firing schedule was rejected because stage-1 was trained on
single-reassessment trajectory prefixes — repeated fires would be
out-of-distribution. Handshake on a switch is
`reset_opt_state` + `best_one` (fresh sub-optimizer state around the
best-so-far, population = best + sampler draws), reproducing
`reset_from_stage0`; on a same-optimizer reselection the run state
continues untouched. This deliberately differs from the rl2co
wrapper's `continue_opt_state` handshake.

## Consequences

- `AgenticL2COModel` (LLM fires, inherently eager — jax traces
  `lax.cond` branches even outside jit) cannot ride the bridge. The
  eager two-phase driver moves downstream into l2co_experiments'
  `RunOptimization` subclass, next to its only caller (the
  agentic-l2co branch). l2co itself retains zero optimizer
  special-casing.
- Stage-1 must have a traceable `__call__`
  (`EquinoxStage1Model`); this is validated at construction.
- `task.global_min` is required at construction (stage-1's
  staircase features were trained against the true optimum; a
  running-min fallback would be silently out-of-distribution).
- `swap_optimizers` becomes an explicit list of `OptimizationStep`
  dicts in the hyperparameters (the hardcoded 4-optimizer list in
  two repos is deleted); `budget_fn` and the handshake `sampler`
  also arrive via hyperparameters, following the `"rl2co"` registry
  precedent.
- Run histories carry a dynamic per-generation `update_step` hash
  (stage-0's optimizer before the switch, stage-1's after) instead
  of two concatenated per-stage histories.
