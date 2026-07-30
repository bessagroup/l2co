# L-BFGS bills one feval per line-search trial, not one per iteration

Every optimizer in the registry reported what it actually spent
except one. Population-based optimizers bill `popsize` evaluations per
generation; plain gradient optimizers bill the single evaluation they
make. `lbfgs` billed a flat `1` while its zoom line search evaluated
the objective once per trial point — up to `max_linesearch_steps`
(20) times in a single iteration. `docs/adr/0006` named this and
deferred it, on the grounds that the under-reporting rule was
unchanged from the previous wiring so relative comparisons stayed
fair. That reasoning held only while nothing depended on the absolute
number. It no longer does: `rl2co`'s `AUCEnv` bills each step its
`fevals` and normalizes by the episode's maximum billable cost, so an
optimizer that under-reports is not merely mis-measured, it is
mis-rewarded.

We decided to bill the line search. `OptHistory.fevals` for a
`lbfgs` step is now `num_linesearch_steps + recompute`, computed by
`step_fevals` in `l2co._src.optimizers.update_class` and used by both
`lbfgs` wirings (`optax_per_eval_key_fn` on stochastic tasks,
`optax_extra_kwargs_fn` on deterministic ones). This follows
`docs/adr/0004`, which made the same call for the strategy wrapper's
handshake generation and likewise accepted that convergence curves
shift.

## Where the count comes from

The line search already keeps it. `ZoomLinesearchInfo.
num_linesearch_steps` is reset at the start of each outer update and
incremented once per trial, and it is exactly the number of objective
evaluations: optax's line-search `step_fn` dispatches through a
`lax.cond` to *either* the interval search *or* the zoom, each of
which calls `value_and_grad_fn` once and increments the count once.
The `_try_safe_step` fallback selects among values already computed
and evaluates nothing. `optax.tree.get(state, "num_linesearch_steps")`
reads it, and returns a Python `None` for any optimizer that has no
line search — a *static* `None` at trace time, so `step_fevals` falls
back to `1` with no runtime branch and every other registry entry is
untouched.

The `recompute` term is the cache miss in
`optax.value_and_grad_from_state`, which recomputes only when the
value cached by the previous step is non-finite — in practice the
first step alone. `step_fevals` mirrors optax's own branch condition
rather than assuming "step 0".

## Logical cost, not hardware cost

The recompute term is billed *logically*. Under `eqx.filter_vmap` —
which the batched rollout driver `batch_step_` uses — optax's
`lax.cond` degrades to a `select` and both branches execute, so a
batched run really does pay for the recompute on every step, not just
the first. We bill the logical cost anyway, so that a single run and a
batched run of the same configuration report the same number. The
alternative (bill `+1` unconditionally) would make the reported cost
depend on how the run was driven, which is worse for a benchmark.

## The dominant cost is after convergence, not during the search

This is the part that makes the change matter, and it is not what
`docs/adr/0006` assumed when it estimated the impact.

While L-BFGS is still descending the search is cheap: it accepts its
first trial on nearly every step, so the honest count is 1–3 and the
old flat `1` was almost right. The expense appears once the optimizer
stops making progress. Measured on `create_bbob_task('sphere', dim=10)`:

```
 step count   |grad|      decrease_err   curvature_err   stepsize
    0     1  2.2823e+02   0.000000e+00   0.000000e+00   1.0000e+00
    1     1  2.8227e+01   0.000000e+00   0.000000e+00   1.0000e+00
    2     1  8.5963e-06   0.000000e+00   0.000000e+00   1.0000e+00
    3    17  7.1526e-06   0.000000e+00   2.398626e-14   1.0000e+00
    4    20  7.1526e-06   0.000000e+00   5.098616e-12   1.4736e-07
```

Convergence lands at step 2. From step 3 the run is a **fixed point** —
identical gradient, errors and stepsize forever — spending the full
20 evaluations every iteration for the remainder of the budget. The
mechanism: the curvature criterion is `|slope_step| - 0.9 *
|slope_init| <= tol` with `tol = 0.0` (optax's default, which l2co
does not override), and both errors are clamped at `>= 0`, so the
search succeeds only when they are *exactly* zero. At a residual
gradient of `7e-6` the curvature error is `5e-12` — small, non-zero,
unreachable. The collapsed-bracket early exit does not help: it is
gated on `new_safe_stepsize > 0.0`, and `safe_stepsize` is only set by
a step achieving sufficient *decrease*, which at a minimum does not
exist.

**This tail is landscape-dependent.** An objective that converges to
an *exactly* zero gradient — a textbook quadratic, `sum(x**2)` — has
both errors exactly `0.0`, satisfies `tol = 0.0`, and keeps costing
one trial per step indefinitely. Shifted, scaled, ill-conditioned and
large-constant-offset quadratics all behave the same way. The
saturation needs a small-but-non-zero residual, which is what real
landscapes leave. Mean *line-search trials* per step over 30 steps at
`max_linesearch_steps=20`, dim 10 (billed `fevals` is this plus the
one first-step recompute, so `+1/30` on these means):

| task | mean | at cap |
|---|---|---|
| `sphere` (deterministic) | 18.00 | 26/30 |
| `rosenbrock` (deterministic) | 1.30 | 0/30 |
| `rastrigin` (deterministic) | 4.93 | 3/30 |
| `f101` sphere + Gaussian (moderate) | 15.53 | 22/30 |
| `f103` sphere + Cauchy (moderate) | 18.00 | 26/30 |
| `f104` rosenbrock + Gaussian (moderate) | 1.90 | 0/30 |
| `f107` sphere + Gaussian (severe) | 14.50 | 19/30 |
| `f109` sphere + Cauchy (severe) | 18.00 | 26/30 |

The split is convergence, not noise: every Rosenbrock-based task stays
near 1 because it never converges inside the budget, while every
sphere-based one converges in ~3 steps and saturates for the other 27.
These are single-seed spot checks — they establish the mechanism, not
a distribution.

## Consequences

- **`UpdateClass` gains `max_fevals_per_step`** (static, `None` by
  default) and a `fevals_bound` property resolving `None` to
  `popsize`. `lbfgs_update` sets it to `max_linesearch_steps + 1`.
  Anything that sizes a budget or normalizes a per-evaluation cost
  must read `fevals_bound`, because `popsize` is no longer an upper
  bound on what one iteration can bill.

- **`rl2co`'s `AUCEnv` normalization changes with it.** It derived
  `max_iterations * max(popsize)` as "the episode's maximum billable
  fevals" and bounded returns in `[-1, 0]` on that basis. With `lbfgs`
  in the menu that bound was already wrong the moment this change
  landed — popsize 1, actual 21 — so returns could fall below `-1` and
  multi-task weighting would skew. It now derives
  `max_iterations * max(fevals_bound)`.

- **ERT moves by a small factor; total-budget metrics move by up to
  ~19x.** ERT measures evaluations-to-target, and targets are reached
  during the cheap descent phase while the expensive steps land after,
  so the two diverge sharply. Measured by pricing one run under both
  accountings (the change is accounting-only, so the losses are
  identical), 60 steps, dim 10, targets at `global_min + delta`:

  | task | cost-to-target, old -> new | total fevals, old -> new |
  |---|---|---|
  | `sphere` seed 0 | 3 -> 4 (1.33x) | 60 -> 1141 (19.0x) |
  | `sphere` seed 1 | 3 -> 20 (**6.67x**) | 60 -> 1160 (19.3x) |
  | `rosenbrock` seed 0 | 53 -> 72 (1.36x) | 60 -> 83 (1.4x) |
  | `rosenbrock` seed 1 | 56 -> 69 (1.23x) | 60 -> 75 (1.3x) |
  | `f101` seed 1 | 3 -> 4 (1.33x) | 60 -> 1122 (18.7x) |
  | `f104` seed 0 | 53 -> 90 (1.70x) | 60 -> 99 (1.7x) |
  | `f104` seed 1 | 52 -> 85 (1.63x) | 60 -> 95 (1.6x) |

  Across every run that reached a target the ratio clusters in
  **1.23x–1.70x**, with one outlier at **6.67x** — `sphere` seed 1,
  where the step that happened to cross the target was itself an
  expensive one. So the ERT shift is modest *on average* but not
  uniform: it depends on whether the target is crossed on a cheap step
  or an expensive one, which is essentially luck of the landscape.
  `docs/adr/0006` predicted this would "shift every ERT curve"; it
  does, but by ~1.3x rather than the ~20x that framing implied.

  **Regeneration is a judgement call, not a non-issue.** A ~1.3x
  systematic shift applies to `lbfgs` alone, so ERT-based rankings
  against other optimizers move by roughly that factor — small, but
  not noise. Existing databank records cannot be re-scored: they store
  the flat `1`, and the line-search counts needed for the new number
  were never recorded, so any correction requires re-running. Whether
  that is worth it depends on how much weight a given study puts on
  `lbfgs`'s ERT standing.

- **`lbfgs` data recorded before this change under-reports its cost**
  in `fevals`. Trajectories (losses, parameters) are unaffected: this
  changes accounting only, not a single optimizer step.

- The wrapper paths are unchanged and now inconsistent with the
  registry. `rl2co._src.optax_wrapper` cannot host a line-search
  sub-optimizer at all (no `value_fn` channel), so its flat `1` is
  correct. `rl2co._src.evosax_wrapper` can, and still bills `popsize`;
  fixing it requires reordering `tell`, which pushes its history entry
  *before* dispatching the sub-optimizer step, so the trial count does
  not yet exist. Deferred and documented in that module. This compounds
  the non-comparability `docs/adr/0008` already records for the
  `SubOpt` path.

## Rejected

- **Leaving the accounting alone and fixing only the `tol = 0.0`
  stall.** Raising `tol` would shrink the post-convergence tail at
  the source and is worth doing — a converged optimizer burning 20
  evaluations per iteration is wasted compute in every dataset run to
  date — but it is a separate change: it alters the *trajectory* (the
  search would accept steps it currently rejects) where this one does
  not, and it does not make the remaining line-search evaluations
  honest. Sequencing this ADR first keeps the trajectory-affecting
  change independent and lets any regeneration decision be made once,
  on honest numbers.

- **Billing `max_linesearch_steps` flat.** Simple and always an upper
  bound, but wrong by up to 20x during the descent phase — the phase
  ERT actually measures.

- **Keeping the flat `1` and exposing the trial count as a separate
  diagnostic.** Preserves comparability with existing data, but leaves
  `AUCEnv` rewarding a converged L-BFGS as though it were free, which
  is the concrete defect that forced this decision.
