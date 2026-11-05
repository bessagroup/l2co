---
status: accepted
---

# Budgets count evaluations

A budget function decides how long each run in a dataset is. Until now
every budget function counted *iterations*: `based_on_dimensionality`
gave every optimizer 10³·d of them. One iteration of a population method
costs λ evaluations, so CMA-ES at 40-D got about 15 times the evaluations
of Adam. The dataset paper compares optimizers by evaluations, so
decisions 3 and 4 of its protocol revision give **every run the same
number of function evaluations, B = min(10³·d, 10⁵)**, whatever the
optimizer and whatever the suite. The rationale and the rejected
alternatives are in Section 3.2 of that revision.

The run loop takes a fixed number of iterations and does not change. The
budget function turns B into iterations, and to do that it must know
what one iteration costs. That is the only change to core code.

## The budget function

`evaluations_based_on_dimensionality(task, optimizer,
evaluations_per_dimension=1000, max_evaluations=100_000)` in
`_src/budget.py`, exported from `l2co.budget`, returns ⌊B / e⌋
iterations.
- **B** = min(`evaluations_per_dimension` · d, `max_evaluations`), with
  d = `task.dimensionality`, the number of parameters. For an embedded
  task that is the ambient dimension: `create_embedded_bbob_task` builds
  its model as `jnp.zeros(ambient_dim)`, and a test pins that a 2-in-64
  task gets 64 000 evaluations. `evaluation_budget(task,
  evaluations_per_dimension=1000, max_evaluations=100_000)`, also in
  `l2co.budget`, returns B itself. It needs no optimizer, so a caller
  that must know B per task (the databank dump, which checks every
  cell against it) gets the same number the budget function used.
- **e** is what one iteration of the optimizer costs, read from the
  built `UpdateClass`:
  - `popsize` when `max_fevals_per_step` is unset. Each step then costs
    exactly `popsize` evaluations, so the run spends at most B. This
    covers the evosax methods (λ = ⌊4 + 3 ln d⌋), TuRBO and RBF (5),
    the first-order methods (1), random search (its `popsize`), and the
    scipy and optimistix entries, which bill one evaluation per
    iteration.
  - 1 when `max_fevals_per_step` is set. Only L-BFGS (1 to 21
    evaluations per step) and Nelder–Mead (2 on an ordinary step, n + 1
    on the first, up to n + 3) set it. Their runs can pass B, and a
    Nelder–Mead run always does: on 10-D Rosenbrock with B = 200, two
    Nelder–Mead runs spent 497 and 464 evaluations and two L-BFGS runs
    287 and 275. Nelder–Mead's `popsize` is its simplex, n + 1, so
    reading `popsize` alone would have given it ⌊B / (n + 1)⌋ iterations.

Example: at 1024-D, B = 10⁵. Adam gets 100 000 iterations and CMA-ES
(λ = 24) gets 4 166.

**It refuses to guess.** Called without an optimizer, or with something
other than an `UpdateClass` (such as the `OptimizationStep` schedule), it
raises `TypeError`. Assuming e = 1 would give every population method λ
times its budget, and any re-run check that compares a stored iteration
count with this budget would see every population cell as short. A
budget smaller than one iteration (B < e) raises `ValueError` instead of
returning 0 iterations; only tiny custom settings reach it.

## The one core edit

`RunOptimization.run_optimization` (`_src/rollout/dataset_generation.py`)
used to call `budget_fn(task=task)` before building the optimizer. It
now calls it after `RolloutWrapper.init`, as
`budget_fn(task=task, optimizer=rollout_wrapper.run_state.update_class)`.
That is the same `UpdateClass` the rollout runs, including the
noise-free loss `RolloutWrapper.init` attaches (ADR 0024).

**Every budget function given to `RunOptimization` must accept
`optimizer`.** `fixed_budget` already took `**kwargs`.
`based_on_dimensionality` gains a keyword-only `optimizer=None` that it
ignores. Both keep their meaning, so existing configs run unchanged. A
hand-written `lambda task: n` passed to `RunOptimization` now fails with
`TypeError`; `lambda task, optimizer: n` or `fixed_budget` works.

## Who keeps the iteration budgets

Callers that have no built optimizer keep calling `budget_fn(task=task)`
with an iteration budget:
- the `"l2co"` meta-optimizer (`L2COUpdateClass.l2co_population_update`
  in `meta_optimizer.py`), which computes its switch generation from it;
- rl2co's factory and environment;
- l2co_experiments' re-run check (`_needs_run`, `_filter_to_missing`).

Given the new function, each of these raises `TypeError` rather than
silently using a wrong budget.

## Not covered

- **Cutting L-BFGS and Nelder–Mead at B** in post-processing. Their runs
  get B iterations and can spend more than B evaluations. Until
  post-processing ignores the evaluations past B, their stored runs are
  longer than B.
- **The dump-time check that a store holds one B.**
- **l2co_experiments**: the budget config for the new function and the
  rule that a populated cell counts as done in an evaluation-budget
  store. Its runner injects its own `budget_fn` into `"l2co"` and
  `"rl2co"` schedules, so those cells raise with the new function until
  that is settled.
- **What e should be for the meta-optimizers.** By the rule above they
  get e = `popsize`, which for `"l2co"`, `"rl2co"` and the random
  meta-policy is the largest `popsize` on their menu. They are billed the
  active sub-optimizer's cost per step, so a run that switches to a
  single-point method spends well under B. A menu with L-BFGS can bill
  more than e per step.

## Rejected

In one line each; Section 3.2 of the revision has the reasons.
- Budgeting everyone at their most expensive step (⌊B / `fevals_bound`⌋),
  which gives L-BFGS ⌊B / 21⌋ iterations.
- Stopping inside the run loop at B, which edits the core loop.
- Treating a missing optimizer as e = 1 (above).
