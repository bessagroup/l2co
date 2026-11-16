---
status: accepted (implemented)
---

# The l2co meta-optimizer switches and stops by evaluations

The `"l2co"` meta-optimizer (`L2COUpdateClass.l2co_population_update`
in `meta_optimizer.py`) is a policy schedule: stage-0 picks an
optimizer, stage-1 may switch to another once a fraction `t` of the
budget is spent. Since l2co_experiments#123 every run in a databank
spends the same number of function evaluations,
B = min(1000·d, 10⁵) (ADR 0026 and its 2026-11-16 amendment). Until
this ADR the meta-optimizer still counted generations: it switched
after `round(t · budget_fn(task=task))` generations and ran as many
generations as the scan was long, whatever they cost. We decided that
it switches and stops by evaluations (l2co#128).

## What the budget function means

`budget_fn(task=task)` inside the factory now returns B, in
evaluations: the schedule budget function `evaluation_budget` (ADR 0026
amendment). The same function gives B as the scan length when dataset
generation calls `budget_fn(task=task, optimizer=update_class)`, so
the factory's B and the run's scan length agree. B is the most
generations B could buy at one evaluation each, so the scan is long
enough for the stop below to be reached.

**This breaks the old meaning.** A caller that passed an iteration
budget (`partial(fixed_budget, budget=N)`) now gets a budget of N
evaluations.

## The running count lives in the strategy state

The stop rule `UpdateClass.stop_fn` sees only the recent history
window and the optimizer state (`StopFunction` in l2co-optimizers
`core/typing.py`), and the window holds losses, not evaluations. So the
strategy's `State` gains `total_fevals`, an integer advanced in `tell`
by the cost of the generation.

That cost is billed by the active sub-optimizer's `SubOpt.bill_fevals`
(ADR 0010), not by its `popsize`. The bridge's `step_fn` writes the
same number into `OptHistory.fevals`, so the stop, the stored history,
the dump's final count and ERTD read one number. Before this, the
bridge billed `popsize` per generation, which charged an L-BFGS step 1
evaluation when its line search spent up to 21. **Recorded `fevals` of
L-BFGS generations in l2co runs change** (they go up).

The handshake generation stays a counted generation (ADR 0004): it adds
its cost to `total_fevals` like any other. The unevaluated initial
population costs nothing, as before.

## The stop

The factory passes `stop_fn = _StopAtEvaluations(B, user_stop)`:
`total_fevals >= B`, OR-ed with the step's own stopping criteria (which
`init_run_state` passes in as `stop_fn`; they are never replaced).
`UpdateClass.step` checks it before each generation, so the generation
that brings the count to B runs, and every later step is a skipped
entry (NaN losses, 0 evaluations). The run overshoots B by less than
one generation of the active optimizer, which is what the dump's
schedule check expects (ADR 0027 d, amendment). The rule is a frozen
dataclass, not a closure, so two builds with the same B compare equal
(`stop_fn` is a static field).

## The switch

Stage-1 fires in the first `ask` with `total_fevals >= round(t · B)`:
`t` is the stage-0 fraction of the budget, and the budget is now
evaluations. ADR 0026 listed this factory as one that computed its
switch generation from an iteration budget; that is gone.

The per-generation trajectory buffers (`State.traj_mean`,
`State.traj_min`) get the static length `round(t · B)`. Every
generation bills at least one evaluation, so no more generations than
that can run before the fire.

The public `l2co_population_strategy` keeps its generation switch
(`max_iterations`). Only the private `_l2co_population_strategy`, which
the factory calls, takes `max_evaluations` (exactly one of the two).

## What stage-1 sees at the fire

Stage-1 was trained on loss histories from the databank, which are
nearest-resampled onto `BUDGET_STEPS` (1000) points over the run
(`normalize_budget_grid` in `validation.py`) and cut at `t` by
`DataLoader.select_t`: the grid points labelled `<= t`, 501 for
`t = 0.5`. The axis is the run's history entries (iterations), which
for an optimizer with a fixed cost per step is proportional to
evaluations.

Under the evaluation switch the number of generations before the fire
depends on the stage-0 optimizer's popsize (25 for a popsize-6 ES with
`t · B = 150`, 150 for Adam). Feeding that raw buffer would give the
network an input whose length and spacing it never saw in training. So
the fire resamples the `n` generations recorded so far onto the same
number of points as the training cut, picking generation
`round(j / (K - 1) · (n - 1))` for point `j` of `K`: the training
construction applied to the part of the run that exists at the fire.
`K` comes from `BUDGET_STEPS` and `t` (no stage-1 checkpoint records
its input length). This keeps the input length fixed and close to
training, and costs `K` reads at the fire.

Rejected for now: feeding the raw per-generation buffer (variable length
and spacing, out of distribution) and resampling over evaluations rather
than generations (closer to the new budget, but not what the stored
checkpoints were trained on). Whether stage-1 should be retrained on
evaluation-axis histories is a separate experiment.

## Consequences

- Existing l2co checkpoints (`l2co_20260827_*`) were trained and
  evaluated under iteration budgets; their results are not comparable
  with runs under this ADR. Retraining is out of scope here.
- The scan is B long (up to 10⁵ steps) and most steps are skipped for
  a cheap-per-step sub-optimizer; the run's history buffer is B rows.
  Check memory and compile time at d = 1024 before a full run.
- l2co_experiments must inject the schedule budget function into the
  `"l2co"` row (`RunOptimization._l2co_registry_step`); with
  `evaluations_based_on_dimensionality` the factory raises `TypeError`.
