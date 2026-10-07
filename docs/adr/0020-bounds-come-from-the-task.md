---
status: accepted
---

# Bounds come from the task

Box bounds used to reach optimizers from the experiment config.
`tasks.bounded`, one `(lo, hi)` pair for a whole task set, was passed
alongside the task (`init_run_state(optimizer, task, bounded, key)`,
`RolloutWrapper.init`, `RunOptimization`, rl2co's `EnvState.init`, and
others) and on to every optimizer factory, which clips to it. Because
the bounds sat outside the task, clipping could change the problem
without changing the task's hash, and clipped and unclipped results
would share databank keys. In practice every task config left the pair
empty.

l2co-tasks ADR 0002 makes constraints part of the `Task`. This is l2co's
side of it: **the task is the only source of bounds, and every path that
takes a task refuses what it can't enforce.** It extends the bridge from
ADR 0018.

**One helper.** `task_bounds(task)` in `_src/task_bridge.py` is the only
way bounds leave a task:
- No constraints: `(None, None)`, exactly what today's empty config
  gives.
- A `Box` with the same value on every parameter: today's `(lo, hi)`
  pair, with `±inf` turned into `None`.
- An `Inequality`, an `Equality`, or a `Box` whose values differ between
  parameters: raises. Nothing downstream can enforce these yet.
  Optimizers take one pair for all parameters, and "best so far" ignores
  feasibility.

Its companion, `refuse_constraints(task, where)`, is for a path with no
way to enforce constraints: it raises if the task has any, a box
included.

**Where they are called.** Only two places build optimizers from a
task: `init_run_state` and `RolloutWrapper.init`. Both fill the
factories' `bounded=` keyword from `task_bounds`. Everything else
reaches optimizers through them: l2co's `RunOptimization` through
`RolloutWrapper.init`, and every rl2co entry point (`EnvState.init`,
both `RunOptimization` classes, the on-policy trainer) through
`init_run_state`. Paths with no bounds channel:
- l2co's public `l2co_population_strategy` refuses a constrained task.
  The l2co meta-optimizer builds on an unchecked private version,
  because it clips every population it is handed.
- rl2co's `rl2co_optimizer` refuses a constrained task.
- rl2co's `rl2co_population_strategy` is clipped by its rl2co caller
  (`update_class_wrapper`). l2co_experiments' `benchmark_rollout`
  imports it privately and doesn't clip, so that call site refuses
  constrained tasks once l2co_experiments is updated.

A path moves from "refuses" to "enforces" when something needs it.

The internal factory contract,
`factory(*, task, opt_hash, bounded, stop_fn, **hyperparameters)`,
keeps `bounded`: bare optimizers in `l2co-optimizers` know no task and
still take the pair. Tests of an optimizer's own bound handling (SHADE)
call factories directly with a pair on an unconstrained task.

**`tasks.bounded` is retired.** It goes from l2co-tasks' 36 task
configs, from `base_rl2co_train.yaml` and `test.yaml` in
l2co_experiments, and from the code that reads it: `benchmark_rollout`,
`static_optimizer`, the online multitask evaluation, the `evaluate_rl`
and `agentic_smoke` experiments, and rl2co's `EnvState.from_config`.
Archived configs and experiments (`conf/_archive`,
`experiments/_archive`) are left as they were. The `random` and `grid`
samplers keep their own
`lower_bound` / `upper_bound` (default 0 and 1): a range for drawing
starting points is a sampling choice, not a constraint. No config or
command-line override sets the pair today, so nothing in use is lost.

**Rollout: transitional, in three steps.** l2co and rl2co are editable
installs, so a change on disk is live for the next run from any
checkout, including l2co_experiments' (which also held unrelated
uncommitted work when this was decided). A breaking signature change in
l2co would crash every run started from a not-yet-updated
l2co_experiments. So:
1. **l2co and rl2co.** Bounds come only from the task. The entry points
   keep their `bounded` parameter but accept only "no bounds" (`None` or
   `(None, None)`); anything else raises and points here. Every existing
   caller passes the empty pair, so nothing breaks.
2. **l2co_experiments.** Stop reading `tasks.bounded` and stop passing
   `bounded`; give the sampler configs their own range; add the refusal
   at `benchmark_rollout`'s strategy call; drop rl2co's
   `EnvState.from_config` read. `EnvState.init` still requires the
   argument, so the two `env_state` config sections pass an explicit
   `bounded: null` until step 3. l2co-tasks' configs then drop the key,
   and its slow floor test gives a unit-domain task a `Box(0, 1)` instead
   of passing `bounded=(0, 1)`.
3. **Remove the transitional parameters.** This is the breaking step:
   `init_run_state(optimizer, task, key)`, `RolloutWrapper.init`,
   `RunOptimization`, and in rl2co `EnvState.init`, `RunOptimization`
   and the on-policy trainer lose `bounded`, along with the two
   `bounded: null` config lines. agentic-l2co's runner passes it too and
   changes with it.

All of this is stage 2 of l2co-tasks ADR 0002: it lands before any task
carries a box (stage 3).

**Considered and rejected:**
- *Checking only in `init_run_state`.* `RolloutWrapper.init` and the
  public strategy would silently ignore a box.
- *Making every path enforce the box now.* Work across three repos for
  paths that `gaussian_meta`, the only boxed task, may never take.
- *Keeping `tasks.bounded` as an override or a fallback.* Clipping
  changes the problem but not the hash, so results that differ only in
  clipping would be pooled under one key.
- *Removing the parameters in one step.* See the rollout above.
- *Samplers reading the task's box.* Samplers are built from config
  before any task exists; the bridge would have to re-bind sampler
  keywords by inspecting signatures.
