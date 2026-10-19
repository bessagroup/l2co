---
status: proposed
---

# Per-variable bounds, and starts inside the box

ADR 0020 made the task the only source of bounds. It also limited them
to one `(lo, hi)` pair for every parameter, because that is all the
optimizer factories take: `task_bounds` refuses a `Box` whose bounds
differ between parameters. Until now no task needed more. The next
CUTEst task set does: bound-constrained CUTEst problems
(l2co-tasks, class B) put a bound on each variable, often on only some
of them, and often on one side only. 82 such problems have a fixed size
of at most 100 variables.

Two things go wrong on such a task today:

- **It can't run at all.** `task_bounds` raises, so `init_run_state` and
  `RolloutWrapper.init` refuse it.
- **Its first evaluations can lie outside the box.** l2co-optimizers'
  `reset` (`core/run_state.py`) draws the starting population with
  `sampler(key, best_params, popsize)` and *evaluates* it before any
  optimizer has clipped anything. This already happens on the one boxed
  task we have: `gaussian_meta`'s databank runs used the `normal`
  sampler (N(0, 1)), and most of its draws fall outside `Box(0, 1)`. On
  CUTEst, 11 of those 82 problems start outside their own box, by up to
  5 units, and `relative_normal` spreads its draws around that start.

**Decision.** A box may differ per parameter, and every sampled start is
projected onto the task's box before it is evaluated.

## Per-variable bounds

- **`task_bounds` returns the box per parameter when it has to.**
  - No constraints: `(None, None)`, as today.
  - A box with the same value on every parameter: today's scalar pair,
    with an infinite side as `None`. `gaussian_meta` and every other
    existing task build exactly the optimizers they build today.
  - Any other box: the pair `(box.lower, box.upper)` as the `Box`
    stores them, as pytrees that match the model's floating-point
    parameters, with an open side as `±inf`. A `Box` is already stored
    expanded like this (l2co-tasks ADR 0002), so nothing is computed.
  - An `Inequality` or `Equality` still raises. Enforcing those needs
    feasibility-aware scoring, which is a separate decision.
- **Every `bounded=` keyword accepts both forms.** That covers the bare
  factories in l2co-optimizers, which keep the factory contract
  `factory(*, task, opt_hash, bounded, stop_fn, **hyperparameters)` from
  ADR 0020, and the two meta-optimizers, l2co's `L2COUpdateClass` and
  rl2co's `RL2COUpdateClass`. Each clip site today is
  `jax.tree.map(lambda p: jnp.clip(p, *bounded), params)`. They all move
  to one helper in l2co-optimizers that clips to either form. A
  per-parameter bound broadcasts over leading population and
  realization axes, so a population of shape `(popsize, n)` clips
  against bounds of shape `(n,)`. The scipy and IPOPT driver already
  expands the pair to per-variable arrays (`_np_bounds`); a pytree bound
  is flattened with `ravel_pytree`, in the same order as the parameters.
  How l2co-optimizers does this is its own ADR. This one fixes only what
  l2co passes.

## Starts inside the box

- **One helper projects a sampler's draws.** `task_sampler(task,
  sampler)` in `_src/task_bridge.py` returns a sampler whose every draw
  is clipped into the task's box (to the per-parameter bounds, or the
  scalar pair). For a task without a box it returns `sampler` itself,
  unchanged, so unboxed tasks trace and run exactly as today.
- **It is applied where a task meets a sampler.** `RolloutWrapper.init`
  wraps the sampler it is given, so `batch_evaluate` and everything that
  runs through it start inside the box. rl2co's `EnvState` wraps the
  sampler it passes to `l2co.reset`. Those are the two paths into
  l2co-optimizers' `reset`.
- **Draws pile up on the faces of the box.** A coordinate drawn outside
  lands on the face. This is accepted, as it is for random search
  (l2co-optimizers ADR 0008).
- **The task's own start is not projected here.** l2co-tasks refuses a
  model outside its own box (ADR 0002, rule 6). A task with a prescribed
  start outside its bounds projects it when the task is built; for
  CUTEst that is l2co-tasks' decision. Member 0 of `relative_normal` is
  exactly that start, so it is unchanged by the clip.

This is not the option ADR 0020 rejected as "samplers reading the task's
box". That option re-bound a sampler's own range keywords
(`lower_bound`, `upper_bound`) by inspecting its signature, before any
task existed. Here the sampler is left alone and its output is clipped
afterwards, at the point where the task is known. That needs no
knowledge of the sampler's keywords.

l2co-tasks ADR 0002 says starting points drawn by a sampler "are the
sampler's responsibility". From now on l2co takes that responsibility on
every path through `reset`.

## What changes for existing results

- **Unboxed tasks:** nothing. `task_bounds` returns `(None, None)` and
  `task_sampler` returns the sampler itself.
- **`gaussian_meta`** (`Box(0, 1)`): its optimizers get the same scalar
  pair. Its starts change: draws from a sampler wider than the box,
  such as `normal`, now start on the box's faces instead of being
  evaluated outside it. No stored result changes. The 100 raw loss files
  in `databank/gaussian_meta/` were all written between 21 and 28 Aug
  2026. The task got its box on 7 Oct, which gave it new keys, so those
  files belong to the unboxed task (l2co-tasks ADR 0002).

## What stays open

- **Switching menus run without a box.** The optimizer parts
  (`optimizer_parts`) are still built with `bounded=(None, None)`
  (l2co-optimizers ADR 0008), so an rl2co switching policy on a boxed
  task runs its sub-optimizers unbounded. The mid-run resamples a switch
  makes (`population_best_one` and the handshakes) are not projected
  either. Both wait until a switching study needs a boxed task.
- **The public strategy paths still refuse every constraint.** That is
  `l2co_population_strategy` and rl2co's `rl2co_optimizer` (ADR 0020).

## Rollout

l2co, rl2co and l2co_experiments install their siblings editable, so a
change on disk is live for the next process. The stages are each safe on
their own, as in ADR 0020:

1. **l2co-optimizers.** Add the clip helper; every factory accepts both
   forms. Nothing passes a pytree yet.
2. **l2co.** `task_bounds` returns per-parameter bounds,
   `task_sampler` exists and `RolloutWrapper.init` uses it, and
   `L2COUpdateClass` clips with the helper. Every task that exists has
   no box or a uniform one, so only `gaussian_meta`'s starts change.
3. **rl2co.** `RL2COUpdateClass` clips with the helper, and `EnvState`
   wraps its sampler. Between stages 2 and 3, a task with a
   per-parameter box would reach `RL2COUpdateClass` and fail loudly at
   its clip. No such task exists until stage 4.
4. **l2co-tasks.** Bound-constrained CUTEst tasks: the first tasks with a
   per-parameter box.

Before each stage, check `squeue` and the equeue lanes so nothing is
rewired under a running job.

## Considered and rejected

- **Rescaling each bounded variable to `[0, 1]` inside the loss.** It
  fails on one-sided and infinite bounds, and it breaks l2co-tasks ADR
  0004: a CUTEst task stays in the problem's own coordinates.
- **Clipping inside the task's loss** (`loss(clip(x))`). Every optimizer
  would only ever clip. The 12 registry entries that keep their own
  state inside a box (l2co-optimizers ADR 0008) would lose that.
- **Always passing pytrees, even for a uniform box.** It would change
  every factory call for `gaussian_meta` for no gain. The scalar pair
  keeps today's traces and behaviour wherever it is enough.
- **Projecting inside l2co-optimizers' `reset`.** `reset` knows no task,
  and an `UpdateClass` doesn't carry its bounds; every update class
  would need a new field.
- **Leaving sampled starts to the samplers.** That is today's behaviour,
  and on `gaussian_meta` it already evaluates points outside the box.
