# The task stops at l2co

ADR 0016 let a task reach `l2co-optimizers` as `TaskLike`, a structural
protocol (`model`, `loss_fn`, `pass_rng`) that `l2co_tasks.Task`
satisfies without either package importing the other. ADR 0017 widened
it to `RunnableTaskLike` (adding `loaded_dataset` and `batch_size`) for
`RunState.init`. This supersedes both protocols. **`l2co-optimizers` no
longer has any notion of a task.** It never sees a `Task`, and
`l2co-tasks` never sees an `OptimizationStep`. l2co is the only place
where the two meet.

**Why a protocol was not enough.** `TaskLike` removed the *import* but
kept the *concept*. A bare optimizer still took "a task" and read three
attributes off it. Every meta-optimizer registered into the same
registry, under the same `task=` keyword, but required the whole
`Task` (`budget_fn(task=task)`, `task.tag`, `task.global_min`). So the
one `task=` channel did two jobs. The bare package's contract was
"whatever object you pass, as long as these attributes are there",
while its registry actually carried full tasks for factories that
weren't its own.

**What changes in `l2co-optimizers`:**
- `TaskLike` and `RunnableTaskLike` are deleted, with no deprecation
  alias.
- Every bare factory takes `*, model, loss_fn, pass_rng` (required and
  keyword-only) in place of `task`. `pass_rng` has no default: a
  forgotten flag would silently run a stochastic loss with no key.
- `PopSize = int | Callable[[int], int]`: a popsize callable receives
  the dimensionality, which every built-in rule needs and none goes
  beyond.
- `resolve_popsize(opt_step, *, model, loss_fn, pass_rng)` and
  `optstep_to_subopt(...)`: the same keywords, because `resolve_popsize`
  builds a factory to read the popsize from. That build stays the one
  source of truth for popsize.
- `RunState.init(update_class, *, model, dataset, batch_size, key)`:
  it takes an already-built `UpdateClass`, so it no longer looks up the
  registry and no longer takes `bounded`.
- `register_optimizer` is for bare optimizers only.
- A test rejects any public callable with a `task` parameter. Ruff
  already rejects an `l2co_tasks` import.

**What l2co gains** (`_src/task_bridge.py`):
- `meta_optimizers` and `register_optimizer`, the registry for
  optimizers that need a whole task. `"l2co"` registers here, and so
  does rl2co's `"rl2co"`. Registering a name that is already a bare
  optimizer raises.
- `optimizer_mapping(name)`, the Task-level lookup. A meta name returns
  its factory. Any other name falls through to the bare registry,
  wrapped so it is called with `task=` and unpacks `task.model` /
  `task.loss_fn` / `task.pass_rng`. The wrapper is cached per name (one
  object per name) and looks the bare factory up per call (so
  re-registering a bare optimizer still takes effect).
- `init_run_state(optimizer, task, bounded, key)`, which resolves and
  then calls `RunState.init` with the task's model and data.

The names `l2co.optimizer_mapping` and `l2co.register_optimizer` are
kept and now mean the Task-level versions. `l2co.optimizers` stops
re-exporting the bare `optimizer_mapping` / `optimizers` /
`register_optimizer`, so inside l2co each name means one thing. Call
sites that already wrote `l2co.optimizer_mapping(name)(task=...)` keep
working unchanged.

**Enforcement.** `l2co-tasks` gets the mirror ruff ban (`l2co`,
`l2co_optimizers`). The real-`Task` battery that tested the `TaskLike`
bet in `l2co-optimizers` moves here as
`tests/integration/test_task_bridge.py`. It now runs every bare factory
through the bridge.

**Consequences:**
- **Breaking** for any caller of a bare factory with `task=`, of
  `RunState.init(optimizer=, task=, ...)`, of `resolve_popsize` /
  `optstep_to_subopt` with a task, and of a popsize callable with a
  task. Callers with a `Task` should go through l2co
  (`optimizer_mapping`, `init_run_state`). Callers without one pass the
  keywords.
- Meta-optimizers register with `l2co.register_optimizer`. rl2co's two
  registrations change their import.
- The two packages now share a name with different meanings:
  `l2co_optimizers.optimizer_mapping` takes the three keywords and
  `l2co.optimizer_mapping` takes `task=`. That is the layering this ADR
  sets up, and each package only exposes its own version.

**Considered and rejected:**
- *One registry with an opaque pass-through* (`context=` or `**extra`
  that only meta-optimizers read). The bare package would still carry a
  task slot under a different name.
- *Meta-optimizers close over the task at the call site*
  (`partial(factory, task=task)`). A global name → factory map can't
  hold per-task entries.
- *Popsize resolved from the factory's signature default* instead of by
  building the factory. That would add a second rule that can drift
  from what the factory actually does.
- *Keeping the `l2co.RunState.init(optimizer=, task=)` spelling* via an
  l2co subclass. It would override the parent's signature (an LSP
  violation) and leave two classes called `RunState`. Under the free
  function, old call sites fail loudly with a `TypeError`.
