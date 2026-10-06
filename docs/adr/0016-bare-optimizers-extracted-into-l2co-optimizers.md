# The bare optimizers move to `l2co-optimizers`; l2co is the bridge to tasks

> **Superseded in part by ADR 0017:** the run loop (`init_` / `step` /
> `run_` / `batch_run_*`), `BatchState`, `Carry` and `HistoryState`'s
> buffer moved to `l2co-optimizers` after all. The rest stands.

ADR 0007 kept the optimizers in-repo, behind a hardened facade, and
named the condition for revisiting that: a split becomes worth it once it
is mechanical *and* someone wants the bare optimizers without the
meta-learning stack. Both now hold. 0007's own changes made the registry
know no meta-optimizer (`"l2co"` and `"rl2co"` self-register from their
side), and the optimizer configs are wanted as a reusable Hydra group, the
way `l2co-tasks` ships its task configs. So we **extract the bare
optimizers into a sibling package, `l2co-optimizers`**, mirroring
`l2co-tasks`. This supersedes the "stay in-repo" half of 0007. Its other
decisions stand: meta-optimizer self-registration, `sub_optimizer` as the
shared adapter layer, and the facade as the stable surface.

**Layering.** `l2co-optimizers` imports neither `l2co` nor `l2co-tasks`,
and l2co is the bridge between the two. Making that possible meant
answering 0007's three objections:

- **The shared `_src` substrate** moves *with* the optimizers wherever it
  is optimizer-side, and stays where it is not:
  - moves: `optimizer_schedule` (`OptimizationStep`), `stopping_criteria`,
    `opt_history`, `sampler`, the optimizer-contract aliases of `typing`,
    and the population loss evaluators of `tree_utils`;
  - stays: `BudgetFn`, `Carry`, `HistoryState`, and `tree_utils`'
    feature-scaling and IO helpers.
- **The registry mixing in a meta-optimizer** was resolved by 0007 itself.
  One residue remained: `OptimizationStep.name` hard-coded how `"l2co"` and
  `"rl2co"` steps are named. That becomes an extension point,
  `register_schedule_namer`, which meta-optimizer packages call next to
  `register_optimizer`. l2co registers `"l2co"` in `meta_optimizer.py`,
  and rl2co registers `"rl2co"`.
- **`Task`** reaches the optimizers only through `TaskLike`, a structural
  `Protocol` (`model`, `loss_fn`, `pass_rng`). `l2co_tasks.Task`
  satisfies it unchanged. The registry keyword stays `task=`, so bare and
  meta factories share one calling convention, and meta factories still
  receive a full `Task`. Dimensionality is derived from `model`, by the
  same formula as `Task.dimensionality`.

**What stays in l2co.** Everything that *runs* an optimizer on a task,
because it takes the dataset and the batch sampling: `init_` / `step` /
`run_` / `batch_run_*`, `RunState`, `BatchState`, `HistoryState` and
`RolloutWrapper`. So does everything that *chooses* between optimizers:
`strategy_wrapper`, `meta_optimizer` and the models. The strategy-facing
per-optimizer layer moves: `SubOpt`, the handshake policy (0009), state
transfer (0012–0014) and `menu_loss_and_grad` (0011). These describe
properties of individual bare optimizers, and they remain the single
implementation l2co and rl2co dispatch through.

**Compatibility.** Every public l2co path stays as a plain re-export of
the same objects, with no deprecation warnings:
- `l2co.optimizers`, `l2co.OptimizationStep`, `l2co.optimization`;
- `l2co.sampling`, `l2co.history.OptHistory`;
- `l2co.tree_utils.vmapped_*`, `l2co.naming`'s aliases.

Nothing stored references the moved modules by path: menus round-trip
through `OptimizationStep.to_dict` / `from_dict`, schedules through JSON,
and checkpoints through `eqx.tree_serialise_leaves`. So the private
`l2co._src.optimizers*` modules are deleted, not shimmed. Whether to
deprecate the public re-exports is left for later.

**Schedule names are databank keys**, and `OptimizationStep.hash` is
derived from `name`. `l2co-optimizers` locks the names and hashes of every
config it ships against a golden file recorded from l2co at `f677d8c`.
At the split, every list-shaped config in `l2co_experiments/conf/optimizers/`
was compared: 217 steps across 51 configs. Only the two `rl2co` steps
without an `alias` differed, and rl2co's namer restores both. **The l2co
change and rl2co's namer registration therefore have to land together.**

**Hydra configs.** The 14 bare-optimizer configs move to
`l2co_optimizers/conf/optimizers/`, resolved via
`pkg://l2co_optimizers.conf`. Meta-optimizer and factory-shaped configs,
and `conf/samplers/` (which interpolates `${tasks.bounded}`), stay in
`l2co_experiments`, in the same `optimizers` group; Hydra merges options
across search paths.

**Considered and rejected:**
- *Explicit arguments* (`model`, `loss_fn`, `pass_rng`) instead of a
  protocol, or an optimizer-owned `Objective` value type. Both split the
  registry's calling convention between bare and meta factories.
- *Carrying git history* with `filter-repo`. The split halves of `typing`
  and `tree_utils` could not follow, and the full history stays readable
  in l2co.
- *A minimal run loop* in `l2co-optimizers`. It needs `BatchState` and the
  dataset, which are task-side; tests use a test-only driver instead.
