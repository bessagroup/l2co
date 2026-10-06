# The run loop moves onto `UpdateClass`; l2co keeps the exports

ADR 0016 kept everything that *runs* an optimizer on a task in l2co —
`init_` / `step` / `run_` / `batch_run_*`, `BatchState`, `HistoryState`
— and rejected *a minimal run loop* in `l2co-optimizers` because "it
needs `BatchState` and the dataset, which are task-side". This
supersedes that half of 0016. **The run loop becomes methods of
`UpdateClass`**, and the machinery it needs moves with it. Every other
decision of 0016 stands: `TaskLike`, the schedule namers, the
compatibility re-exports, and neither sibling importing the other.

**Why the objection does not hold.** It was about ownership, not
imports. Read at the code level, the loop never touches a `Task`:

- the dataset arrives as a plain `dict[str, Array]` argument;
- `BatchState` is a generic index sampler over that dict, depending on
  nothing but jax and equinox;
- `Carry` is a tuple alias whose parts already lived in
  `l2co-optimizers` (`InputParameters`, `RecentHistory`) except
  `BatchState`.

The only genuinely task-side piece was `HistoryState`'s *exports*:
`to_xarray`, `to_xarray_realizations` and `to_dataloader` need xarray,
`DataLoader`, the ERT and `Task`. Those stay.

**What moves to `l2co-optimizers`:**
- `BatchState`, verbatim.
- `HistoryState`'s pure-JAX buffer: the seven fields and `init`,
  `reset`, `from_history`, `from_reduced`, `add`.
- `Carry`, into `_src/typing.py` next to the other loop aliases.
- The loop, as methods: `UpdateClass.init_state`, `step`, `batch_step`,
  `run`, `batch_run`, `batch_run_fused` and `batch_run_sequential`.
- `RunState` and its entry points `reset` / `batch_reset` / `run` /
  `batch_run` / `batch_evaluate`, plus `evaluate`, which `reset` needs.
  Once the loop moved, these are thin rewraps of it. The one task-side
  read, `RunState.init` taking `task.model`, `loaded_dataset` and
  `batch_size`, is a structural read like the factories' `TaskLike`. It
  gets its own protocol, `RunnableTaskLike` (`TaskLike` plus those two
  attributes), so that factories keep requiring only what they read.

**Dispatch becomes polymorphism.** `run_state.run` / `batch_run` no
longer inspect the optimizer:
- `RandomSearchUpdateClass` overrides `run` with the chunked one-shot
  runner. Its `batch_run` is `batch_run_sequential`, which maps
  `self.run` over realizations, so the separate
  `batch_run_randomsearch` disappears: it was that map already.
- `UpdateClass.batch_run` reads `sequential_realizations` itself. The
  flag stays a class-level marker rather than becoming an override,
  because it is a declarative property of a step function ("branches
  per realization") that ADR 0013 and rl2co ADR 0007 already cite, and
  `L2COUpdateClass` / `RL2COUpdateClass` keep setting it unchanged.

**What stays in l2co:**
- The exports, as free functions: `history_to_xarray`,
  `history_to_xarray_realizations`, `history_to_dataloader` (in
  `l2co.history` and at the package root). They were methods; they
  cannot stay methods of a class l2co no longer defines, and an l2co
  subclass would not help, because the moved `run` constructs the base
  class.
- `RolloutWrapper`, which bundles a `RunState` with a `Task`'s static
  structure, loss and sampler, and every meta-optimizer.

**Compatibility.**
- `l2co.BatchState`, `l2co.HistoryState`, `l2co.history`,
  `l2co.RunState`, `l2co.reset`, `l2co.run` and `l2co.evaluate`
  re-export the moved objects, signatures unchanged.
- **Breaking:** `l2co.init_` is removed rather than kept as a one-line
  delegate. Its only caller was rl2co's handshake, which now calls
  `update_class.init_state` itself.
- **Breaking:** `HistoryState.to_xarray` / `to_xarray_realizations` /
  `to_dataloader` are gone; call the free functions instead.
- The private `l2co._src.batching`, `l2co._src.update_class` and
  `l2co._src.run_state` modules are deleted, not shimmed, as 0016 did for the optimizer modules. Nothing
  stored references them by path: `HistoryState` and `BatchState` are
  never pickled, `RunState` is rebuilt per run rather than stored, and
  checkpoints go through
  `eqx.tree_serialise_leaves`, which is path-independent.
- `l2co-optimizers` gains `jax-tqdm` (the `verbose` progress bar). It
  must be released before the l2co that uses it; sibling dependencies
  are unpinned, so nothing enforces that order.

**Considered and rejected:**
- *Protocols for `BatchState` / `HistoryState`*, the `TaskLike` route.
  A protocol pays off when the concrete class drags in dependencies the
  package cannot take. `BatchState` drags none and has one
  implementation. `HistoryState` is *constructed* by the loop, not
  handed in, so a protocol would need a factory threaded through every
  method.
- *Returning the raw reduced statistics* and letting l2co wrap them in
  `HistoryState.from_reduced`. No call sites change, but the methods
  would return an untyped tuple and every caller would rebuild it.
- *Replacing `sequential_realizations` with `batch_run` overrides.* See
  above.
