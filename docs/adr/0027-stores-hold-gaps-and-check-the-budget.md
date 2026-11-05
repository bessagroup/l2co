---
status: accepted
---

# Stores hold gaps, check each cell's budget and record the run settings

The databank is the store of every optimizer's runs on every task that
the dataset paper scores. The paper's re-run changes three things about
what goes into it. The reasoning and the user's decisions are in the
dataset paper's protocol revision (`DATASET_paper/agent/
GRILLING-2026-11-03-protocol-revision.md`): §3.9, §5 item 2 and the
decisions c, d, e and g of §9. This record says how l2co implements
them, and which choices were left to the implementation.

## Decision

### c. The gap is a storage format

With |f\*| near 100, a float32 loss resolves only about 10⁻⁵, too coarse
for the 10⁻⁸ targets. So the store and the raw files hold f − f\*.

- **When.** `records._build_task_records` computes every dump-time
  number from the losses first: targets, `success` / `cost`, and the
  breach record. Only then does `records.to_gap` subtract f\* (the
  task's `global_min`) in float64 from `output_min`, `output_min_true`
  and `output_mean`. The result is stored as float32. `output_std` and
  the counters are not shifted. So the dump's scores are the same as
  before.
- **Tasks without a finite f\*** are stored and returned raw.
- **Where f\* is kept.** In a per-task float64 store variable,
  `f_star`, next to `target_value`. Each raw netCDF file carries `f_star`
  (float64) and `value_kind` (`"gap"` or `"raw"`) as attributes, so it
  describes itself outside l2co. The stamp has no `value_kind` field.
- **Readers add f\* back** in float64: `reader._load_one`, the helper
  behind every `load_databank*` call (overlay reads included, store by
  store), and `load_loss_raw`. They return losses, as float64.
  `load_databank` drops `f_star` from the DataLoader.
- **The protocol is bumped** to `dataset-paper-2026-11.1`: a stored
  number now means something else, which is the stamp's own rule (ADR
  0025). A store written in between is refused.

### d. Per-task evaluation budget, checked against every cell

`dump_databank`, `dump_runs` and `dump_databank_template` take
`evaluation_budget=`, a mapping from task hash to B. When it is given,
it must cover every task of the dump, and:

- **Every cell is checked** in the template step, before anything is
  written, deleted or marked, so a mismatch leaves the store as it was.
  `n` is the cell's
  number of iterations. Its cost per iteration is read from the
  `fevals` increments, the first one counted from 0. If every iteration
  costs the same λ, the cell must have n·λ ≤ B < (n+1)·λ. That is n = B
  for one evaluation per iteration. If the cost varies (L-BFGS,
  Nelder–Mead), the cell is budgeted at one evaluation per iteration
  (ADR 0026) and must have n = B.
- **B is stored** per task as the int variable `evaluation_budget`.
- **An append refuses**, before anything is written: a task whose
  incoming B differs from the stored one, budgeted runs into a store
  without the variable, and unbudgeted runs into a store with it.
- **On the parallel path**, the template step takes the keyword and
  hands B on in `DumpPlan.evaluation_budget`. So `dump_databank_task`
  needs no new argument.

### e. The cut at B happens in `calculate_ert`

`calculate_ert(..., evaluation_budget=B)` ignores every point where
`fevals` > B. The run is treated as if its history ended at its last
point within B. A target first reached after B counts as missed. A
missed run is charged at its last improvement within B, so at most B
evaluations. Its `iterations` cost is read at the same point. The
breach check still reads every point, and the raw files keep the full
history. `_ert_from_quantile` passes B through; `history_state`'s call
keeps the default.

### g. The stamp records the run settings

`run_settings=` on the same entry points is a plain JSON mapping, opaque
to l2co. Creation stores it in the stamp under `"run_settings"`. An
append compares it with the stamp for equality, as JSON, and refuses a
difference before anything is written. `check_run_settings(root,
dataset_name, run_settings)` is the same check, for a create step to
call before any cell runs. The refusal is `RunSettingsError`, a
subclass of `StoreProtocolError`.

### The simulated restart orders lose their dead tail (§5 item 2)

`draw_samples` returned a permutation of the runs followed by draws with
replacement up to `ERTD_N_PICKS` (50). The tail was never reached: the
expected runtime sums costs up to the first success, and a permutation
of all runs holds it if any run has one. `draw_samples` now returns the
permutation alone, and `ERTD_N_PICKS` is gone. No ERTD value changes:
the permutation is drawn first from the same generator, and a test
compares against the old draws. The resampled arrays shrink from 50
picks to the number of runs.

## Local choices

These were left open by the decisions and settled here:

- **Without `evaluation_budget=`**, nothing is checked, cut or stored,
  as before. l2co_experiments keeps working until its wiring PR passes
  B.
- **f\* is a store variable**, not parsed back from the `global_min`
  string in `task_labels`.
- **What "charged at most B" means:** the existing stopping rule (the
  step after the last improvement), applied to the history cut at B. A
  missed L-BFGS run can be charged a little under B. The `iterations`
  metric follows the same cut, so the two metrics stay consistent.
- **Fixed or variable cost is read from the data**, not from the
  optimizer. A variable-cost run whose increments happen to all equal
  some λ > 1 is judged as fixed-cost and refused. That is loud, not
  silent.
- **Where the cell check runs:** in the template step, which reads only
  the `iterations` and `fevals` counters of every cell (the template
  step's memory bound allows at most two whole histories). Reading every
  cell's counters costs one small read per raw file in one process;
  the alternative, checking in the per-task step, left a fresh store
  half-written, or an append marked mid-append, on a mismatch. The
  per-task step checks again, from the histories it loads anyway.
- **Policy schedules are checked like any other cell.** How
  meta-optimizer rows are budgeted is still undecided.
- **Run settings are compared as JSON**: key order and tuple-versus-list
  do not count. `check_run_settings` checks the protocol first, so an
  old store is refused for its protocol, and it passes when the store
  does not exist yet.

## Consequences

- Loaded loss arrays are float64, twice today's memory.
- Code that reads the raw files or the zarr store directly sees gaps.
  The direct raw readers in l2co_experiments and the paper move to
  l2co's readers in their own PRs.
- Every store written between ADR 0025 and this change is refused.

## Not done

- The `evaluation_budget` helper itself (its own PR, ADR 0026).
- l2co_experiments passing B and the run settings to the dump, and
  calling `check_run_settings` in its create step.
- Cutting the report helpers' final-loss scores at B (`dl.report()`
  still sees L-BFGS's progress after B).

## Rejected

- **Shifting before scoring:** it would change the dump's scores and
  move the breach check onto gaps.
- **A `value_kind` field in the stamp:** decided against (decision c);
  the per-file attributes and `f_star` carry it.
- **Deciding the cost model from the optimizer:** the dump has only the
  `OptimizationStep` recipe, not the built optimizer.
