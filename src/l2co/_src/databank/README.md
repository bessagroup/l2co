# Databank (consolidated-zarr)

A lazy-loadable databank that stores a whole dataset in **one zarr store** plus
a sidecar of raw loss histories. It exists to make reads cheap at scale
(hundreds of tasks × tens of optimizers), especially on networked HPC
filesystems where opening thousands of HDF5 files is expensive.

```
<root>/<dataset_name>.zarr/                       # consolidated store
    loss_history (interp), performance, task, optimizer
    l2co_stamp.json                               # protocol stamp (ADR 0025)
<root>/<dataset_name>/loss_history_raw/           # per-pair raw netCDF
    loss_<taskID>_<optimizer_hash>.nc
<root>/<dataset_name>/tasksets/<name>.txt         # named task selections
<root>/<dataset_name>/policy_artifacts/<schedule>/  # archived policy artifacts
    <checkpoint files> + <stem>.manifest.json + provenance.json
<root>/<dataset_name>/global_min_breaches.json    # runs below global_min
```

`<root>` is the databank base directory supplied by the caller (the application
resolves it from its own config, e.g. a per-cluster `databank_root` key — the
library holds no machine paths). `<taskID>` is
`l2co_tasks.Task.hash` (a 16-hex string); `<optimizer_hash>` is
`l2co.OptimizationStep.hash` (`int(sha256(...)[:6], 16)`) for static
optimizers and `policy_identity_hash` (weights hash + evaluation settings,
never file paths) for policy schedules. The **human-readable
optimizer name** (`OptimizationStep.name`) is the `schedule` coordinate of the
store; the integer hash is used only in the raw sidecar filenames.

## Why one store, chunked per pair

The store is chunked **one chunk per `(taskID, schedule)`**: a subset read is
`xr.open_zarr(store).sel(taskID=..., schedule=...)`, lazily fetching only the
selected chunks. A missing `(task, optimizer)` pair is simply an unwritten chunk
that reads back as `NaN` (zero bytes on disk). zarr v3 **sharding** packs each
cell's chunks into one file, so the inode count stays modest rather than
exploding into one-file-per-record.

## Storage decisions

- **All numerical variables are `float32` on disk.** A missing cell must be
  `NaN`, which only floats hold, so `fevals` / `iterations` / `update_step` /
  `cost` / `success` are stored (and returned) as `float`. Cast to int/bool
  downstream if needed — the reader does not. Storage is `float32` to halve disk
  usage; `float64` is reserved for the upstream optimization runs that *generate*
  the data (numerical stability), not for storing the results, which feed the
  `float32` JAX training path. The `loss_history_raw/` sidecar applies the same
  downcast to its **float** variables only, leaving its integer counters intact.
- **Fixed interior dims.** `realization`, `target`, `feature`, `output_dim` and
  `metric` are constant across `(taskID, schedule)` per dataset; the writer
  asserts the `feature` axis is consistent. Appends only ever extend `taskID` /
  `schedule`.
- **`loss_history` is split in two.** The store holds the loss history
  **interpolated** to a fixed `BUDGET_STEPS` grid (uniform, dense, read with no
  per-pair resampling). The original variable-length histories are kept under
  `loss_history_raw/` as the source of truth — regenerate the interpolated store
  at a different `BUDGET_STEPS` from there (see `load_loss_raw`).

## Per-type schemas (in the store)

| Array | Dims | Notes |
|---|---|---|
| `output_min` / `output_mean` / `output_std` / `fevals` / `iterations` / `update_step` | `(taskID, schedule, realization, output_dim, budget)` | `budget` = `BUDGET_STEPS`, canonical grid. `output_min` and `output_mean` hold the **gap** `f - f*` (`docs/adr/0027`; see "Gaps, not losses" below); `output_std` and the counters are unshifted |
| `output_min_true` | `(taskID, schedule, realization, output_dim, budget)` | **Conditional** (`docs/adr/0024`): present only in a store holding a noisy task (one with a `loss_fn_true`). The lowest *noise-free* loss per step, which noisy tasks are scored on (`output_min` stays the observed one). A noise-free task's cells of it are never written: no chunk file, `NaN` on read. Stored as a gap, like `output_min` |
| `success` | `(taskID, schedule, realization, output_dim, target)` | float (0/1, NaN if missing) |
| `cost` | `(taskID, schedule, realization, output_dim, target, metric)` | float |
| `task_labels` | `(taskID, feature)` | str |
| `target_value` | `(taskID, target)` | float |
| `f_star` | `(taskID,)` | float64: the f* (`global_min`) the task's losses are stored relative to; `NaN` for a task without a finite f*, stored raw (`docs/adr/0027`; dropped by the loader) |
| `evaluation_budget` | `(taskID,)` | int: the task's evaluation budget B its cells were checked against. **Conditional**: only in a store dumped with `evaluation_budget=` (`docs/adr/0027`; dropped by the loader) |
| `optimizationstep` | `(schedule,)` | str (JSON recipe; dropped by the loader) |
| `provenance` | `(schedule,)` | str (JSON policy-artifact record; `""` for static optimizers; dropped by the loader) |
| `in_training` | `(taskID, schedule)` | float (1/0 for policy schedules, NaN for static; dropped by the loader) |

`metric` coord is `["fevals", "iterations"]`. `target` is an integer index; the
target *values* live in `target_value`. `task_labels` carries the task tag plus
`global_min` (one `feature` per key).

## Policy artifacts & provenance

Learned optimizers (rl2co / l2co policies) enter the store as regular
schedules — **format-identical arrays** — identified by a *policy artifact*:
one checkpoint of one training run. The contract (design record:
`docs/adr/0001-policy-artifact-provenance-manifest.md`):

- **Training** writes a *manifest* next to every checkpoint
  (`mint_run_id` + `write_manifest`, or `bump_manifest` for in-place
  overwriting trainers): run id, step, full resolved config, git SHAs,
  weights hash, and the **explicit training task-hash list** (task hashes
  are cloudpickle-derived and not stable across versions — record them
  eagerly). A checkpoint without a manifest is **not admissible**; there is
  no backfill and no bypass flag.
- **Evaluation** loads the manifest (`load_manifest` — fails fast on a
  missing manifest or overwritten weights), assembles a record
  (`provenance_record`, with a `components` map for composites like the
  L2CO stage-0 + stage-1 pair) and attaches it to the loss-history dataset
  as the `l2co.naming.PROVENANCE` attribute.
- **The writer** stores the record in the `provenance` variable (a policy
  schedule without one is refused — the backstop), materializes
  `in_training` from the record's task hashes, archives the artifact files
  listed under `evaluation.artifacts` into `policy_artifacts/<schedule>/`
  (training dirs live on purged scratch; the dataset stays
  self-sufficient), and enforces **alias consistency**: a schedule name may
  only ever refer to one artifact (compared by combined weights hash).
- **Switching sequence**: no extra variable — `update_step` already stamps
  the active optimizer's `OptimizationStep.hash` per iteration (static
  optimizers are constant at their own hash).

Read back with `load_provenance` (parsed records per policy schedule) and
`held_out_task_ids` (tasks a policy was *not* trained on — pass as
`task_ids` for held-out-only ERTD comparisons).

## Writing

`dump_databank` reuses the shared f3dasm preamble and ERT pipeline
(`records._iter_task_runs`, `records._build_task_records`). Per pair it writes
the raw history to `loss_history_raw/` and region-writes the interpolated history
+ performance into the store.

- **Creation** (`mode="error"` / `"replace"`) streams one `(task, optimizer)`
  cell at a time into a `compute=False` template, so the full dense array is
  never held in memory. All cell writes go through zarr array handles opened
  once per dump (`_open_big_arrays`) — a per-cell `Dataset.to_zarr` call would
  re-open the store and re-read every array's metadata, which dominates the
  cost of the small per-pair writes.
- **Append** (`mode="append"`) **grows the existing store in place** rather
  than rebuilding it. `zarr.Array.resize` on these sharded arrays is
  metadata-only — every untouched old cell keeps its file key and its bytes,
  and a new `(taskID, schedule)` position simply reads back as the fill value
  — so the append costs O(new cells), not O(existing store size). The
  `taskID` / `schedule` coordinate arrays are the one exception (their
  fixed-length string dtype would truncate a longer new entry), so they are
  deleted and rewritten fresh at the union values. Append reuses the
  existing budget grid and each existing task's stored ERT `target_value`
  (so appended optimizers stay comparable), and requires the incoming runs'
  variable set and interior shapes (e.g. `realization` count) to match the
  store's exactly — a mismatch raises rather than silently growing an
  interior axis, which would need every stored cell re-encoded.

  Growing in place means a crashed append can leave the store mid-mutation.
  Before touching anything, the template step writes an
  `append_in_progress.json` marker *inside* the store (every reader routes
  through a guard that refuses a marked store — see Reading below) and
  stages a hard-linked snapshot of every path the grow is about to mutate
  under a sibling `<dataset_name>.zarr.rollback/` directory — root/array
  metadata, the two coordinate arrays, the five small arrays, and the chunk
  file of any incoming pair that already has a stored cell (a re-run pair
  overwrites it). The snapshot costs disk proportional to the *incoming*
  runs, not to the store. Finalize removes the marker and snapshot on
  success; a leftover snapshot found by the *next* append is auto-restored
  before that append proceeds, so re-submitting a dump chain after a crash
  always converges.

Writes that touch distinct `(taskID, schedule)` cells are chunk-disjoint and
safe to parallelise; resizing and `consolidate_metadata` are single-writer.

### Gaps, not losses

The store and the raw files hold `f - f*`, where f* is the task's
`global_min` (`docs/adr/0027`, the spec's decision c). With |f*| near 100, a
float32 loss resolves only about `1e-5`, too coarse for the `1e-8` targets; a
float32 gap is precise relative to the gap itself. The per-task step
(`records._build_task_records`) first computes every dump-time number from
the losses — targets, `success` / `cost`, the breach record — and only then
subtracts f* in float64 from `output_min`, `output_min_true` and
`output_mean` (`records.to_gap`). `output_std` and the counters are not
shifted. A task without a finite f* is stored raw. f* is kept per task as the
float64 `f_star` variable, and each raw netCDF file carries two attributes,
so it describes itself outside l2co:

| Attribute | Value |
|---|---|
| `f_star` | float64 f*, `NaN` when the task has none |
| `value_kind` | `"gap"`, or `"raw"` for a task without a finite f* |

The readers add f* back in float64 (see Reading), so callers keep getting
losses. Code that reads the raw files or the zarr store directly sees gaps.

### The evaluation budget

`evaluation_budget=` on `dump_databank`, `dump_runs` and
`dump_databank_template` gives B per task, as a mapping from task hash to B
(`docs/adr/0027`, the spec's decision d). When it is given, it must cover
every task of the dump, and:

- **every cell is checked** in the per-task step. `n` is the cell's number of
  iterations and its cost per iteration comes from the `fevals` increments.
  A cell whose every iteration costs the same `λ` must have
  `n·λ ≤ B < (n+1)·λ` (`n = B` at one evaluation per iteration); a cell
  whose cost varies (L-BFGS, Nelder-Mead) is budgeted at one evaluation per
  iteration and must have `n = B`. A mismatch raises `ValueError` naming the
  task and schedules.
- **the scores are cut at B** (`calculate_ert(..., evaluation_budget=B)`,
  decision e): points with `fevals > B` are ignored, as if each run ended at
  its last point within B. A target first reached after B counts as missed,
  and a missed run is charged at its last improvement within B, so at most B.
  The breach record still reads every point, and the raw files keep the full
  history.
- **B is stored** as the `evaluation_budget` variable.

An append refuses, before anything is written, a task whose incoming B
differs from the stored one, budgeted runs into a store without the
variable, and unbudgeted runs into a store with it. On the parallel path the
template step takes `evaluation_budget=` and hands it on in
`DumpPlan.evaluation_budget`, so `dump_databank_task` needs no new argument.
Without `evaluation_budget=` nothing is checked, cut or stored, as before
(l2co_experiments passes B once its wiring lands).

### Run settings

`run_settings=` on the same entry points is a plain JSON mapping of the
settings that define a cell (budget rule, sampler and its parameters, seed,
`n_realizations`; decision g). l2co does not interpret it. Creation stores it
in the stamp under `"run_settings"`; an append compares it with the stamp for
equality (as JSON, so key order and tuple-vs-list do not matter) and refuses
a difference with `RunSettingsError`, a `StoreProtocolError`, before anything
is written. `check_run_settings(root, dataset_name, run_settings)` makes the
same check for a create step to call before any cell runs; it first checks
the protocol, and passes when the store does not exist yet. When neither the
store nor the caller has settings, nothing changes.

### The protocol stamp

A store's numbers depend on the run protocol that made them (budget, starting
points, noise-free scoring, task hashes), and stores of two protocols can
share a dataset name under different roots. So creation writes
`l2co_stamp.json` inside the new store (`docs/adr/0025`):

```json
{"protocol": "dataset-paper-2026-11.1", "l2co_version": "...", "created": "...",
 "run_settings": {"...": "..."}}
```

`protocol` is `stamp.DATABANK_PROTOCOL`, set by hand and bumped in any change
after which new numbers are not comparable with stored ones. It was bumped
from `dataset-paper-2026-11` to `dataset-paper-2026-11.1` when stores started
holding gaps (`docs/adr/0027`), so a store written in between is refused.
`run_settings` is there only when the store was created with them (see Run
settings above). It is a plain
file, not a zarr root attribute, because every `to_zarr(mode="a")` the writer
makes replaces the root attributes; append never touches the file. Before
anything else, `mode="append"` and `mode="replace"` refuse a store whose stamp
is missing or names another protocol (`StoreProtocolError`): append writes
nothing, and replace deletes nothing, so an old store is only ever removed by
hand. A store without a stamp was written by l2co 1.8.x or earlier.

### Runs below `global_min`

A task's `global_min` is sometimes set empirically (CUTEst), so a run can
reach a lower loss. The dump scores that run as if it reached `global_min`:
the quality value (`loss - global_min`) is floored at 0, so it solves every
target, and the read-side log scales stay finite. Finalize then writes
`global_min_breaches.json`, so the value can be re-tuned:

```json
{
  "<taskID>": {
    "task": {"fn_name": "MEXHAT", "dimensionality": 2},
    "global_min": -0.04,
    "best": -0.0402,
    "margin": 0.0002,
    "schedules": {
      "<schedule>": {"best": -0.0402, "margin": 0.0002, "realization": 3}
    }
  }
}
```

`best` is the lowest loss over every run of the task, `margin` how far it
sits below `global_min`. Losses come from the raw histories, at the
precision the run wrote, not from the float32 store, and before the shift
to gaps. With an evaluation budget, the check still reads points past B: a
point below f* means f* is wrong, whatever the budget. A loss counts only
when it is more than `1e-6 * max(1, |global_min|)` below `global_min`
(`l2co._src.performance.GLOBAL_MIN_BREACH_RTOL`, the CUTEst floor test's
rule), so float noise at an exact optimum is not reported. Every dump
writes the file, so an empty `{}` means nothing breached. An append merges
into it: a schedule it re-runs replaces that schedule's entry, or drops it.

## Reading

`load_databank_dataset` opens the consolidated store, `.sel`s the requested
tasks/optimizers (optimizers given as `OptimizationStep`, schedule-name strings,
or integer hashes), and returns the lazy selection — no per-pair resampling,
since `loss_history` is already on the `BUDGET_STEPS` grid. It adds each
task's `f_star` back to the stored gaps in float64 (`reader._add_f_star`, part
of the lazy graph), so `output_min`, `output_min_true` and `output_mean` are
losses, as float64; overlay reads get the same, store by store. `load_loss_raw`
does the same from the raw file's `f_star` attribute and drops `value_kind`.
`load_databank` drops the per-schedule metadata variables
(`optimizationstep`, `provenance`, `in_training`) and the per-task `f_star` /
`evaluation_budget`, and returns a lazy `DataLoader` (dask reads only the
chunks each operation touches; call `.load()` to force it all into memory).
`load_databank_performance` keeps only the performance variables, so the large
loss chunks are never read (use it for ERTD reporting). `load_loss_raw` reaches
a single raw history (optionally resampled). `populated_pairs` returns the set
of `(task hash, schedule name)` cells already written (a non-null `success`
cell) — empty when the store is absent — so a caller can decide which
`(task, optimizer)` runs are still missing before launching them (then
`dump_databank(..., mode="append")` adds just those). `populated_iterations`
is the budget-aware variant: it returns `{(task hash, schedule name):
max_iterations}` (the `iterations` counter's max, which equals the stored run's
`n_iterations`), so a caller can also re-run a cell whose proposed budget
*exceeds* the stored one — `mode="append"` then overwrites that cell in place.

Every reader entry point that opens the zarr store routes through one choke
point (`reader._open_store`). It first refuses a store whose protocol stamp is
missing or names another protocol (`StoreProtocolError`, see Writing above),
so numbers from two protocols are never read together and a launch pointed at
the wrong root stops at its first read; read such a store with the l2co that
wrote it (1.8.x for a store without a stamp). It then refuses a store an
aborted append left mid-flight (see Writing above), raising with the marker's
timestamp and incoming task/schedule counts unless called with
`allow_incomplete=True`. `verify_databank` skips both and reports them as
`store.protocol` and `append.in_progress` findings instead — auditing a broken
store is its job. `load_loss_raw` reads the raw sidecar rather than the
store, so it checks the store's stamp itself: the raw histories were written
under the store's protocol. `store_stamp(root, dataset_name)` returns a
store's stamp (`None` without one).

**Notes / gotchas**
- Missing `(task, schedule)` cells read back as `NaN`; `load_databank_dataset`
  logs an `n_filled / n_expected` completeness check.
- Loss variables come back as float64 (gap + f*), twice the memory of the
  float32 store once loaded.
- `target` is matched **positionally** (its only coordinate is an integer
  index); all cells share the identical target ordering.

## Verification

`verify_databank(root, dataset_name, *, task_ids=..., taskset=...,
schedules=...)` audits a store and returns a structured
`DatabankVerificationReport` (per-finding severity/check/pair fields, an
`ok` property meaning "no errors", and a readable `summary()`). It never
raises on anomalies — the caller decides whether to log the report or
fail on `not report.ok`. The severity model ("non-finite" = `NaN` or
`inf`):

- **warning** — *scattered* non-finite values inside the `loss_history`
  variables of a populated pair (expected: the optimizer produced them
  while traversing the problem), non-integer or budget-decreasing
  counters, taskset/subset hashes absent from the store.
- **error** — a `loss_history` variable *entirely* non-finite for a
  populated pair; any non-finite value in `success` / `cost`; a
  half-written cell (performance present but all loss variables NaN, or
  vice versa — an interrupted region write); a store whose protocol stamp
  is missing or names another protocol (`store.protocol`); a store an
  aborted append left mid-flight (`append.in_progress`); schema violations (missing
  variables, wrong dims, bad budget grid / metric coord / recipe JSON);
  internal inconsistencies (non-binary or target-non-monotone `success`,
  negative `output_std` or counters, non-finite / unsorted
  `target_value`, a task with the noise-free series `output_min_true` on
  some populated pairs only -- `task.noise_free_partial`).
- **info** — completeness (`n_populated/n_expected` plus the
  fully-unwritten pairs, which are by design `NaN` and *not* errors) and
  a non-default budget-grid resolution.

`output_min_true` is conditional (`docs/adr/0024`): a store without it
gets no finding about it, and an all-`NaN` cell of it (a noise-free
task's) is fine on its own. Where a pair has it, its non-finite values
follow the `loss_history` rules above.

`f_star` is a required per-task variable (`docs/adr/0027`): a store without it
gets a `structure.missing_var` ERROR, next to its `store.protocol` one.
`evaluation_budget` is conditional, like `output_min_true`. No check reads the
sign of a loss, so gaps and losses are audited the same way.

All per-pair checks are evaluated as one batch of lazy dask reductions
computed in a single pass over small per-pair summary arrays, so
verifying a large store never loads the full arrays into memory. A
missing store raises `FileNotFoundError` (use `populated_pairs` to probe
for presence instead).

## Tasksets

A **taskset** is a named selection of tasks: a plain-text file under
`tasksets/<name>.txt`, one task hash per line. Blank lines and `#` comments are
ignored; an inline `#` comment may follow a hash to record what it is. Load with
`load_taskset(rootdir, name)` (raises on missing/empty rather than returning an
empty list); write with `save_taskset`.

```
# tasksets/bbob.txt
a1b2c3   # sphere     10D i3
9f8e7d   # rastrigin  10D i1
# f12 omitted intentionally — known instability
```
