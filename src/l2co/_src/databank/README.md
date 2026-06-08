# Databank (consolidated-zarr)

A lazy-loadable databank that stores a whole dataset in **one zarr store** plus
a sidecar of raw loss histories. It exists to make reads cheap at scale
(hundreds of tasks × tens of optimizers), especially on networked HPC
filesystems where opening thousands of HDF5 files is expensive.

```
<root>/<dataset_name>.zarr/                       # consolidated store
    loss_history (interp), performance, task, optimizer
<root>/<dataset_name>/loss_history_raw/           # per-pair raw netCDF
    loss_<taskID>_<optimizer_hash>.nc
<root>/<dataset_name>/tasksets/<name>.txt         # named task selections
```

`<root>` is the databank base directory supplied by the caller (the application
resolves it from its own config, e.g. a per-cluster `databank_root` key — the
library holds no machine paths). `<taskID>` is
`l2co_tasks.Task.hash` (a 16-hex string); `<optimizer_hash>` is
`l2co.OptimizationStep.hash` (`int(sha256(...)[:6], 16)`). The **human-readable
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
| `output_min` / `output_mean` / `output_std` / `fevals` / `iterations` / `update_step` | `(taskID, schedule, realization, output_dim, budget)` | `budget` = `BUDGET_STEPS`, canonical grid |
| `success` | `(taskID, schedule, realization, output_dim, target)` | float (0/1, NaN if missing) |
| `cost` | `(taskID, schedule, realization, output_dim, target, metric)` | float |
| `task_labels` | `(taskID, feature)` | str |
| `target_value` | `(taskID, target)` | float |
| `optimizationstep` | `(schedule,)` | str (JSON recipe; dropped by the loader) |

`metric` coord is `["fevals", "iterations"]`. `target` is an integer index; the
target *values* live in `target_value`. `task_labels` carries the task tag plus
`global_min` (one `feature` per key).

## Writing

`dump_databank` reuses the shared f3dasm preamble and ERT pipeline
(`records._iter_task_runs`, `records._build_task_records`). Per pair it writes
the raw history to `loss_history_raw/` and region-writes the interpolated history
+ performance into the store.

- **Creation** (`mode="error"` / `"replace"`) streams one `(task, optimizer)`
  cell at a time into a `compute=False` template, so the full dense array is
  never held in memory.
- **Append** (`mode="append"`) rebuilds the store over the union of old and new
  `taskID` / `schedule` into a temporary store and swaps it in atomically,
  copying untouched cells from the old store one at a time. It reuses the
  existing budget grid and each existing task's stored ERT `target_value` (so
  appended optimizers stay comparable). The rebuild — rather than an in-place
  coordinate resize — avoids silently truncating optimizer names in the
  fixed-length string `schedule` coordinate.

Writes that touch distinct `(taskID, schedule)` cells are chunk-disjoint and
safe to parallelise; resizing and `consolidate_metadata` are single-writer.

## Reading

`load_databank_dataset` opens the consolidated store, `.sel`s the requested
tasks/optimizers (optimizers given as `OptimizationStep`, schedule-name strings,
or integer hashes), and returns the lazy selection — no per-pair resampling,
since `loss_history` is already on the `BUDGET_STEPS` grid. `load_databank`
drops `optimizationstep` and returns a lazy `DataLoader` (dask reads only the
chunks each operation touches; call `.load()` to force it all into memory).
`load_databank_performance` keeps only the performance variables, so the large
loss chunks are never read (use it for ERTD reporting). `load_loss_raw` reaches
a single raw history (optionally resampled).

**Notes / gotchas**
- Missing `(task, schedule)` cells read back as `NaN`; `load_databank_dataset`
  logs an `n_filled / n_expected` completeness check.
- `target` is matched **positionally** (its only coordinate is an integer
  index); all cells share the identical target ordering.

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
