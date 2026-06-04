# Per-record databank

A lazy-loadable databank. A sweep is stored as **one file per record**, split
into four typed subfolders plus a `tasksets` folder. Files are hash-named so a
`(task, optimizer)` selection is addressable directly from the chosen
combinations — no globbing. The submodule is self-contained (no dependency on
the legacy per-schedule databank).

> **Concat-axis rule.** Every payload variable carries its concat axis as a
> *leading dimension* — `taskID`/`schedule` for `loss_history` and
> `performance`, `taskID` for `task`, `schedule` for `optimizer`. This is
> required for `xarray.open_mfdataset(combine="by_coords")` to stack files
> along those axes. If a payload variable carried the axis only as a
> *coordinate* (not a dimension), `data_vars="minimal"` + `compat="override"`
> would silently keep only the first file's values and discard the rest.

## Storage layout

```
<root>/<dataset_name>/
├── task/<taskID>.nc                       # one file per task
├── optimizer/<optimizer_hash>.nc          # one file per optimizer
├── loss_history/loss_<taskID>_<optimizer_hash>.nc
├── performance/perf_<taskID>_<optimizer_hash>.nc
└── tasksets/<taskset>.txt
```

`<root>` comes from `_LOCATION_ROOTS[location]` (defined in `layout.py`).
`<taskID>` is `l2co_tasks.Task.hash` (a 16-hex
string); `<optimizer_hash>` is `l2co.OptimizationStep.hash`
(`int(sha256(...)[:6], 16)`). The **human-readable optimizer name**
(`OptimizationStep.name`) is carried inside each file as the `schedule`
coordinate; the integer hash is used only in filenames.

For *n* tasks × *m* optimizers: *n* files in `task/`, *m* in `optimizer/`, and
*n × m* in each of `loss_history/` and `performance/`.

## Per-type schemas

### `loss_history` — dims `(taskID, schedule, realization, output_dim, budget)`

| Variable | Dtype | Notes |
|---|---|---|
| `output_min` / `output_mean` / `output_std` | float | |
| `fevals` / `iterations` / `update_step` | int | cumulative |

`budget` is **raw** (variable length per file) and has no coordinate on disk; it
is resampled to a common grid at open time (see *Opening & merging*).

### `performance` — dims `(taskID, schedule, realization, output_dim, target[, metric])`

| Variable | Dims | Dtype |
|---|---|---|
| `success` | `(…, target)` | bool |
| `cost` | `(…, target, metric)` | int |

`metric` coord is `["fevals", "iterations"]`. `target` is an integer index
`0..N-1`; the actual target *values* live in the `task` file (`target_value`).

### `task` — dims `(taskID, feature, target)`

| Variable | Dims | Dtype |
|---|---|---|
| `task_labels` | `(taskID, feature)` | str (mixed types stringified) |
| `target_value` | `(taskID, target)` | float |

`task_labels` carries the task tag plus `global_min` (one `feature` per key).

### `optimizer` — dims `(schedule,)`

| Variable | Dims | Dtype |
|---|---|---|
| `optimizationstep` | `(schedule,)` | str (`json.dumps(OptimizationStep.to_dict())`) |

## Opening & merging

`load_databank_dataset` opens each type in its own `open_mfdataset` call (they
have different variables and concat axes) and `xr.merge`s the four:

```python
open_kw = dict(combine="by_coords", join="outer", data_vars="minimal",
               coords="minimal", compat="override", engine="netcdf4")
loss = xr.open_mfdataset(loss_paths, preprocess=loss_preprocess, **open_kw)
perf = xr.open_mfdataset(perf_paths, **open_kw)
task = xr.open_mfdataset(task_paths, **open_kw)
opt  = xr.open_mfdataset(opt_paths,  **open_kw)
ds = xr.merge([loss, perf, task, opt], join="outer", compat="override")
```

`loss_preprocess` normalizes each file's raw `budget` to the canonical L2CO grid
by delegating to `l2co.validation.normalize_budget_grid` (the `[-step, 0, …, 1]`
slot-0 / "pre-optimization" convention), then casts the integer counters back to
`int` (nearest-neighbour interpolation promotes them to float). `nearest`
returns *recorded* values, so `budget=0` is the true initial loss and the final
iteration is preserved exactly.

`load_databank` materializes the merge, drops the `optimizationstep` provenance
variable, and returns a `DataLoader` (the remaining variables match the legacy
`entire_dataset.odata` schema, so training is unaffected).

**Notes / gotchas**
- `join="outer"` silently NaN-pads missing `(task, schedule)` cells;
  `load_databank_dataset` logs an `n_filled / n_expected` completeness check.
- `compat="override"` keeps the open lazy; `"equals"`/`"identical"` read data to
  compare and are not lazy.
- `target` has no coordinate value beyond its integer index → it is matched
  **positionally**; all files must share the identical target ordering.

## Selecting paths

When both the `taskID` list and the optimizer list are known, filenames are
fully determined, so `select_paths` builds the candidate paths from the cross
product and splits them into existing / missing (one `stat` per candidate, no
directory listing). Missing combinations are returned (and logged) rather than
letting the dataset silently shrink. The `task` / `optimizer` paths come from
the *distinct* ids, not the cross product.

> If only *one* identifier is known (e.g. tasks selected, optimizers unknown),
> glob with a single wildcard per task and filter in Python. Reserve this for
> the unknown-identifier case, since each glob scans the directory.

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
