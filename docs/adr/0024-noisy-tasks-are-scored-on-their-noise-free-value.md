---
status: accepted
---

# Noisy tasks are scored on their noise-free value

A noisy task's loss is stochastic. Scored on its observed values, a run
looks better than it is: the lowest of many noisy draws tends to sit
below the function's real value at the points evaluated. Decision 12 of
the dataset paper's protocol
revision (COCO's convention) therefore scores a noisy task on the
**minimum of its noise-free loss over all evaluated points**. The
optimizer still sees only noisy values, and both series are stored.

Two sibling changes provide the pieces:
- l2co-tasks: `Task.loss_fn_true`, the noise-free loss, set only on
  stochastic tasks.
- l2co-optimizers: `UpdateClass.true_loss_fn`, set through
  `with_true_loss_fn`. When it is set, the run records
  `HistoryState.output_min_true`, the lowest noise-free loss of each
  step's evaluated population. When it is `None`, the task is
  noise-free and `output_min` already is the noise-free value.

This ADR is l2co's part: connecting the two, storing the series, and
scoring on it.

## Attaching the noise-free loss

**One helper, one call site.** `noise_free_loss(task)` in
`_src/task_bridge.py` turns `task.loss_fn_true` into the function the
optimizer takes. That function receives one member's parameters,
combines them with the model's static part, and calls `loss_fn_true`.
It returns `None` for a noise-free task. `init_run_state` attaches it
when called with `record_noise_free=True`. `RolloutWrapper.init` now
builds its run state through `init_run_state` with the flag on, instead
of repeating that function's body. So the only place an optimizer is
built from a task is `init_run_state`, and dataset generation
(`RunOptimization` → `RolloutWrapper.init`) records the series.

**Why it is opt-in.** rl2co's environment builds run states through
`init_run_state` and appends each step's history into one buffer with
`HistoryState.add`. Once one part carries the series, the buffer
changes structure, which breaks the environment's `lax.cond` and
autoreset. Turning it on for rl2co is rl2co's work.

**What is covered.**
- Every bare optimizer.
- The `"l2co"` and `"rl2co"` meta-optimizers, when built here. Their
  run loop is the inherited one, and their step records the population
  they asked for, NaN-padded. The series is therefore the noise-free
  minimum over the rows the active sub-optimizer evaluated. The
  sub-optimizers inside them get no noise-free loss and need none. This
  is from reading the code: neither was run, as both need checkpoints.
- The padding never counts: the helper returns NaN for a member with a
  NaN parameter, whatever `loss_fn_true` makes of it. That is the rule
  the menu evaluator's `row_has_nan` applies to the observed loss.

**What raises instead of computing something wrong.**
- A noisy task with a dataset: `NotImplementedError`. The run calls
  the noise-free loss without a data batch. No such task exists.
- The whole-run scipy entries (`slsqp`, `trustconstr`) on a noisy
  task: they raise `NotImplementedError` themselves. Noisy suites do
  not schedule them.

## Storing the series

- `OUTPUT_MIN_TRUE = "output_min_true"` sits next to `OUTPUT_MIN`. The
  history exports add it **only** when the run recorded it, so a
  noise-free run's dataset is unchanged.
- The raw sidecar and the interpolated store carry it like any other
  float variable.
- **A store has the variable exactly when it holds a noisy task**, one
  with a `loss_fn_true`. Its variables come from one sample cell, so the
  template takes that cell from the first noisy task. Taking it from the
  first run would drop the variable whenever a noise-free task comes
  first. The task is told by its `loss_fn_true`, so choosing it opens
  no file.
- **Cost in files.** A noise-free task's cells of the variable are
  never written. They have no chunk file and read back as NaN. A noisy
  cell costs one extra file, 10 instead of 9 per cell. The variable
  itself costs two files per store.
- **The dump refuses mixed scoring within a task.**
  `_build_task_records` raises when some of a task's schedules carry
  the series and others don't. It also raises when a task with
  `loss_fn_true` has no run that carries it, and when a task without
  `loss_fn_true` carries it. The last keeps a store's schema a function
  of its tasks, which the template's choice above relies on.
- **Append.** Noise-free runs may join a store that has the variable.
  Noisy runs may not join a store without it: that store's targets
  were computed from observed values, so the fix is a re-dump with
  `mode="replace"`. Decision 10 changes every task hash anyway.
- **Verify.** A store without the variable gets no finding about it. A
  task with the series on some populated pairs only is an ERROR
  (`task.noise_free_partial`). Scattered non-finite values in the
  series are a WARNING, as for the other loss variables.

## Scoring on it

`scoring_loss(data)` in `performance.py`, also exported from
`l2co.data`, is the one rule:
- No `output_min_true` in the data: it returns `output_min` itself.
- Otherwise the choice is per task. A task none of whose runs has the
  series is scored on `output_min`: a noise-free task in a mixed store,
  or in loaders concatenated from both kinds, reads the series as NaN.
  A task with the series is scored on `output_min_true`.
- A task with the series on some runs and observed values alone on
  others raises `ValueError`, naming it. The dump refuses that within
  one store, but combining sources can still produce it: an overlay
  read, or `DataLoader.join` / `+` of a store with the series and runs
  made without it (rl2co's `history_to_dataloader`). A run with neither
  series, an unwritten pair, counts as neither.
- Data without the variable is returned untouched and stays lazy.
- **Cost on data with the series.** A scoring call reads both series
  twice: once to tell which runs have the series, and once more when
  its lazy result is used. That is about four reads of a loss variable
  where a noise-free score makes one. The report scores several times
  per call. Accepted for now and timed in the re-run's pilot; computing
  the scored series once per `DataLoader` is the fix if it hurts.

Where it is applied:
- `shift_to_quality_value`. It puts the scored series in `output_min`
  and drops `output_min_true`. Everything downstream of the shift reads
  one variable: `compute_task_targets` (including the random-search
  anchor of the easiest target) and `calculate_ert`. This covers the
  dump, `history_to_dataloader` and the legacy `.odata` path.
  `compute_task_targets` now refuses unshifted data that still holds
  the series.
- `global_min_breaches`. A noisy observed value below `global_min` is
  noise, not a `global_min` to re-tune.
- `DataLoader._performance` (the SBS/VBS picks), the report's
  final-loss scores (`per_task_shift`, `shifted_final_min`, and
  `build_report`'s `final_raw`), and the "solved" mark of
  `plot_update_step`.

**What stays observed.** `DataLoader.to_traj_and_stair` is the model's
input, so it must show what a policy sees during a run. The report's
crash check and the `log_relative_improvement` feature also stay
observed.

**Noise-free tasks are unaffected.** `scoring_loss` returns the same
array, so every score is computed by the same operations. On a
synthetic noise-free store, before and after this change, the stored
arrays, the raw sidecars, every `zarr.json`, the breach file, the
verify findings and the loader report all came out identical.

## Not covered

- The legacy `.odata` path (`create_combined_odata`) does not run the
  dump's checks. Through `scoring_loss` it still refuses a task with
  the series on some schedules only: concatenating the schedules
  NaN-fills the series where it is missing. But it scores a noisy task
  none of whose runs carries the series on its observed values. The
  dump's check lives in `databank/records.py`, which imports
  `data_validation`, so calling it from there would be circular. The
  pipeline's databank dump, which runs on the same runs, refuses them.
  Left open on purpose: the dataset pipeline no longer goes through
  this path, and the two that do (`evaluate_rl`, `agentic_smoke`) run
  every schedule through `RolloutWrapper`, so their noisy runs carry
  the series.
- rl2co's training environment and any direct `init_run_state` call
  without the flag. Their runs on noisy tasks have no series, and the
  dump refuses them.
- l2co_experiments modules that read `output_min` directly (the
  selection reports, the CUTEst floor, validation analysis). They keep
  scoring observed values until they adopt `scoring_loss`.
- Decision 14 (store the gap f − f\*) will rewrite both series. Keeping
  every read behind `scoring_loss` keeps that change in one place.

## Rejected

- Attaching by default in `init_run_state`: breaks rl2co's environment
  (above).
- A warning instead of an error, in `scoring_loss` and in the dump, for
  a task scored on two series. On a noisy task the observed minimum
  sits below the noise-free one, so a comparison of rl2co's runs
  against databank baselines would favour rl2co, with only a log line
  to show it.
- Wrapping `optimizer_mapping`: it returns factories, and the registry
  guards that a name resolves to the registered object.
- A `series` dimension on `output_min`: changes `output_min`'s dims for
  every consumer.
- Declaring the variable in every store: changes every noise-free
  store's schema and breaks append against existing stores.
