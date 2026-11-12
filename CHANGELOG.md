# Changelog

All notable changes to L2CO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Switch-level transfer mode** (#117, `docs/adr/0028`).
  `l2co.optimizers` exports `TRANSFER_MODES` (`"full"`, `"none"`),
  `DEFAULT_TRANSFER_MODE` (`"full"`), `validate_transfer_mode` and
  `transfer_mode_index` (the mode's stable integer code, for a traced
  per-switch argument). `none` is a cold switch: the incoming
  optimizer starts as at the beginning of a run and nothing crosses the
  switch. The behaviour will be implemented in the rl2co environment.
  `l2co_population_strategy` accepts `transfer_mode` (default `"full"`)
  and raises `NotImplementedError` on `"none"`.
- **`opt_state_handshake="bundle"` on `l2co_population_strategy`**
  (l2co#120). At a switching stage-1 fire the wrapper now can apply
  rl2co's three-step handshake: reset or continue the incoming state as
  `HANDSHAKE_POLICY` says, fold in the scale read from the stage-0
  optimizer, then take the population from the incoming optimizer's own
  `ask` (where `own_ask`) or the canonical layout, NaN-padded. This is
  the handshake an RL2CO policy is deployed with, for a matched
  comparison. `"legacy_reset"` stays the default, so published L2CO
  results reproduce; the handshake generation is billed as before
  (ADR 0004).
- **`l2co.optimizers.switch_handshake`**: the three steps on one
  `SubOpt` and a `TransferBundle`, free of any caller's run state, so
  rl2co's environment and wrappers can share it (ADR 0019).
- **`l2co.data.time_shared_portfolio`: the time-shared portfolio of
  stored schedules as a schedule of its own** (l2co#115). Run r of the
  portfolio interleaves run r of K members in round robin, with no state
  transfer: `best_portfolio(F) = min_k best_k(w_k F)`. It is built from
  the stored histories, with no new optimization, on the members' budget
  grid, with `output_min` (and `output_min_true` for noisy tasks),
  `fevals` and `iterations`, so `calculate_ert`, ERTD and AUC read it
  unchanged. Options: `weights` (equal by default) and `granularity`
  (`"fevals"`, idealized, or whole `"iterations"`). The schedule is named
  `portfolio(<m1>+<m2>+...)` and its attrs mark it as derived.
- **Optimizers whose population shrinks get their whole evaluation
  budget.** `evaluations_based_on_dimensionality` divided the budget B
  by the optimizer's `popsize`, which for JDE21, MadDE and NL-SHADE-RSP
  is only the starting size, so their runs spent a fraction of B. An
  optimizer that defines `generations_for_budget(B, d)` on its
  `UpdateClass` now gets that many steps instead; it wins over
  `popsize` and `max_fevals_per_step`, and a value below 1 raises
  `ValueError`. Every other optimizer keeps floor(B / e) (#121,
  `docs/adr/0026`).

## [2.0.1] - 2026-11-08

### Fixed

- **A dump of a task whose cells differ in length no longer fails.** Under
  the evaluation budget a task's cells record different numbers of steps
  (B for random search and first-order methods, about B / lambda for a
  population method), and the run histories carry no `budget` coordinate,
  so combining them to score the task raised `AlignmentError: cannot
  reindex or align along dimension 'budget'` in both the template step
  of a new store and every per-task step. `_build_task_records` now
  extends each cell to the longest one by repeating its last recorded
  step before scoring (a finished run stays at its final state), which
  leaves every cell's ERT, successes and breaches unchanged. Stored cells
  and raw histories are written as they ran.

## [2.0.0] - 2026-11-06

The databank and run protocol of the dataset paper's re-run. Stores now
carry a protocol stamp, hold gaps to f\*, score noisy tasks on their
noise-free values, and check an evaluation budget; stores written before
this release are refused. Requires `bbob-jax>=3.0.0`, `l2co-tasks>=0.5.0`
and `l2co-optimizers>=0.5.0`.

### Changed

- **Stores carry a protocol stamp** (`docs/adr/0025`). Every new store
  holds `l2co_stamp.json`, naming the run protocol (`DATABANK_PROTOCOL`)
  it was written under. Every reader, `load_loss_raw` included, an append
  and a `mode="replace"` dump refuse a store whose stamp is missing or
  names another protocol (`StoreProtocolError`, a `RuntimeError`), before
  anything is opened, written or deleted; `verify_databank` audits any
  store and reports the stamp as a finding. **Every store written by
  l2co 1.8.1 or earlier is refused**: re-run it, or read it with 1.8.x.
- **Noisy tasks are scored on their noise-free value** (`docs/adr/0024`).
  Dataset generation records `output_min_true`, the noise-free minimum
  of each step's evaluated points (l2co-tasks `Task.loss_fn_true`,
  l2co-optimizers' noise-free history), stores it, and the dump-time
  scores of a noisy task (targets, ERT, breaches) read it instead of the
  observed `output_min`.
- **Databank stores hold gaps** (`docs/adr/0027`). The writer stores
  `output_min`, `output_min_true` and `output_mean` as f − f\* (f\* =
  the task's `global_min`, subtracted in float64 after the dump-time
  scores are computed), in the store and in the raw files. Every reader
  (`load_databank*`, overlay reads, `load_loss_raw`) adds f\* back, so
  callers still get losses, now as float64 and precise to float32 of the
  gap. Each store has a float64 `f_star` variable per task; each raw
  netCDF file has `f_star` and `value_kind` attributes. Tasks without a
  finite f\* are stored raw. Code reading the store or raw files
  directly now sees gaps.
- **`DATABANK_PROTOCOL` is `dataset-paper-2026-11.1`.** Stores written
  under `dataset-paper-2026-11` are refused.
- The simulated restart orders of the ERTD are permutations only:
  `draw_samples(seed, n_realizations)` lost its never-reached
  with-replacement tail and its `n_picks` argument, `ERTD_N_PICKS` is
  removed, and `compute_ert_multi_seed` lost `n_picks`. No ERTD value
  changes.
- `RunOptimization` now calls its budget function after the optimizer is
  built, as `budget_fn(task=task, optimizer=update_class)`. A custom budget
  function must accept the `optimizer` keyword: `lambda task: n` now
  raises `TypeError`; use `lambda task, optimizer: n`, or
  `fixed_budget` / `based_on_dimensionality`, which accept and ignore it.
- **An append refuses a task whose f\* differs from the stored one**
  (`docs/adr/0027`). With l2co-tasks 0.5.0 a corrected `global_min` no
  longer changes the task hash, so the same task can arrive with another
  f\*. The append now stops before anything is written, deleted or
  marked, naming each such task with its stored and incoming f\*
  (`FStarError`); NaN (no finite f\*) equals NaN. Before, the incoming
  f\* silently replaced the stored one for the tasks a dump wrote. A
  dump (create, replace or append) also refuses runs that give one task
  hash two different f\* values.
- Dependency floors: `bbob-jax>=3.0.0` (CEC 2005 compositions follow the
  official definitions), `l2co-tasks>=0.5.0` and `l2co-optimizers>=0.5.0`.

### Added

- `l2co.budget.evaluations_based_on_dimensionality`: every run gets the
  same number of function evaluations, B = min(1000·d, 10⁵), returned as
  ⌊B/e⌋ iterations, where e is what one iteration of the optimizer costs
  (its population size, or 1 for L-BFGS and Nelder–Mead). See
  `docs/adr/0026`.
- `l2co.budget.evaluation_budget(task, evaluations_per_dimension=1000,
  max_evaluations=100_000)`: the number of function evaluations B a run on
  the task gets, without needing an optimizer.
  `evaluations_based_on_dimensionality` uses it.
- `evaluation_budget=` on `dump_databank`, `dump_runs` and
  `dump_databank_template` (B per task hash): every cell is checked
  against its task's B, the scores are cut at B, and B is stored as the
  per-task `evaluation_budget` variable; an append refuses a differing
  B. Without it, nothing changes.
- `calculate_ert(..., evaluation_budget=B)`: points with `fevals` > B
  are ignored, and a missed run is charged at most B.
- `run_settings=` on the same entry points, stored in the protocol stamp
  and compared on append; `l2co.databank.check_run_settings` and
  `RunSettingsError` for a create step to check them before any cell
  runs.
- `l2co.databank.check_f_star(root, dataset_name, f_star_by_task)` and
  `FStarError` (a `ValueError`): a create step checks each task's f\*
  against the store before any cell runs. A missing store, and a task
  the store does not hold, pass; a store left mid-append is refused.

## [1.8.1] - 2026-11-05

The first release on PyPI (`pip install l2co`), with the repository made
public now that the paper is published. One fix, for matplotlib 3.11.

### Added

- How to cite the paper, *Learning to Choose Optimizers* (IEEE CEC 2026,
  doi:10.1109/CEC69861.2026.11727310): a BibTeX entry in the README and
  on the docs home page, and a `preferred-citation` in `CITATION.cff`, so
  GitHub's "Cite this repository" gives the paper.
- The README says the paper's results were produced with v1.3.1.
- Documentation on readthedocs (`.readthedocs.yaml`), and a release
  workflow that publishes to PyPI when a new version reaches `main`.

### Fixed

- With matplotlib 3.11, an overlong curve label in `plot_ertd`'s
  right-hand column (`right_labels=True`) ran a few pixels past the
  axes and was cropped. Shrinking it to fit assumed text width is
  proportional to font size, which stopped holding in 3.11: widths now
  snap to whole pixels. The labels are now measured again after the
  first shrink and shrunk further until they fit.

### Changed

- The experiment configs `l2co`, `medium_plus_l2co` and
  `gaussian_classification_plus_l2co` no longer hold paths to the
  author's trained models. The model paths are now required values
  (`???`) that you pass on the command line; Hydra stops with a
  "missing mandatory value" error naming the key if you don't.
- `experiments/generate_figures.py` takes its data folder as the first
  command-line argument and writes figures to `./plots/`, as
  `experiments/README.md` already said.

### Removed

- `experiments/conf/optimizers/l2co_gaussian_classification.yaml`, an
  unreferenced near-copy of `gaussian_classification_plus_l2co`.
- The README's list of related repositories; l2co-optimizers joins the
  dependency table instead.

## [1.8.0] - 2026-11-04

### Changed

- A loss below a task's `global_min` now scores as reaching it. The
  quality value `loss - global_min` is floored at 0 in
  `shift_to_quality_value` (ERT targets and success), `min_custom`
  (`scale_by_task`), the report's `shifted_final_min` and
  `log_stabilized_minmax_scale_with_global_min` (`traj_and_stair`, so
  training and inference features alike). Before, an empirical
  `global_min` that a run beat gave a negative gap, and NaN under the
  log scales. ERT success is unchanged: such a run already solved
  every target.

### Added

- Every databank dump writes `<root>/<dataset>/global_min_breaches.json`:
  each task on which a run went below `global_min`, with the lowest
  loss, how far below it is, and which schedule and realization
  reached it. A finalize warning gives the count and path. The margin
  for float noise is `1e-6 * max(1, |global_min|)`, as in the CUTEst
  floor test. `l2co._src.performance.global_min_breaches` finds them
  for one task.
- Performance profiles (`docs/adr/0021`): how each optimizer's cost to
  reach a target compares with the cheapest optimizer's, task by task.
  - `DataLoader.expected_runtime(positions, cost_metric="fevals")`: for
    each task and schedule, the cost spent over all runs divided by the
    runs that reached the target, at each ladder position (0 = easiest
    stored target, 1 = hardest). NaN is no data, infinity never reached.
  - `l2co.data.performance_profile(ert, max_ratio=1e4, n_points=201)`:
    ratio curves over any subset of that table, with `win_rate`,
    `solve_rate`, and counts of tasks nobody solved and tasks dropped
    for missing data.
  - `l2co.plotting.plot_performance_profile`: draws one profile, with
    `plot_ertd`'s styling options and an ∞ point at each solve rate.
  - `l2co.naming.POSITION` and `RATIO`: the two new axes.

### Removed

- `l2co._src.performance.performance_profile` (a private final-loss
  ratio profile with no callers). The name now belongs to the function
  above.

## [1.7.0] - 2026-10-07

The run loop moves onto `UpdateClass` in `l2co-optimizers`
(`docs/adr/0017`, superseding that half of `docs/adr/0016`).
**Requires an `l2co-optimizers` that ships `UpdateClass.run` and
`RunState`** (the release after 0.1.0).

Then the task stops at l2co (`docs/adr/0018`): `l2co-optimizers` no
longer has a task type, and l2co is the only bridge between `Task` and
`OptimizationStep`. **Requires the matching `l2co-optimizers`**
(bare factories take `model` / `loss_fn` / `pass_rng`).

Then the switching layer comes back to l2co (`docs/adr/0019`): l2co
holds both the bridge to tasks and the machinery that switches between
optimizers; `l2co-optimizers` keeps the transfer ports and a new
`optimizer_parts` seam. **Requires the matching `l2co-optimizers`**
(ships `optimizer_parts`, no longer ships `SubOpt`).

Then bounds come from the task (`docs/adr/0020`): a task declares its
box as an `l2co_tasks.Box` in its constraints, and the optimizers clip
to it. Callers no longer pass `bounded`. **Requires an `l2co-tasks` with
constraints on `Task`** (l2co-tasks ADR 0002).

### Added

- `init_run_state(optimizer, task, key)` (in `l2co.optimization`
  and at the package root): resolves an `OptimizationStep` against a
  `Task` and builds the `RunState`, clipping to the task's own box.
- `task_bounds(task)` and `refuse_constraints(task, where)` (in
  `l2co.optimization` and at the package root): the only way bounds
  leave a task. `task_bounds` turns a uniform `Box` into the `(lo, hi)`
  pair optimizers clip to and raises for an inequality, an equality or a
  non-uniform box; a path that can't enforce constraints calls
  `refuse_constraints`.
- `l2co.optimizers.meta_optimizers`: the registry `register_optimizer`
  fills.
- `history_to_xarray`, `history_to_xarray_realizations` and
  `history_to_dataloader` in `l2co.history` and at the package root: the
  `HistoryState` exports, as free functions.
- `l2co.optimizers.optimizer_parts`, `GradientParts` and
  `PopulationParts` (re-exported from `l2co_optimizers`): an optimizer
  unpacked into the optax transform or `(init, ask, tell)` triple a
  switching loop drives, with popsize, family and transfer ports but no
  handshake.

### Changed

- `l2co-tasks>=0.3.0` and `l2co-optimizers>=0.3.0` are declared
  lower bounds: l2co imports `l2co_tasks.Box` (`docs/adr/0020`) and is
  tested against l2co-optimizers 0.3.0. Until now both were unpinned.
- The switching layer is implemented in l2co again
  (`l2co._src.switching`; `docs/adr/0019`): `SubOpt`,
  `grad_sub_optimizer`, `pop_sub_optimizer`, `optstep_to_subopt`,
  `HandshakePolicy`, `HANDSHAKE_POLICY`, `DEFAULT_HANDSHAKE_POLICY`,
  `handshake_policy_for`, `population_best`, `population_best_one` and
  `menu_loss_and_grad` keep their `l2co.optimizers` names but are no
  longer re-exports of `l2co_optimizers` objects. `optstep_to_subopt`
  now unpacks `l2co_optimizers.optimizer_parts` and attaches the
  handshake; the `SubOpt` it builds is unchanged.

- `BatchState` and `HistoryState` are re-exported from `l2co_optimizers`
  (same names under `l2co` / `l2co.history`). `run_state.run` /
  `batch_run` call `UpdateClass.run` / `batch_run`, so the driver choice
  (fused, sequential for `sequential_realizations`, sequential for random
  search) belongs to the optimizer.
- `RunState`, `reset`, `run` and `evaluate` move to `l2co_optimizers`
  (with `batch_reset` / `batch_run` / `batch_evaluate`) and are
  re-exported unchanged as `l2co.RunState` / `l2co.reset` / `l2co.run` /
  `l2co.evaluate` and from `l2co.optimization`.
- **BREAKING:** `l2co.optimizer_mapping` / `l2co.register_optimizer` (and
  the `l2co.optimizers` versions) are now l2co's Task-level registry.
  `optimizer_mapping(name)` returns a factory called with `task=` for
  every name, meta or bare (unchanged for existing call sites).
  `register_optimizer` registers a *meta-optimizer*, and raises if the
  name is already a bare optimizer. `l2co.optimizers` no longer
  re-exports the bare `optimizers` dict; import it from
  `l2co_optimizers`.
- **BREAKING:** `RunState.init` (from `l2co_optimizers`) takes
  `(update_class, *, model, dataset, batch_size, key)`. Replace
  `RunState.init(optimizer=, task=, bounded=, key=)` with
  `l2co.init_run_state(optimizer=, task=, key=)`, and declare the bounds
  as an `l2co_tasks.Box` in the task's `constraints` (`docs/adr/0020`).

### Removed

- **BREAKING:** the `bounded` parameter of `RolloutWrapper.init` and
  `RunOptimization`. Declare bounds as an `l2co_tasks.Box` in the task's
  `constraints` instead (`docs/adr/0020`). The optimizer factories keep
  their `bounded=` keyword, filled from `task_bounds`.
- **BREAKING:** `l2co_population_strategy` refuses a task with
  constraints: a strategy driven by hand has no way to enforce them.
- **BREAKING:** `l2co.optimizers.resolve_popsize` and
  `CONSTRUCTOR_HYPERPARAMETERS` (private in `l2co-optimizers` now). Read
  the popsize as `optimizer_parts(opt_step, model=, loss_fn=,
  pass_rng=).popsize`.
- **BREAKING:** re-exports of names `l2co-optimizers` made private (no
  sibling package used them): from `l2co.optimizers`, the per-library
  factories `optax_update`, `optax_update_extra_kwargs`,
  `evosax_distribution_update`, `evosax_population_update`,
  `random_search_update`, `lbfgs_update`, `lbfgs_per_eval_key`,
  `scale_by_zoom_linesearch_per_eval_key`, `normalized_optax_from_state`,
  the SubOpt tables `l2co_native_optax`, `l2co_native_evosax`, the tables
  `GRAD_EVALUATION_WIDTH`, `LINESEARCH_FEVAL_BOUND`, `OPT_STATE_VARIANTS`,
  `POPULATION_VARIANTS`, `POPULATION_HANDSHAKES`, `TRANSFER_OVERRIDES`,
  `TransferSpec`, `transfer_spec_for`, `population_handshake_for`,
  `write_none`, and `vmapped_loss_and_grad_with_rng`; `SHADE` from
  `l2co` and `l2co.optimizers`; `vmapped_loss_and_grad_with_rng` from
  `l2co._src.tree_utils`. Built-in optimizers are reached by registry
  name (`l2co.optimizer_mapping("shade")`).
- **BREAKING:** the `TaskLike` re-export from `l2co._src.typing`
  (`l2co_optimizers` deletes `TaskLike` / `RunnableTaskLike`).
- **BREAKING:** `HistoryState.to_xarray`, `to_xarray_realizations` and
  `to_dataloader`. Call `l2co.history_to_xarray(history_state, ...)`,
  `l2co.history_to_xarray_realizations(history_state, ...)` and
  `l2co.history_to_dataloader(history_state, ...)` instead.
- **BREAKING:** `l2co.init_` (also `l2co.optimization.init_`). Call
  `update_class.init_state(params, batch_state, dataset, key)` instead.
- The private `l2co._src.batching`, `l2co._src.update_class` and
  `l2co._src.run_state` modules (`l2co.evaluate`, `l2co.RunState` and
  friends are unchanged; inside-library code imports them from
  `l2co_optimizers`), and with them `step` / `batch_step_` / `run_` /
  `batch_run_` / `batch_run_sequential_` / `run_randomsearch` /
  `batch_run_randomsearch`, plus the `Carry` alias of `l2co._src.typing`.
  They are `UpdateClass` methods and `l2co_optimizers.Carry` now.

- **BREAKING:** `RolloutWrapper.init_from_stage0`, `RolloutWrapper.evaluate`
  and `RolloutWrapper.run`, plus `reset_from_stage0` and `evaluate` in
  `_src/run_state.py` (neither was re-exported). They only served the eager
  two-stage driver for agentic-l2co, which l2co_experiments has dropped
  (bessagroup/l2co_experiments#45). The L2CO strategy reaches stage 1
  through the registry bridge (`docs/adr/0002`). Use
  `RolloutWrapper.batch_evaluate` (one realization: `n_realizations=1`).

## [1.6.0] - 2026-10-02

The bare optimizers move to the new sibling package `l2co-optimizers`
(`docs/adr/0016`); every public l2co path re-exports the same objects.
**Requires `l2co-optimizers` 0.1.0** and, for rl2co users, an rl2co that
registers its own `"rl2co"` schedule namer (bessagroup/rl2co#49).

### Added

- Two surrogate-based optimizers, filling the portfolio's empty
  model-based category (now shipped by `l2co-optimizers`):
  - `rbftrustregion`: an RBF-surrogate trust-region searcher -- fit a cheap
    surrogate to the evaluated points and spend the next evaluations where
    it predicts low loss, inside a trust region that grows on success. Its
    candidate count scales with dimensionality, keeping per-step cost in
    check at 1024 dimensions.
  - `turbo`: TuRBO, trust-region Bayesian optimization.
- `ertd(..., realizations=)` and `DataLoader.ertd_profile(realizations=)`:
  the runs the restart bootstrap may draw from. Two disjoint halves give two
  independent estimates of the same profile, whose spread is the sampling
  error. Default `None` keeps every run.
- l2co now depends on the new sibling package **`l2co-optimizers`**, which
  holds the bare optimizers: the registry, `UpdateClass`, `OptimizationStep`,
  the optax / evosax / SHADE / TuRBO / RBF-trust-region / L-BFGS /
  random-search factories, `SubOpt`, the handshake policy, state transfer,
  `menu_loss_and_grad`, the samplers and `OptHistory` (`docs/adr/0016`). It
  imports neither l2co nor l2co-tasks; tasks reach it as `TaskLike`. Every
  public l2co path (`l2co.optimizers`, `l2co.OptimizationStep`,
  `l2co.sampling`, `l2co.history.OptHistory`, `l2co.tree_utils.vmapped_*`,
  `l2co.naming`'s aliases) re-exports the same objects, so downstream code
  keeps working unchanged. **Release `l2co-optimizers` 0.1.0 before or with
  this version.**
- `register_schedule_namer` (from `l2co-optimizers`): l2co registers how
  `"l2co"` steps are named, byte-identical to before. rl2co must register
  `"rl2co"` itself, so update rl2co together with this version.
- `l2co.optimizers.optax_update`, `optax_update_extra_kwargs`,
  `evosax_distribution_update` and `evosax_population_update` (formerly
  `UpdateClass` classmethods).
- `l2co.optimizers.AskFunction`, `TransferReadFunction` and
  `TransferWriteFunction`: type aliases for `UpdateClass.ask_fn`,
  `transfer_read_fn` and `transfer_write_fn`.

### Changed

- Stopped tracking 27 accidentally committed scratch artefacts (three
  unreadable root-level zarr fragments, `FEEDBACK_*.md`,
  `scratch_precision_check.py`) and added the `.gitignore` rules that should
  have covered them.
- `StepFunction`, `InitFunction` and `LossFunction` aliases now describe the
  real call signatures (`step_fn(carry, sample) -> (carry, OptHistory)`,
  `init_fn(params, key=None, **sample)`, `loss_fn(model, **sample)` returning
  a scalar).
- Built-in `step_fn` closures no longer accept unused `*args, **kwargs`.
- `UpdateClass.stop_fn` defaults to `None`, meaning "never stop", resolved at
  trace time in `step`; `OptimizationStep.stopping_fn` returns `None` when no
  stopping criteria are set. Previously every factory forwarded its own
  `stop_fn=None` default over the field default, so a factory called without
  `stop_fn` crashed on the first step (`'NoneType' object is not callable`).
- `StopFunction` now describes the real call, `stop_fn(recent_history,
  opt_state)` returning a boolean array; the stopping criteria are annotated
  against `RecentHistory` to match.
- The popsize helpers (`variable_popsize`, `variable_popsize_even`,
  `shade_popsize`, `resolve_popsize`) moved from `l2co._src.optimizers.utils`
  to `l2co._src.optimizers.popsize`; the `l2co.optimizers` facade is unchanged.

### Fixed

- `bounded=None` now means unbounded in every built-in closure factory, and
  `optax_update` / `optax_update_extra_kwargs` default to `(None, None)` like
  the evosax factories. Before, `bounded=None` (the optax default) crashed the
  first step on `jnp.clip(p, *bounded)`, and passing it explicitly to SHADE or
  TuRBO did the same.
- The `(init_fn, step_fn)` closure factories moved out of
  `l2co._src.optimizers.update_class`, which now holds only `UpdateClass`:
  `optax_fn`, `optax_extra_kwargs_fn` and `step_fevals` to
  `optax_implementations`; `evosax_ask_fn`, `evosax_distribution_based_fn` and
  `evosax_population_based_fn` to `evosax_implementations`;
  `l2co_population_based_fn` to `l2co._src.meta_optimizer`.
- **Breaking:** the `UpdateClass` factory classmethods are now plain functions
  returning an `UpdateClass`, next to the closures they wrap:
  `UpdateClass.optax_update` -> `l2co.optimizers.optax_update` (likewise
  `optax_update_extra_kwargs`, `evosax_distribution_update`,
  `evosax_population_update`). Same signatures minus `cls`; all four are
  exported from `l2co.optimizers`. `UpdateClass` itself is now only the
  container.

### Removed

- The private `l2co._src.optimizers` subpackage and
  `l2co._src.{optimizer_schedule,stopping_criteria,opt_history,sampler}`,
  moved to `l2co-optimizers`. Nothing stored references them by path, so
  there are no shims. `l2co._src.tree_utils.combined_loss` /
  `combined_loss_with_rng` are not re-exported (no users).
- The `"rl2co"` branch of `OptimizationStep.name`; rl2co now registers it.
- The deprecated `l2co._src.optimizers.sub_optimizer` underscore aliases
  (`_SubOpt`, `_optstep_to_subopt`, `_resolve_popsize`,
  `_CONSTRUCTOR_HYPERPARAMETERS`), scheduled for removal in this version.
- `docs/register_optimizer.ipynb`, moved to `l2co-optimizers`.
- `l2co._src.optimizers.update_class._default_stop_fn` (superseded by
  `stop_fn=None`).
- `UpdateClass.evosax_random_search_update`. It was unreachable: evosax's
  `RandomSearch` normalizes to the `randomsearch` registry key, which l2co's
  own `random_search_update` shadowed. evosax's entry is now dropped from the
  registry explicitly. As a meta-optimizer menu entry, `randomsearch` now fails
  with "not recognized" instead of a `TypeError` about `sampling_fn` (it never
  built).

## [1.5.0] - 2026-09-01

### Added

- `l2co.optimizers.menu_loss_and_grad`, the menu-dispatched population
  evaluator, and the `GRAD_EVALUATION_WIDTH` constant naming how many rows a
  gradient generation evaluates. Given a menu it evaluates only the *active*
  optimizer's rows — its `popsize` value-only for a population optimizer, the
  single iterate with gradients for a gradient one — instead of the menu-wide
  `max(popsize)`. Exposes `branches` / `menu_to_branch` for introspecting how
  many distinct evaluation shapes a menu compiles to (`docs/adr/0011`).
- `UpdateClass.max_fevals_per_step` (static, defaults to `None`) and the
  `UpdateClass.fevals_bound` property resolving it to `popsize`. The bound is
  what an optimizer can bill in one iteration; it exceeds `popsize` only for
  line-search optimizers. Consumers that size a budget or normalize a
  per-evaluation cost must read `fevals_bound` rather than `popsize`.
- `l2co._src.optimizers.update_class.step_fevals`, the shared feval count for
  optax step functions, and
  `l2co._src.optimizers.lbfgs.DEFAULT_MAX_LINESEARCH_STEPS`.
- Three-phase parallel databank dump on `l2co.databank`: `dump_databank_template`,
  `dump_databank_task`, `dump_databank_finalize` and the `DumpPlan` hand-off
  between them. Decomposes `dump_databank` into a single-process store layout,
  an embarrassingly parallel per-task region write (one shard file per
  `(task, schedule)`, so separate processes never touch the same file), and a
  single-process finalize. `dump_databank` itself is unchanged and the two
  paths produce identical stores.
- `l2co.features.DERIVED_FEATURES`, mapping a feature name to a function of the
  task tag for schema features that are computed rather than stored (currently
  `deterministic`, read off the `noise` tag). `FeatureProcessor.transform`
  computes a derived feature when the key is absent, so an existing databank
  gains the feature with no trajectory re-runs.
- `seed_batch` on `l2co._src.performance.ertd` and `DataLoader.ertd_profile`,
  bounding the bootstrap's peak memory. Affects allocation only, never the
  result.
- `FeatureProcessor.unknown_as_absent` (static field, default `False`), the
  `unknown_as_absent=` keyword on `FeatureProcessor.init`, and the
  `FeatureProcessor.with_unknown_as_absent()` copier. Encodes a categorical or
  boolean value that is *not* in its vocabulary as an absent one — the
  feature's `impute` fill if it has one, otherwise the `ABSENT` row, otherwise
  a raise naming both — instead of raising. Leaves every vocabulary and the
  feature count untouched, so a model trained against the processor still
  loads; the default still raises. See `docs/adr/0015` for when to turn it on
  and what it does not buy (the `ABSENT` row it routes to receives no gradient
  on a training set where every task carries every feature).

### Changed

- **A task tag no longer has to spell a value the way the vocabulary does.**
  `FeatureProcessor.transform` folds boolean values onto the `"false"` /
  `"true"` keys from every route they arrive by — `True`, `np.True_`,
  `"True"`, `1`, `1.0`, `"1"`, a zero-dimensional JAX array — where previously
  only `True` and `"true"` encoded and everything else raised `KeyError`. The
  same fold runs between a numeric category and its string in both directions
  (`10` ↔ `"10"`), which is what a databank round-trip and every artifact
  written before vocabulary keys kept their type produce. Only spellings that
  name an existing key are accepted, so no fold can invent one, and
  case-folding stays confined to boolean vocabularies — a categorical schema
  may still list `"True"` and `"true"` as two categories (`docs/adr/0015`).
- `FeatureProcessor` vocabulary keys keep their type through `store` / `load`.
  A vocabulary is a JSON object and JSON object keys are strings, so a
  categorical feature keyed by the integers `2, 3, 5, 10` reloaded keyed by
  `"2", "3", "5", "10"` and stopped matching a live `Task.tag` — an artifact
  that encoded fine before it was stored and raised after. Affected
  vocabularies are now written a second time as `[[key, index], ...]`, where
  JSON keeps the difference; `reconstruct` prefers those. A processor whose
  keys are all strings — every processor built from a `'boolean'` or
  string-categorical schema — serializes to exactly the bytes it always did.
  A vocabulary key JSON cannot carry at all is now rejected by `to_dict`
  rather than written into an artifact that fails to load.
- A non-finite feature value counts as absent rather than as a value, in
  numeric and string form (`inf`, `-inf`, `"inf"`), joining `nan` / `None` /
  `""`. It therefore takes an `impute` fill and reads `false` on a
  `<name>__present` companion, instead of an infinity reaching a `Linear`. A
  processor with neither mechanism still passes an explicit null-ish value
  through untouched, so what an existing checkpoint encodes is unchanged.
- `FeatureProcessor.swap_names` rejects a mapping that sends two features to
  the same name instead of silently returning a processor with one feature
  fewer — a shorter feature vector than the `MixedEmbedding` built for it.
- `FeatureProcessor.init` accepts an array-valued categorical spec. The
  branch testing `spec == "continuous"` raised `ValueError: the truth value of
  an array ... is ambiguous` before reaching the categorical branch, so a
  schema whose categories came off a NumPy array could not be declared at all;
  its categories are now also unwrapped to the Python values they hold, since
  a vocabulary keyed by `np.int64` is one JSON cannot store.
- **The `"l2co"` meta-optimizer no longer evaluates its whole population every
  iteration.** `l2co_population_based_fn` used to call
  `vmapped_loss_and_grad` over all `max(popsize)` candidates unconditionally,
  so an Adam or L-BFGS iteration — which consumes a single point — paid a
  population-wide forward *and* backward pass, and a CMA-ES or PSO iteration
  paid a backward pass it discards. Function-evaluation counts were always
  correct, so no recorded metric was wrong; only wall-clock time was spent.
  On a Helmholtz PINN (`d=2971`, popsize 27) one iteration cost 1067 ms where
  a population iteration needs 219 ms and a gradient one 42 ms. Now 4.9x
  faster on population iterations, 25.3x on gradient ones, and 4.9x per
  realization under the batched driver (`docs/adr/0011`).
- **The `"l2co"` meta-optimizer is no longer bit-reproducible against earlier
  runs.** Gradient iterations now evaluate one candidate, where XLA takes a
  different code path than the wide pass and the gradient differs by 4.33e-06
  relative; Adam is chaotic enough that a stored convergence curve will
  visibly diverge over a few hundred iterations. Population iterations are
  also affected on some platforms, because dispatching per optimizer emits a
  different compiled graph than the old single-shape call and XLA may fuse it
  differently (identical on x86_64 Linux, up to one float32 ulp apart on
  macOS/arm64). Trajectories are semantically unchanged throughout — this is
  float32 last-bit noise, not a behaviour change.
- Population iterations of the `"l2co"` meta-optimizer now record all-`NaN`
  gradients in `OptHistory.grads`, matching what the native EvoSax factories
  already record. Unobservable on every configured path — `run_` discards
  gradients, and the opt-in `gradient_based_stopping_criteria` was already
  inapplicable to population optimizers.
- **`"lbfgs"` now bills one function evaluation per line-search trial**
  instead of a flat 1 per iteration, on both the deterministic and the
  stochastic wiring (`docs/adr/0010`, taking the decision `docs/adr/0006`
  deferred). Optimizer trajectories are unaffected — this is accounting only —
  but `fevals` and every metric derived from it change for `lbfgs`. The
  under-reporting was largest *after* convergence, where the line search
  exhausts its budget every iteration without making progress. `lbfgs` records
  written before this change under-report their cost.

### Fixed

- The derived `deterministic` feature read a `noise` value of `nan`, `inf` or
  `None` as *stochastic*, contradicting the documented "a missing `noise` key
  is read as deterministic". `nan` is how a dense databank label matrix
  represents "this suite does not tag noise", so every task in a store that
  merely contained a noisy suite encoded as stochastic. Those now read as
  deterministic, agreeing with the missing key; an uninterpretable non-null
  token (`"heavy"`) still reads as stochastic.

## [1.4.0] - 2026-06-09

### Added

- Public facade modules `l2co.data`, `l2co.models`, `l2co.optimization`,
  `l2co.features`, `l2co.history`, and `l2co.validation`. Every public symbol
  is now reachable through exactly one topical facade in addition to the
  flat `l2co` namespace.
- `py.typed` marker (PEP 561). The package now ships type information to
  consumers.
- `CHANGELOG.md` and `CITATION.cff`.
- Explicit `[build-system]` declaration (`hatchling`) and `[tool.hatch.build]`
  configuration in `pyproject.toml`.
- `tests/unit/` and `tests/integration/` split with a `slow` pytest marker.

### Changed

- `l2co.__version__` is now populated at runtime via
  `importlib.metadata.version("l2co")`. The version lives in
  `pyproject.toml` only.
- Internal split of `l2co._src.models`, `l2co._src.data_loader`, and
  `l2co._src.run_optimization` into subpackages. The qualified class names
  remain stable (`l2co._src.models.<Class>` etc.) so pickled checkpoints
  continue to load. Each old top-level `_src` module is preserved as a
  deprecation shim.
- `l2co._src.optimizerschedule` and `l2co._src.optimizationdynamics` renamed
  to `optimizer_schedule` and `optimization_dynamics`. Deprecation shims
  preserve the old import paths.
- Experiments now import only from the public API; no more reaches into
  `l2co._src.*`.
- CI workflows install with `[tool.uv.sources]` (the `--no-sources` flag is
  removed). Sibling deps are cloned to the expected paths in CI.

### Deprecated

- Importing from `l2co._src.optimizerschedule`, `l2co._src.optimizationdynamics`,
  `l2co._src.data_loader`, and `l2co._src.run_optimization`. Use the renamed
  modules or, preferably, the public facade (`l2co.<topic>`). Shims emit
  `DeprecationWarning` and will be removed in v2.0.

## [1.3.1]

- Baseline before the restructure work began.
