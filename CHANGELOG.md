# Changelog

All notable changes to L2CO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **BREAKING:** `RolloutWrapper.init_from_stage0`, `RolloutWrapper.evaluate`
  and `RolloutWrapper.run`, plus `reset_from_stage0` in `_src/run_state.py`.
  They only served the eager two-stage driver for agentic-l2co, which
  l2co_experiments has dropped (bessagroup/l2co_experiments#45). The L2CO
  strategy reaches stage 1 through the registry bridge
  (`docs/adr/0002`). Use `RolloutWrapper.batch_evaluate`, or the
  free functions `l2co.optimization.evaluate` / `run`, which stay.

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
