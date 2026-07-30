# Changelog

All notable changes to L2CO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **Gradient iterations of the `"l2co"` meta-optimizer are no longer
  bit-reproducible against earlier runs.** They now evaluate one row, where
  XLA takes a different code path than the wide pass and the gradient differs
  by 4.33e-06 relative. Trajectories are semantically unchanged, but Adam is
  chaotic enough that a stored convergence curve will visibly diverge over a
  few hundred iterations. Only menus containing a gradient optimizer are
  affected; population-only menus are bit-identical.
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
