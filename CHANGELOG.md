# Changelog

All notable changes to L2CO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `UpdateClass.max_fevals_per_step` (static, defaults to `None`) and the
  `UpdateClass.fevals_bound` property resolving it to `popsize`. The bound is
  what an optimizer can bill in one iteration; it exceeds `popsize` only for
  line-search optimizers. Consumers that size a budget or normalize a
  per-evaluation cost must read `fevals_bound` rather than `popsize`.
- `l2co._src.optimizers.update_class.step_fevals`, the shared feval count for
  optax step functions, and
  `l2co._src.optimizers.lbfgs.DEFAULT_MAX_LINESEARCH_STEPS`.
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
