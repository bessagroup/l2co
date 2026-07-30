# Changelog

All notable changes to L2CO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
