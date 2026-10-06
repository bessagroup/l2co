# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**L2CO (Learning to Choose Optimizers)** is a research framework that learns to *select* from a set of static optimizers (Adam, SGD, CMA-ES, etc.) based on problem features and optimization history, rather than designing a new optimizer from scratch. Built on JAX/Equinox. Supports Python 3.12 and 3.13.

The package uses a **src layout** (`src/l2co/`) and is managed with `uv` (see `uv.lock`). It depends on four sibling editable packages declared under `[tool.uv.sources]`: `l2co-tasks` (task definitions, exposes `Task` and `create_*_task` factories), `l2co-optimizers` (the bare optimizers — registry, `UpdateClass`, `OptimizationStep`, handshake and state transfer; see below), `f3dasm`, and `bessaplots`.

## Commands

```bash
# Install / sync the environment (uv-managed)
uv sync

# Run all tests (unit + integration; slow rollouts are marked)
make test            # → uv run pytest
uv run pytest -m "not slow"           # exclude full JAX rollouts
uv run pytest tests/integration/test_run_optimization.py -k "test_pde"

# Lint
make lint            # → uv run ruff check
uv run ruff check --fix

# Build docs / package
make docs            # → uv run mkdocs build → site/
make build           # → uv build
```

**Ruff config** (`pyproject.toml`): line length 79. Lint excludes `tests/`, `experiments/*`, `notebooks/`, `docs/source`. Selects E, W, F, I, B, UP. Ignores E226, E402, E721, E3, E731, C901, F722. `notebooks/` and `docs/*.ipynb` are also globally excluded via `extend-exclude`.

**Pytest marker**: `slow` — tests that exercise full JAX rollouts. Excluded from default PR runs via `-m "not slow"`.

## Architecture

### Public facade vs. `_src/`

`src/l2co/` is a **stable public facade**: top-level modules (`data.py`, `models.py`, `optimization.py`, `features.py`, `history.py`, `optimizers.py`, `plotting.py`, `sampling.py`, `validation.py`, `budget.py`, `naming.py`) re-export from the private `src/l2co/_src/` implementation. **All external users — including `experiments/` — must import only from the facade**, never from `l2co._src.*`. `tests/unit/test_experiments_no_private_imports.py` enforces this via AST walk and will fail CI if violated.

Inside the package every import is absolute (`from l2co._src.data import DataLoader`, never `from .data import ...`); ruff's `TID252` with `ban-relative-imports = "all"` enforces this.

When adding a new public symbol: implement in `_src/`, then add to the matching facade module (`l2co.<area>`) **and** to `src/l2co/__init__.py` if it should be importable as `l2co.X`.

### Two-Stage Optimizer Selection

```
Stage 0 (ScikitStage0Model / EquinoxStage0Model — both subclass Stage0Model)
  Input:  Task features only
  Output: Initial optimizer logits

Stage 1 (EquinoxStage1Model — subclasses Stage1Model)
  Input:  Task features + optimizer features + optimization history
  Output: Per-step optimizer selection logits
  Key components:
    - LSTM_Encoder: encodes optimization trajectory
    - MixedEmbedding: handles categorical + continuous features
    - MLP: final classification head
```

`L2COModel` combines both stages and is the main inference entry point. `Stage0Loader` / `Stage1Loader` handle (de)serialization. All of these are importable as `l2co.<name>` or `l2co.models.<name>`.

### Core Data Flow

1. **Dataset creation** (`experiments/create_optimization_dataset/`) — run an algorithm portfolio on benchmark problems (BBOB, CEC2005, PDE) via `RunOptimization` (a `DataGenerator`), collect RTD trajectories as xarray `Dataset`s.
2. **Data loading** (`l2co.data`, backed by `l2co._src.data/` and `l2co._src.optimization_dynamics`) — a single `DataLoader` holds both the loss trajectories and the ERT performance metrics and owns all the operations (selection, feature extraction, `to_traj_and_stair`, `ertd_profile`, `to_params`, `report`). It loads xarray data **lazily** (zarr-backed; operations stay lazy w.r.t. the heavy `loss_history` variables), extracts features via `FeatureProcessor`, and splits train/val. `OptimizationDynamics` is the learning dataset `to_params` produces.
3. **Training** — Stage 0 (`experiments/l2co_train_stage0/`) trains sklearn regressors or Equinox MLPs; Stage 1 (`experiments/l2co_train_stage1/`) trains LSTM+MLP with MSE loss (`mse_loss` / `train_equinox` in `l2co.models`).
4. **Inference** (`l2co.optimization`) — `RolloutWrapper` applies model predictions, calls the selected optimizer via `UpdateClass`, accumulates `HistoryState` / `OptHistory`.

See `experiments/README.md` for full pipeline documentation including Hydra configuration.

### Public surface (`l2co.<module>`)

| Facade module | Re-exports | Backed by |
|---|---|---|
| `l2co.models` | `AbstractModel`, `Stage0Model`, `ScikitStage0Model`, `EquinoxStage0Model`, `Stage1Model`, `EquinoxStage1Model`, `L2COModel`, `LSTM_Encoder`, `MLP`, `Stage0Loader`, `Stage1Loader`, `mse_loss`, `train_equinox` | `_src/models/` (split into `base`, `components`, `stage0`, `stage1`, `l2co_model`, `training`) |
| `l2co.data` | `DataLoader`, `create_combined_odata` | `_src/data/` (`loader`, `baselines`, `report_helpers`) + `_src/data_validation` |
| `l2co.databank` | `dump_databank`, `dump_runs`, `load_databank`, `load_databank_dataset`, `load_databank_performance`, `load_loss_raw`, `load_taskset`, `save_taskset`, `loss_preprocess`, `populated_pairs`, `populated_iterations`, `verify_databank`, `DatabankVerificationReport`, `VerificationFinding`, `databank_root`, `zarr_store_path`, `BUDGET_STEPS` | `_src/databank/` (`layout`, `schema`, `records`, `reader`, `writer`, `tasksets`, `verify`) — a consolidated-zarr databank: a whole dataset in one store (chunk per `(task, optimizer)`) for fast subset reads at scale. `loss_history` is pre-interpolated to `BUDGET_STEPS`; the raw variable-length histories live in a `loss_history_raw/` netCDF sidecar and named selections under `tasksets/`. `records.py` holds the shared f3dasm/ERT pipeline (`_iter_task_runs`, `_build_task_records`, `loss_preprocess`). `dump_databank` is f3dasm bookkeeping (three `ExperimentData` tables -> a `runs` list) on top of `dump_runs`, which drives the three phases serially; a caller that already holds runs uses the latter |
| `l2co.optimization` | `RolloutWrapper`, `RunOptimization`, `RunState`, `UpdateClass`, `OptimizationStep`, `reset`, `run`, `evaluate`, `init_run_state` | `_src/rollout/` (`rollout_wrapper`, `dataset_generation`), `l2co_optimizers` (`UpdateClass` — whose methods are the run loop — `RunState`, `reset`, `run`, `evaluate` and `OptimizationStep`, re-exported; `docs/adr/0017`), `_src/task_bridge` (`init_run_state`; `docs/adr/0018`) |
| `l2co.features` | `FeatureProcessor`, `MixedEmbedding`, `preprocessing_factory`, `DERIVED_FEATURES` | `_src/task_features`, `_src/preprocessing`. `DERIVED_FEATURES` maps a feature name to a function of the task tag, for schema features that are *computed* rather than stored (currently `deterministic`, read off the `noise` tag). Deriving keeps `Task.hash` — and so every databank `taskID` — unchanged, letting an existing store gain a feature with no trajectory re-runs; a tag that already carries the key wins over the derivation |
| `l2co.history` | `HistoryState`, `OptHistory`, `BatchState`, `history_to_xarray`, `history_to_xarray_realizations`, `history_to_dataloader` | the three states are re-exported from `l2co_optimizers` (`docs/adr/0017`); the `history_to_*` exports live in `_src/history_state` because they need xarray, the ERT and `Task` |
| `l2co.optimizers` | **The optimizer-authoring surface** (`docs/adr/0007`) — the *Task-level* registry (`optimizer_mapping`, `register_optimizer`, `meta_optimizers`; from `_src/task_bridge`, `docs/adr/0018` — the bare `optimizers` dict is imported from `l2co_optimizers`), `UpdateClass`, contract types (`InitFunction`, `StepFunction`, `StopFunction`, `AskFunction`, `TransferReadFunction`, `TransferWriteFunction`, `InputParameters`, `OptState`, `SamplerFunction`, `PopSize`), loss helpers (`OptHistory`, `vmapped_loss*`, plus `menu_loss_and_grad` — the menu-dispatched evaluator that spends only the *active* optimizer's evaluation width instead of the menu-wide generation width, with branches deduplicated by `(kind, width)` — for compile time and introspection only, *not* runtime: XLA's CSE already collapses identical branch bodies, as `docs/adr/0011` records with measurements), sub-optimizer primitives (`SubOpt`, `grad_sub_optimizer`, `pop_sub_optimizer`, `optstep_to_subopt`, `resolve_popsize`, `CONSTRUCTOR_HYPERPARAMETERS`), handshake policy (`HandshakePolicy`, `HANDSHAKE_POLICY`, `DEFAULT_HANDSHAKE_POLICY`, `handshake_policy_for` — the single source of truth for which handshake an optimizer needs on a switch, shared with rl2co's env and wrappers; `docs/adr/0009`), state transfer (`TransferBundle`, `build_transfer_fns`, the `FAMILY_*` / `CONF_*` constants — *what information* crosses a switch, as opposed to which population: one reader and one writer per optimizer rather than a pairwise table, carrying a search scale so a re-selected optimizer resumes at the current scale instead of its constructor's. `UpdateClass` grows `family` / `ask_fn` / `transfer_read_fn` / `transfer_write_fn` for it, and an optimizer whose `TransferSpec.own_ask` is set regenerates its own population through `ask_fn` rather than being handed one — every `'distribution'` algorithm plus the mutation GAs (`mr15ga`, `samrga`, `gesmrga`), because their next update reads state the handshake just wrote: cached draws for the former, the repaired `state.fitness` baseline for the latter. That repair *claims a measurement* about the handed-over point, and makes it unconditionally: whether a loss is deterministic is not something a handshake may read, and the bias it adds is one MR15-GA's own `(µ+λ)` selection already has; `docs/adr/0012`, `docs/adr/0013`, `docs/adr/0014`), built-ins (`normalized_optax_normal`, `shade_update`, `shade_popsize`), `create_schedules_experimentdata`. Only what a sibling package uses, plus the bare-optimizer authoring surface, is public in `l2co-optimizers`; the per-library factories, `SHADE`, the L-BFGS building blocks, the `l2co_native_*` tables and the handshake/transfer tables are private there, so they are not re-exported here either | **Re-exports `l2co_optimizers`** (`docs/adr/0016`), except the registry (`_src/task_bridge`, `docs/adr/0018`) — the same objects, implemented in the sibling `l2co-optimizers` package (`../l2co-optimizers`, its own `CLAUDE.md`). Nothing optimizer-side is implemented in l2co any more. Downstream packages (rl2co, agentic-l2co) import from this facade or from `l2co_optimizers`, never `l2co._src.*`. The deprecated `sub_optimizer` underscore aliases were removed in 1.6.0 |
| `l2co.validation` | `validation_fn`, `normalize_budget_grid` | `_src/validation` |
| `l2co.naming` | xarray dimension/coord constants (`BUDGET`, `REALIZATION`, `PROBLEM`, `SCHEDULE`, `OPT_CHOICE`, …) + typing aliases (`InputParameters`, `OptState`, `SamplerFunction`) | `_src/naming`, `_src/typing` |
| `l2co.budget` | `fixed_budget`, `based_on_dimensionality` | `_src/budget` |
| `l2co.sampling` | `random_sampling`, `grid_sampling`, `normal_sampling`, `xavier_sampling`, `constant_sampling`, `get_sampler` | re-exported from `l2co_optimizers` |
| `l2co.plotting` | `plot_ertd`, `plot_ertd_profile`, `plot_metrics`, `plot_r2`, `plot_update_step`, `plot_2d/3d_loss_landscape*`, `generate_report`, `set_size`, `bbob`, `cec2005`, `get_schedule_styles`, … | `_src/visualize/` (`bbob_plot`, `cec2005_plot`, `colors`, `plotting`) |

### The l2co-optimizers boundary (`docs/adr/0016`)

l2co is the **bridge** between `l2co-tasks` and `l2co-optimizers`, which never import each other:
- **The task stops here** (`docs/adr/0018`). `l2co-optimizers` has no task type: bare factories take `model`, `loss_fn`, `pass_rng` as keywords. `_src/task_bridge.py` is the only place a `Task` meets an `OptimizationStep`: `optimizer_mapping(name)` returns a `task=`-taking factory (meta-optimizers as registered; bare ones wrapped to unpack the task), `register_optimizer` / `meta_optimizers` hold the meta-optimizers (raises on a bare name), and `init_run_state(optimizer, task, bounded, key)` resolves and builds the `RunState`. Anything that holds a `Task` and needs an optimizer goes through these; `optstep_to_subopt` / `resolve_popsize` are called with `model=task.model, loss_fn=task.loss_fn, pass_rng=task.pass_rng`.
- **The run loop is in `l2co-optimizers`** (`docs/adr/0017`, superseding that half of 0016): `UpdateClass.init_state` / `step` / `run` / `batch_run`, plus `BatchState`, the `HistoryState` buffer and `Carry`. So is `RunState` with `reset` / `run` / `batch_run` / `batch_evaluate` and `evaluate`, which just call those methods; `RunState.init` takes a built `UpdateClass` plus `model`, `dataset`, `batch_size` (use `init_run_state` from a `Task`). `batch_run`'s driver choice (fused, or sequential for `sequential_realizations` and random search) is the optimizer's.
- **What stays here:** `RolloutWrapper` (which builds a `RunState` from a `Task` via `init_run_state`) and `HistoryState`'s exports (`history_to_xarray` / `history_to_xarray_realizations` / `history_to_dataloader`).
- **So does every meta-optimizer:** `strategy_wrapper`, `meta_optimizer`, the models.

Import optimizer-side names from the `l2co_optimizers` root only, never from its `_src`. When something l2co needs isn't exported there, export it in l2co-optimizers first.

Meta-optimizers register a **schedule namer** next to their factory, because `OptimizationStep.name` is a databank key and `hash` derives from it. `meta_optimizer.py` registers `"l2co"` (`l2co_schedule_name`), and rl2co registers `"rl2co"`.

`Task` is re-exported from `l2co` for convenience; the `create_*_task` factories (`create_bbob_task`, `create_cec2005_task`, `create_pde_task`, …) live in the sibling `l2co_tasks` package and must be imported from there directly.

`RunState` lives in `l2co_optimizers`; l2co's facade re-exports `RunState`, `reset`, `run` and `evaluate`. Inside the library, import `batch_reset` / `batch_evaluate` from the `l2co_optimizers` root (`batch_run` is private there; use `UpdateClass.batch_run`).

### JAX/Equinox Patterns

- All models are `eqx.Module` subclasses; configuration fields use `eqx.field(static=True)`.
- Batching via `eqx.filter_vmap()`; gradients via `eqx.filter_value_and_grad()`.
- Serialization uses `to_dict()` / `reconstruct()` pattern on models (see `Stage0Loader` / `Stage1Loader`). The `_src/models/` package keeps qualified paths like `l2co._src.models.L2COModel` stable so older pickles still load after the split.
- `l2co._src.tree_utils` contains helpers for JAX pytree operations used throughout (also re-exported as `l2co.tree_utils`). Its `vmapped_loss*` evaluators are re-exported from `l2co_optimizers`.
- When re-initializing a freshly constructed Equinox module's weights, reuse the *same* PRNG key the constructor used rather than splitting a new subkey.
- Default float dtype is `jnp.float32` and integer counters (`update_step`, `fevals`, `iterations`) use plain `int` / `jnp.int32` — keep dtypes consistent across modules to avoid silent JIT recompilations.

### Data Structures

xarray `Dataset`s with named dimensions defined in `l2co.naming` (e.g., `BUDGET`, `REALIZATION`, `PROBLEM`, `SCHEDULE`, `OPT_CHOICE`, `OPT_HISTORY`, `UPDATE_STEP`, `TRAJECTORY`) are the canonical data format for optimization trajectories. Avoid bare NumPy arrays when working with trajectory data. The standardized optimizer name for random search is `"randomsearch"` (no separator).

### Tests

`tests/` is split into `tests/unit/` (fast — dataloader/report helpers, databank round-trip, schedule namer, public-import guard; the optimizer unit tests, and since ADR 0017 the run-loop and `HistoryState` buffer tests, live in `l2co-optimizers`) and `tests/integration/` (`test_run_optimization.py`, exercises full JAX rollouts; many cases carry `@pytest.mark.slow`).

## Related repositories

Part of the L2CO ecosystem (Bessa Research Group). These repositories work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
