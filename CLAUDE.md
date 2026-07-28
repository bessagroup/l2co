# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**L2CO (Learning to Choose Optimizers)** is a research framework that learns to *select* from a set of static optimizers (Adam, SGD, CMA-ES, etc.) based on problem features and optimization history, rather than designing a new optimizer from scratch. Built on JAX/Equinox. Supports Python 3.12 and 3.13.

The package uses a **src layout** (`src/l2co/`) and is managed with `uv` (see `uv.lock`). It depends on three sibling editable packages declared under `[tool.uv.sources]`: `l2co-tasks` (task definitions, exposes `Task` and `create_*_task` factories), `f3dasm`, and `bessaplots`.

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
| `l2co.databank` | `dump_databank`, `load_databank`, `load_databank_dataset`, `load_databank_performance`, `load_loss_raw`, `load_taskset`, `save_taskset`, `loss_preprocess`, `populated_pairs`, `populated_iterations`, `verify_databank`, `DatabankVerificationReport`, `VerificationFinding`, `databank_root`, `zarr_store_path`, `BUDGET_STEPS` | `_src/databank/` (`layout`, `schema`, `records`, `reader`, `writer`, `tasksets`, `verify`) — a consolidated-zarr databank: a whole dataset in one store (chunk per `(task, optimizer)`) for fast subset reads at scale. `loss_history` is pre-interpolated to `BUDGET_STEPS`; the raw variable-length histories live in a `loss_history_raw/` netCDF sidecar and named selections under `tasksets/`. `records.py` holds the shared f3dasm/ERT pipeline (`_iter_task_runs`, `_build_task_records`, `loss_preprocess`) |
| `l2co.optimization` | `RolloutWrapper`, `RunOptimization`, `RunState`, `UpdateClass`, `OptimizationStep`, `reset`, `run`, `evaluate` | `_src/rollout/` (`rollout_wrapper`, `dataset_generation`), `_src/run_state`, `_src/update_class` (orchestration: `init_`/`step`/`run_`/`batch_run_`/`evaluate`), `_src/optimizers/update_class` (the `UpdateClass` adapter + `optax_update`/`evosax_*_update` classmethods), `_src/optimizer_schedule` |
| `l2co.features` | `FeatureProcessor`, `MixedEmbedding`, `preprocessing_factory` | `_src/task_features`, `_src/preprocessing` |
| `l2co.history` | `HistoryState`, `OptHistory`, `BatchState` | `_src/history_state`, `_src/opt_history`, `_src/batching` |
| `l2co.optimizers` | **The optimizer-authoring surface** (`docs/adr/0007`) — registry (`optimizer_mapping`, `optimizers`, `register_optimizer`), `UpdateClass`, contract types (`InitFunction`, `StepFunction`, `StopFunction`, `InputParameters`, `OptState`, `SamplerFunction`, `PopSize`), loss helpers (`OptHistory`, `vmapped_loss*`), sub-optimizer primitives (`SubOpt`, `grad_sub_optimizer`, `pop_sub_optimizer`, `optstep_to_subopt`, `resolve_popsize`, `l2co_native_optax`, `l2co_native_evosax`, `CONSTRUCTOR_HYPERPARAMETERS`), base factories (`normalized_optax_normal`, `normalized_optax_from_state`, `random_search_update`), built-ins (`SHADE`, `shade_update`, `shade_popsize`, `lbfgs_update`, `lbfgs_per_eval_key`, `scale_by_zoom_linesearch_per_eval_key`), `create_schedules_experimentdata` | `_src/optimizers/` (`update_class`, `optax_implementations`, `evosax_implementations`, `random_search`, `sub_optimizer`, `mapping`, `experimentdata`, `utils` — incl. `variable_popsize` / `variable_popsize_even`), plus `_src/typing`, `_src/opt_history`, `_src/tree_utils`. Downstream packages (rl2co, agentic-l2co) must import from this facade, never `l2co._src.*`. The `sub_optimizer` symbols were renamed from underscore names; the old names survive as deprecated shims until 1.6.0 |
| `l2co.validation` | `validation_fn`, `normalize_budget_grid` | `_src/validation` |
| `l2co.naming` | xarray dimension/coord constants (`BUDGET`, `REALIZATION`, `PROBLEM`, `SCHEDULE`, `OPT_CHOICE`, …) + typing aliases (`InputParameters`, `OptState`, `SamplerFunction`) | `_src/naming`, `_src/typing` |
| `l2co.budget` | `fixed_budget`, `based_on_dimensionality` | `_src/budget` |
| `l2co.sampling` | `random_sampling`, `grid_sampling`, `normal_sampling`, `xavier_sampling`, `constant_sampling`, `get_sampler` | `_src/sampler` |
| `l2co.plotting` | `plot_ertd`, `plot_ertd_profile`, `plot_metrics`, `plot_r2`, `plot_update_step`, `plot_2d/3d_loss_landscape*`, `generate_report`, `set_size`, `bbob`, `cec2005`, `get_schedule_styles`, … | `_src/visualize/` (`bbob_plot`, `cec2005_plot`, `colors`, `plotting`) |

`Task` is re-exported from `l2co` for convenience; the `create_*_task` factories (`create_bbob_task`, `create_cec2005_task`, `create_pde_task`, …) live in the sibling `l2co_tasks` package and must be imported from there directly.

`RunState` itself lives in `_src/run_state.py` and also provides `reset_from_stage0`, `batch_run`, and `batch_evaluate` (not re-exported through the facade — use the `RunState` methods or import from `_src` only inside the library).

### JAX/Equinox Patterns

- All models are `eqx.Module` subclasses; configuration fields use `eqx.field(static=True)`.
- Batching via `eqx.filter_vmap()`; gradients via `eqx.filter_value_and_grad()`.
- Serialization uses `to_dict()` / `reconstruct()` pattern on models (see `Stage0Loader` / `Stage1Loader`). The `_src/models/` package keeps qualified paths like `l2co._src.models.L2COModel` stable so older pickles still load after the split.
- `l2co._src.tree_utils` contains helpers for JAX pytree operations used throughout (also re-exported as `l2co.tree_utils`).
- When re-initializing a freshly constructed Equinox module's weights, reuse the *same* PRNG key the constructor used rather than splitting a new subkey.
- Default float dtype is `jnp.float32` and integer counters (`update_step`, `fevals`, `iterations`) use plain `int` / `jnp.int32` — keep dtypes consistent across modules to avoid silent JIT recompilations.

### Data Structures

xarray `Dataset`s with named dimensions defined in `l2co.naming` (e.g., `BUDGET`, `REALIZATION`, `PROBLEM`, `SCHEDULE`, `OPT_CHOICE`, `OPT_HISTORY`, `UPDATE_STEP`, `TRAJECTORY`) are the canonical data format for optimization trajectories. Avoid bare NumPy arrays when working with trajectory data. The standardized optimizer name for random search is `"randomsearch"` (no separator).

### Tests

`tests/` is split into `tests/unit/` (fast — dataloader/report helpers, `HistoryState` cursor/front-fill, databank round-trip, even-popsize rounding, public-import guard) and `tests/integration/` (`test_run_optimization.py`, exercises full JAX rollouts; many cases carry `@pytest.mark.slow`).

## Related repositories

Part of the L2CO ecosystem (Bessa Research Group). These repositories work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
