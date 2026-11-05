# Experiments

The L2CO experimental pipeline consists of three stages:

1. **Dataset creation** -- run an algorithm portfolio on benchmark problems and record optimization trajectories (RTD data)
2. **Stage 0 training** -- train $\mathcal{M}_\text{first}$, a cold-start algorithm selection model that maps task features $\phi$ to predicted RTD values
3. **Stage 1 training** -- train $\mathcal{M}_\text{second}$, a trajectory-aware reassessment model (LSTM encoder + MLP) that refines optimizer selection during optimization

All experiments use [Hydra](https://hydra.cc/) for configuration management and [f3dasm](https://github.com/bessagroup/f3dasm) for pipeline orchestration.

## Quick overview

| Experiment | Description | Entry point | Hydra config |
|---|---|---|---|
| `create_optimization_dataset` | Run algorithm portfolio on benchmark tasks, collect RTD trajectories | `create_optimization_dataset/main.py` | `conf/config.yaml` |
| `l2co_train_stage0` | Train $\mathcal{M}_\text{first}$: cold-start algorithm selection (sklearn regressors) | `l2co_train_stage0/main.py` | `conf/config_l2co_train_stage0.yaml` |
| `l2co_train_stage1` | Train $\mathcal{M}_\text{second}$: trajectory-aware reassessment (LSTM + MLP) | `l2co_train_stage1/main.py` | `conf/config_l2co_train_stage1.yaml` |
| `generate_figures.py` | Generate ERTD plots and PDF report from results | `generate_figures.py` | -- |
| `submit_job.sh` | SLURM wrapper for HPC cluster submission | `submit_job.sh` | -- |

## Data flow

```
create_optimization_dataset        l2co_train_stage0          l2co_train_stage1
 ┌──────────────────────┐      ┌───────────────────┐      ┌───────────────────┐
 │ benchmark tasks      │      │ load trajectories │      │ load trajectories │
 │   + algorithm        │─────>│ extract features   │─────>│ build LSTM +      │
 │     portfolio        │      │ train sklearn      │      │   embeddings +    │
 │   = RTD trajectories │      │   regressor        │      │   MLP head        │
 └──────────────────────┘      └───────────────────┘      └───────────────────┘
         .odata files              M_first models             M_second models
```

---

## `create_optimization_dataset`

Runs each optimizer in the algorithm portfolio on every task instance across multiple independent realizations, then combines the results into an xarray `Dataset`.

### Workflow

The f3dasm pipeline has three steps (and an optional fourth):

1. **create** -- Sample task instances (e.g. BBOB functions x dimensions), build optimizer schedule objects, and grid-sample all task-optimizer combinations.
2. **run** -- Execute each task-optimizer pair for `n_realizations` independent runs (different random seeds). Supports parallel and SLURM array jobs.
3. **post** -- Collect per-run NetCDF loss histories and combine them into a single xarray `Dataset` (`entire_dataset.odata`).
4. **report** *(optional)* -- Generate ERTD plots from the combined dataset.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_realizations` | `25` | Number of independent optimization runs per task (different random seeds) |
| `seed` | `469` | Global random seed for reproducibility |
| `budget_steps` | `1000` | Number of discrete timesteps for interpolating trajectories into the combined dataset |
| `verbose` | `False` | Print per-job progress to stdout |
| `clear_jax_cache` | `True` | Clear JAX compilation cache between jobs to limit memory growth |
| `save_input_parameters` | `False` | Store initial sample points alongside the loss history |
| `rootdir` | `null` | Root directory for pipeline outputs (`null` = `cwd/outputs`) |
| `mode` | `local` | Execution mode: `local`, `sequential`, or `cluster` |

---

## `l2co_train_stage0`

Trains the first-decision model $\mathcal{M}_\text{first}$: a supervised regressor that maps task feature vectors $\phi$ to RTD values of the algorithm portfolio, enabling cold-start optimizer selection before any function evaluations.

### Workflow

1. **create** -- Set up one experiment row per sklearn model configuration.
2. **run** -- For each model: load the pre-generated optimization data, extract feature vectors via `FeatureProcessor`, train a supervised regression model mapping $\phi \to \text{RTD}$, and evaluate $R^2$ on the test set.
3. **post** -- Collect all trained models.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `t` | `0.01` | Softmax temperature for strategy selection (lower = more greedy) |
| `t_end` | `100000` | Final temperature at end of annealing schedule |
| `path_train` | -- | Path to the training dataset (output of `create_optimization_dataset`) |
| `path_test` | `${path_train}` | Path to the test dataset (defaults to same as training) |
| `plot_strategy` | `False` | Generate per-model strategy plots after training |
| `rootdir` | `null` | Root directory for pipeline outputs |
| `mode` | `local` | Execution mode |

---

## `l2co_train_stage1`

Trains the second-decision model $\mathcal{M}_\text{second}$: a neural network that incorporates optimization trajectory information to dynamically reassess the optimizer choice during a run.

### Workflow

1. **create** -- Set up one experiment row per architecture configuration (grid search over hyperparameters when using `stage1_models=all`).
2. **run** -- For each architecture: load pre-generated data, build the model (LSTM encoder + mixed embeddings + MLP policy head), train with Adam, and evaluate $R^2$ on the test set.
3. **post** -- Collect all trained models.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `t` | `0.01` | Softmax temperature for strategy selection |
| `t_end` | `100000` | Final temperature at end of annealing schedule |
| `path_train` | -- | Path to the training dataset |
| `path_test` | `${path_train}` | Path to the test dataset |
| `seed` | `0` | JAX random seed for reproducibility |
| `learning_rate` | `1e-2` | Adam learning rate |
| `num_steps` | `1000` | Number of gradient update steps |
| `verbose` | `True` | Print training loss each step |
| `rootdir` | `null` | Root directory for pipeline outputs |
| `mode` | `local` | Execution mode |

### Model architecture parameters

Nested under `stage1_models`:

| Parameter | Default | Description |
|---|---|---|
| `history_encoder.hidden_size` | `16` | LSTM hidden state dimension |
| `history_encoder.out_size` | `4` | LSTM output (embedding) dimension |
| `embedding.dim` | `4` | Per-category embedding dimension |
| `embedding.out_size` | `4` | Embedding MLP output dimension |
| `mlp.width_size` | `16` | Hidden layer width of the policy MLP |
| `mlp.depth` | `3` | Number of hidden layers in the policy MLP |

---

## Supporting scripts

### `generate_figures.py`

Loads a completed optimization dataset via `l2co.DataLoader`, then generates:

- ERTD plots per BBOB function category (separable, low conditioning, etc.)
- ERTD plots per dimensionality (2-D through 40-D)
- An aggregated ERTD profile

Saves all figures to `./plots/` and produces a PDF report via `bessaplots`.

```bash
python generate_figures.py /path/to/datasets
```

The argument is the folder that holds each dataset's `entire_dataset` and the stage-0 model.

### `submit_job.sh`

SLURM job submission wrapper for running experiments on HPC clusters.

```bash
sbatch submit_job.sh experiment=<name> [hydra_overrides...]
```

The script extracts the `experiment=` argument, sets `cluster=brown` and `mode=slurm`, and forwards any remaining arguments as Hydra overrides to `main.py`. SLURM defaults: 30 min walltime, 8 GB memory, 1 CPU, `batch` partition.

---

## Hydra configuration

All config files live under `conf/`. Each experiment selects its defaults from the config groups below.

```
conf/
├── config.yaml                        # create_optimization_dataset defaults
├── config_l2co_train_stage0.yaml      # Stage 0 training defaults
├── config_l2co_train_stage1.yaml      # Stage 1 training defaults
├── tasks/            # benchmark problem suites
├── optimizers/       # algorithm portfolios
├── budgets/          # evaluation budget functions
├── samplers/         # initial point sampling strategies
├── stage0_models/    # sklearn model configurations
├── stage1_models/    # neural net architecture configurations
├── cluster/          # execution environment (local / SLURM)
└── experimentdata/   # f3dasm domain definition
```

### Config groups

#### `tasks/` -- benchmark problem suites

| Config | Description |
|---|---|
| `bbob` | Full BBOB suite: 24 functions x 6 dimensions (2, 3, 5, 10, 20, 40) |
| `bbob_single` | Single BBOB function (override `fn_name` and `fn_dim` via CLI) |
| `cec2005` | CEC 2005 suite: 25 functions x 3 dimensions (10, 30, 50) |
| `gaussian_classification` | Neural net classification: 11 seeds x 3 hidden sizes x 3 depths |

#### `optimizers/` -- algorithm portfolios

| Config | Description |
|---|---|
| `medium` | 4 baseline optimizers: Sep-CMA-ES, Adam (lr=1e-2), L-BFGS, PSO |
| `l2co` | Learned L2CO optimizer (requires trained Stage 0 + Stage 1 model paths) |
| `medium_plus_l2co` | The 4 baselines + L2CO |
| `gaussian_classification` | Adam (lr=3e-1), Sep-CMA-ES, L-BFGS, PSO |
| `gaussian_classification_plus_l2co` | Gaussian classification baselines + L2CO |
| `lr_sweep_pde` | Adam learning rate sweep: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 |

#### `budgets/` -- evaluation budget functions

| Config | Description |
|---|---|
| `fixed` | Constant budget of 1000 evaluations |
| `based_on_dim` | Budget scales with dimensionality: 1000 x *d* evaluations |
| `gaussian_classification` | Constant budget of 100 evaluations |

#### `samplers/` -- initial point sampling

| Config | Description |
|---|---|
| `normal` | Isotropic Gaussian (mean=0, std=1) |
| `random` | Uniform random within task bounds |
| `grid` | Uniform grid (100 points per dimension) within task bounds |
| `xavier` | Xavier/Glorot uniform initialization |

#### `stage0_models/` -- sklearn model configurations

| Config | Description |
|---|---|
| `best` | Single `ExtraTreesRegressor` (best performing from hyperparameter search) |
| `all` | 51 sklearn regressors (trees, linear models, SVMs, KNN, MLPs) |

#### `stage1_models/` -- neural net architecture configurations

| Config | Description |
|---|---|
| `best` | Single best architecture (`hidden_size=32`, `width_size=32`, `depth=3`) |
| `all` | 27 configurations: `hidden_size` x `width_size` x `depth` in {8, 16, 32} x {8, 16, 32} x {2, 3, 4} |

#### `cluster/` -- execution environment

| Config | Description |
|---|---|
| `local` | Local execution (no SLURM) |
| `brown` | OSCAR cluster (Brown University) -- SLURM `batch` partition |
| `delft` | DelftBlue cluster (TU Delft) -- SLURM `compute` partition |

### Overriding configuration from the CLI

Hydra allows swapping entire config groups or overriding individual parameters:

```bash
# Use BBOB benchmark with dimension-based budget
python create_optimization_dataset/main.py tasks=bbob budgets=based_on_dim

# Train Stage 0 with all 51 sklearn models
python l2co_train_stage0/main.py stage0_models=all

# Train Stage 1 on cluster with architecture grid search
python l2co_train_stage1/main.py stage1_models=all cluster=brown

# Override individual parameters
python create_optimization_dataset/main.py n_realizations=50 seed=42

# Run a single BBOB function
python create_optimization_dataset/main.py tasks=bbob_single tasks.fn_name=sphere tasks.fn_dim=10
```
