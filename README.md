# Learning to Choose Optimizers (l2co)

<div align="center">
  <img src="img/logo.png" alt="BBOB functions 3D overview" width="50%">
</div>

[![GitHub license](https://img.shields.io/badge/license-BSD-blue)](https://github.com/bessagroup/l2co)


**Docs**
| [**GitHub**](https://github.com/bessagroup/l2co)
| **PyPI**

> Some links are not yet available, as the package is still under a private repository.

***

## Summary

**L2CO** is a Python project accompanying the paper 'Learning to Choose Optimizers'. 

## Statement of need

Optimization plays a central role in data-driven design. However, choosing an appropriate optimization algorithm for a specific task is often challenging, as it depends on the structure of the optimization problem and practical limitations such as the computational budget. To tackle this challenge, the field has seen a shift from manually designing optimizers toward using machine learning methods that attempt to learn the optimization process itself. Despite significant progress, existing approaches frequently exhibit limited generalization to new distributions and typically require retraining to transfer their knowledge.

We introduce a new approach called ''Learning to Choose Optimizers'' (L2CO). In our approach, a meta-learner first selects an optimizer before any function evaluations---based solely on available problem characteristics---and then reassesses that choice using a unified model that jointly encodes problem context, optimizer identity, and the observed optimization trajectory. By restricting selection to well-established algorithms, L2CO avoids the convergence issues of learned update rules while retaining the adaptability of dynamic selection.

In this work, we train our model offline on a collection of benchmark functions, employing a mixture of gradient-based, population-based, and probabilistic model-based optimizers. We demonstrate the effectiveness of our method by comparing its performance to classical optimizers on standard benchmark problems and a neural net classification task. The empirical results indicate that L2CO is a promising tool with improved generalization capabilities.

<div align="center"><img src="img/overview.svg" width="800"/></div>

## Key Features

- **30+ optimizers** -- gradient-based (Optax: Adam, SGD, L-BFGS, ...) and evolutionary (EvoSax: CMA-ES, PSO, DE, ...)
- **Two-stage selection** -- Stage 0 (cold-start from task features) + Stage 1 (trajectory-aware reassessment via LSTM)
- **Built on JAX/Equinox** -- JIT-compiled, vmappable, functional
- **Multiple benchmarks** -- BBOB, CEC 2005, PDE, neural network training tasks
- **Hydra + f3dasm pipeline** -- reproducible experiment orchestration with HPC/SLURM support

## Dependencies

L2CO builds on the following packages:

| Package | Description |
| --- | --- |
| [f3dasm](https://github.com/bessagroup/f3dasm) | Framework for data-driven design and analysis of structures and materials |
| [bbob-jax](https://github.com/bessagroup/bbob-jax) | JAX implementation of the BBOB benchmark functions |
| [l2co-tasks](https://github.com/bessagroup/l2co-tasks) | Benchmark functions and tasks used by L2CO (BBOB, CEC 2005, PDE, neural network training) |
| [bessaplots](https://github.com/bessagroup/bessaplots) | Plotting utilities |
| [JAX](https://github.com/jax-ml/jax) | Composable transformations of NumPy programs |
| [Equinox](https://github.com/patrick-kidger/equinox) | Neural networks and more for JAX |
| [Optax](https://github.com/google-deepmind/optax) | Gradient-based optimization in JAX |
| [EvoSax](https://github.com/RobertTLange/evosax) | Evolutionary strategies in JAX |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Machine learning in Python |

## Authorship

* Current creator and developer: [M.P. van der Schelling](https://github.com/mpvanderschelling/) (M.P.vanderSchelling@tudelft.nl)

## Getting started

The package is currently not available on PyPI. To install the package, follow the steps below:

### Installation instructions for users


1. Create a new virtual environment (optional) with Python 3.12+


2. Clone the repository:

```bash
git clone https://github.com/bessagroup/l2co.git
```

3. Navigate to the repository and install the package using `pip`:

```bash
pip install -e .
```

### Installation instructions for developers

To install the package for development (or for building the mkdocs documentation), install the development dependencies after normal installation:

```bash
pip install -e '.[dev,docs,tests]'
```

> For detailed instructions on how to develop on this package, please refer to the [Contributing Guide](CONTRIBUTING.md).

After installation, start by looking at the [Getting Started Guide](./docs/usage.ipynb). To plug your own optimizer into the `RolloutWrapper`, see [Register your own optimizer](./docs/register_optimizer.ipynb).

## Experiments

The experimental pipeline consists of three stages, available in the `/experiments/` folder:

| Experiment | Description |
| --- | --- |
| [`create_optimization_dataset`](https://github.com/bessagroup/l2co/tree/main/experiments/create_optimization_dataset) | Run algorithm portfolio on benchmark tasks, collect RTD trajectories |
| [`l2co_train_stage0`](https://github.com/bessagroup/l2co/tree/main/experiments/l2co_train_stage0) | Train cold-start algorithm selection model (sklearn regressors) |
| [`l2co_train_stage1`](https://github.com/bessagroup/l2co/tree/main/experiments/l2co_train_stage1) | Train trajectory-aware reassessment model (LSTM + MLP) |

The `experiments/` folder also includes `generate_figures.py` for ERTD plot generation and `submit_job.sh` for SLURM cluster submission. See [`experiments/README.md`](experiments/README.md) for full documentation on configuration, parameters, and Hydra overrides.

## Community Support

* If you find any **issues, bugs or problems** within this repository, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co/issues) to report them.
* If you have **questions, feature requests or ideas** for this project, please use the [GitHub Discussions](https://github.com/bessagroup/L2CO/discussions)

Please refer to L2CO's [Code of Conduct](CODE_OF_CONDUCT.md)

## License

Copyright 2026, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full license text.

## Related repositories

This package is part of the L2CO ecosystem developed in the [Bessa Research Group](https://github.com/bessagroup). The repositories below work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
