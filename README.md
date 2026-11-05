# Learning to Choose Optimizers (l2co)

<div align="center">
  <img src="https://raw.githubusercontent.com/bessagroup/l2co/main/img/logo.png" alt="BBOB functions 3D overview" width="50%">
</div>

[![Python](https://img.shields.io/pypi/pyversions/l2co)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/l2co.svg)](https://pypi.org/project/l2co/)
[![GitHub license](https://img.shields.io/badge/license-BSD-blue)](https://github.com/bessagroup/l2co)
[![Documentation Status](https://readthedocs.org/projects/l2co/badge/?version=latest)](https://l2co.readthedocs.io/en/latest/?badge=latest)

[**Docs**](https://l2co.readthedocs.io/)
| [**GitHub**](https://github.com/bessagroup/l2co)
| [**PyPI**](https://pypi.org/project/l2co/)
| [**Paper**](https://doi.org/10.1109/CEC69861.2026.11727310)

***

## Summary

**L2CO** is a Python project accompanying the paper [*Learning to Choose Optimizers*](https://doi.org/10.1109/CEC69861.2026.11727310) (IEEE CEC 2026).

## Statement of need

Optimization plays a central role in data-driven design. However, choosing an appropriate optimization algorithm for a specific task is often challenging, as it depends on the structure of the optimization problem and practical limitations such as the computational budget. To tackle this challenge, the field has seen a shift from manually designing optimizers toward using machine learning methods that attempt to learn the optimization process itself. Despite significant progress, existing approaches frequently exhibit limited generalization to new distributions and typically require retraining to transfer their knowledge.

We introduce a new approach called ''Learning to Choose Optimizers'' (L2CO). In our approach, a meta-learner first selects an optimizer before any function evaluations---based solely on available problem characteristics---and then reassesses that choice using a unified model that jointly encodes problem context, optimizer identity, and the observed optimization trajectory. By restricting selection to well-established algorithms, L2CO avoids the convergence issues of learned update rules while retaining the adaptability of dynamic selection.

In this work, we train our model offline on a collection of benchmark functions, employing a mixture of gradient-based, population-based, and probabilistic model-based optimizers. We demonstrate the effectiveness of our method by comparing its performance to classical optimizers on standard benchmark problems and a neural net classification task. The empirical results indicate that L2CO is a promising tool with improved generalization capabilities.

<div align="center"><img src="https://raw.githubusercontent.com/bessagroup/l2co/main/img/overview.svg" width="800"/></div>

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
| [l2co-optimizers](https://github.com/bessagroup/l2co-optimizers) | Bare optimizers compatible with the L2CO library |
| [bessaplots](https://github.com/bessagroup/bessaplots) | Plotting utilities |
| [JAX](https://github.com/jax-ml/jax) | Composable transformations of NumPy programs |
| [Equinox](https://github.com/patrick-kidger/equinox) | Neural networks and more for JAX |
| [Optax](https://github.com/google-deepmind/optax) | Gradient-based optimization in JAX |
| [EvoSax](https://github.com/RobertTLange/evosax) | Evolutionary strategies in JAX |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Machine learning in Python |

## Getting started

### Installation instructions for users

Install `l2co` from PyPI into an environment with Python 3.12 or newer:

```bash
pip install l2co
```

### Installation instructions for developers

Clone the repository and install it in editable mode, with the development dependencies (also needed for building the mkdocs documentation):

```bash
git clone https://github.com/bessagroup/l2co.git
cd l2co
pip install -e '.[dev,docs,tests]'
```

With `uv`, a plain `uv sync` expects checkouts of `f3dasm`, `l2co-tasks`, `l2co-optimizers` and `bessaplots` next to this repository (see `[tool.uv.sources]` in `pyproject.toml`). Run `uv sync --no-sources` to take them from PyPI instead.

> For detailed instructions on how to develop on this package, please refer to the [Contributing Guide](https://github.com/bessagroup/l2co/blob/main/CONTRIBUTING.md).

After installation, start by looking at the [Getting Started Guide](https://l2co.readthedocs.io/en/latest/usage/). To plug your own optimizer into the `RolloutWrapper`, see [Register your own optimizer](https://github.com/bessagroup/l2co-optimizers/blob/main/docs/register_optimizer.ipynb) in `l2co-optimizers`.

## Experiments

The experimental pipeline consists of three stages, available in the `/experiments/` folder:

| Experiment | Description |
| --- | --- |
| [`create_optimization_dataset`](https://github.com/bessagroup/l2co/tree/main/experiments/create_optimization_dataset) | Run algorithm portfolio on benchmark tasks, collect RTD trajectories |
| [`l2co_train_stage0`](https://github.com/bessagroup/l2co/tree/main/experiments/l2co_train_stage0) | Train cold-start algorithm selection model (sklearn regressors) |
| [`l2co_train_stage1`](https://github.com/bessagroup/l2co/tree/main/experiments/l2co_train_stage1) | Train trajectory-aware reassessment model (LSTM + MLP) |

The `experiments/` folder also includes `generate_figures.py` for ERTD plot generation and `submit_job.sh` for SLURM cluster submission. See [`experiments/README.md`](https://github.com/bessagroup/l2co/blob/main/experiments/README.md) for full documentation on configuration, parameters, and Hydra overrides.

The results in the paper were produced with l2co [v1.3.1](https://github.com/bessagroup/l2co/tree/v1.3.1). The code has evolved since.

## Authorship & Citation

Current creator and developer: [M.P. van der Schelling](https://github.com/mpvanderschelling/) (M.P.vanderSchelling@tudelft.nl, [ORCID](https://orcid.org/0000-0003-3602-0452))

If you use `l2co` in your research or in a scientific publication, it is appreciated that you cite the paper below:

**IEEE Congress on Evolutionary Computation (CEC) 2026** ([paper](https://doi.org/10.1109/CEC69861.2026.11727310)):
```bibtex
@inproceedings{vanderSchelling2026,
  title     = {Learning to Choose Optimizers},
  author    = {M. P. van der Schelling and D. Toshniwal and M. A. Bessa},
  booktitle = {2026 IEEE Congress on Evolutionary Computation (CEC)},
  year      = {2026},
  pages     = {1--8},
  publisher = {IEEE},
  doi       = {10.1109/CEC69861.2026.11727310},
  url       = {https://doi.org/10.1109/CEC69861.2026.11727310}
}
```

## Community Support

* If you find any **issues, bugs or problems** within this repository, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co/issues) to report them.
* If you have **questions, feature requests or ideas** for this project, please use the [GitHub Discussions](https://github.com/bessagroup/l2co/discussions)

Please refer to L2CO's [Code of Conduct](https://github.com/bessagroup/l2co/blob/main/CODE_OF_CONDUCT.md)

## License

Copyright 2026, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/l2co/blob/main/LICENSE) for the full license text.
