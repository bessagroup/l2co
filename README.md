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

Optimization plays a central role in Computational Mechanics. However, selecting the right optimization algorithm for a particular task is usually not trivial, as that results from the nature of the optimization problem to be solved and constraints such as available resource budget and training data. In order to address this, there has been a transition from creating hand-designed optimization algorithms to learning how to optimize (L2O), i.e. an approach attempting to learn the optimization conditions themselves. While many studies in the field of L2O have demonstrated good performance on a specific task distribution, they often suffer from poor generalization to other distributions. In addition, established L2O methods are often limited to a specific class of optimization algorithms.

We propose a new approach called "Learning to Choose Optimizers" (L2CO). Our method allows a meta-learner to select at test time from a range of static and well-established optimizers. This enables the system to adapt to different task distributions and enhance generalization performance. The framework is set up to continuously incorporate new optimization algorithms and problems, inviting domain experts to contribute their insights and expertise for ongoing expansion.

In this study, we train our model offline on a diverse set of benchmark loss-functions and apply a range of gradient-based, population-based, and probabilistic model-based optimizers. The potential of this approach is demonstrated by comparing its performance to classical optimizers and established L2O-architectures on benchmark loss-functions and simple material design studies. The results suggest that L2CO can be a useful tool and exhibits better generalization.

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

After installation, start by looking at the [Getting Started Guide](./docs/usage.ipynb).

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

Copyright 2025, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full license text.