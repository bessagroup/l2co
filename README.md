L2CO
------
*Learning to Choose Optimizers*

***

## Summary

Welcome to `l2co`, a Python package for learning to choose optimizers.
In this study, we investigate the use of machine learning to select the best optimizer for a given optimization problem problem.

## Statement of need

When solving a new optimization problem, we do not know what is the best optimization algorithm and its parameters that would quickly get us to the optimum. Unfortunately, the "one optimizer to rule them all" does not exist. According to the famous paper of Wolpert et al.[^1], algorithms tend to exploit certain problem-specific characteristics, and if this particular feature is not present in the optimization problem then the performance of the optimizer will be worse than random search, i.e. random sampling (random trial-and-error!).

However, what if we could train a machine learning model on many different optimization problems considering many different optimizers? The machine learning algorithm would start to learn when an optimizer should be used or not. If it is trained properly, then the next time we consider a new problem this algorithm could make an informed decision on which optimizer to use at a given iteration! This is the field of research called "learning to optimize".

In this study, we will consider a discrete number of diverse and well-known optimization algorithms, and we will train a machine learning model to select the best optimizer for a given optimization problem. We will consider a wide range of optimization problems, including convex and non-convex problems, and we will consider a wide range of optimizers, including gradient-based, gradient-free, and meta-heuristic algorithms. Before and during the optimization, the model will choose an optimizer before the optimization starts and is able to switch during optimization.

[^1]: Wolpert, D. H., & Macready, W. G. (1997). *No free lunch theorems for optimization*. IEEE transactions on evolutionary computation, **1**(1), 67-82.

## Authorship

* Current created and developer: [M.P. van der Schelling](https://github.com/mpvanderschelling/) (M.P.vanderSchelling@tudelft.nl)

## Getting started


### Installation instructions for users

The package is currently not available on PyPI. To install the package, follow the steps below:

1. Create a new virtual environment (optional) with Python 3.8+


2. Clone the repository:

```bash
git clone https://github.com/bessagroup/L2CO.git
```

3. Navigate to the repository and install the package using `pip`:

```bash
pip install -e .
```

### Installation instructions for developers

To install the package for development, install the development dependencies after normal installation:

```bash
pip install -r requirements_dev.txt
```

## Experiments

The following experiments are currently implemented:

| Experiment | Description | Status |
| ---------- | ----------- | ------ |
| [`create_training_data_two_optimizers`](https://github.com/bessagroup/L2CO/tree/main/experiments/create_trainingdata_two_optimizers) | Creates the training data for the two-optimizer experiments. | Stable |
| [`recommend_optimizer`](https://github.com/bessagroup/L2CO/tree/main/experiments/recommend_optimizer) | Train and test the recommender system model to recommend optimizers. | Under development |
| [`investigate_stage0`](https://github.com/bessagroup/L2CO/tree/main/experiments/investigate_stage0) | Hyperparameter optimization of classification models for 'stage 0' in the recommendation process | Under development |

## Community Support

* If you find any **issues, bugs or problems** within this repository, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co/issues) to report them.
* If you have **questions, feature requests or ideas** for this project, please use the [GitHub Discussions](https://github.com/bessagroup/L2CO/discussions)

Please refer to L2CO's [Code of Conduct](https://github.com/bessagroup/L2CO/blob/main/CODE_OF_CONDUCT.md)

## License

Copyright 2023, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/l2co/blob/main/LICENSE) for the full license text.