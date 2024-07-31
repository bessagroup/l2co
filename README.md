Learning to Choose Optimizers (L2CO)
------

***
[![GitHub license](https://img.shields.io/badge/license-BSD-blue)](https://github.com/bessagroup/l2co)


**Docs**
| [**GitHub**](https://github.com/bessagroup/l2co)
| **PyPI**

> Some links are not yet available, as the package is still under a private repository.

***

## Summary

Optimization plays a central role in Computational Mechanics. However, selecting the right optimization algorithm for a particular task is usually not trivial, as that results from the nature of the optimization problem to be solved and constraints such as available resource budget and training data. In order to address this, there has been a transition from creating hand-designed optimization algorithms to learning how to optimize (L2O), i.e. an approach attempting to learn the optimization conditions themselves. While many studies in the field of L2O have demonstrated good performance on a specific task distribution, they often suffer from poor generalization to other distributions. In addition, established L2O methods are often limited to a specific class of optimization algorithms.

We propose a new approach called "Learning to Choose Optimizers" (L2CO). Our method allows a meta-learner to select at test time from a range of static and well-established optimizers, also allowing to switch to a different optimizer during the optimization process itself. This enables the system to adapt to different task distributions and enhance generalization performance. The framework is set up to continuously incorporate new optimization algorithms and problems, inviting domain experts to contribute their insights and expertise for ongoing expansion.

In this study, we train our model offline on a diverse set of benchmark loss-functions and apply a range of gradient-based, population-based, and probabilistic model-based optimizers. The potential of this approach is demonstrated by comparing its performance to classical optimizers and established L2O-architectures on benchmark loss-functions and simple material design studies. The results suggest that L2CO can be a useful tool and exhibits better generalization.

<div align="center"><img src="img/overview.png" width="800"/></div>

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

This package includes a collection of illustrative benchmark studies that demonstrate the capabilities of the framework. These studies are available in the `/experiments/` folder.

## Community Support

* If you find any **issues, bugs or problems** within this repository, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co/issues) to report them.
* If you have **questions, feature requests or ideas** for this project, please use the [GitHub Discussions](https://github.com/bessagroup/L2CO/discussions)

Please refer to L2CO's [Code of Conduct](https://github.com/bessagroup/L2CO/blob/main/CODE_OF_CONDUCT.md)

## License

Copyright 2024, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/l2co/blob/main/LICENSE) for the full license text.