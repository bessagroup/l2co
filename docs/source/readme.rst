Welcome to the documentation page of the 'Learning to Choose Optimizers' (:mod:`l2co`).
Here you will find all information on installing, using and contributing to the Python package.


-----------------


Summary
-------

When solving a new optimization problem, we do not know what is the best optimization algorithm and its parameters that would quickly get us to the optimum. Unfortunately, the "one optimizer to rule them all" does not exist. According to the famous paper of Wolpert et al, algorithms tend to exploit certain problem-specific characteristics, and if this particular feature is not present in the optimization problem then the performance of the optimizer will be worse than random search, i.e. random sampling (random trial-and-error!).

However, what if we could train a machine learning model on many different optimization problems considering many different optimizers? The machine learning algorithm would start to learn when an optimizer should be used or not. If it is trained properly, then the next time we consider a new problem this algorithm could make an informed decision on which optimizer to use at a given iteration! This is the field of research called "learning to optimize".

In this study, we will consider a discrete number of diverse and well-known optimization algorithms, and we will train a machine learning model to select the best optimizer for a given optimization problem. We will consider a wide range of optimization problems, including convex and non-convex problems, and we will consider a wide range of optimizers, including gradient-based, gradient-free, and meta-heuristic algorithms. Before and during the optimization, the model will choose an optimizer before the optimization starts and is able to switch during optimization.

-----------------

Authorship & Citation
---------------------

:mod:`l2co` is created and maintained by Martin van der Schelling [1]_.

.. [1] PhD Candiate, Delft University of Technology, `Website <https://mpvanderschelling.github.io/>`_ , `GitHub <https://github.com/mpvanderschelling/>`_

If you use :mod:`l2co` in your research or in a scientific publication, it is appreciated that you reference the GitHub package
`f3dasm <https://github.com/bessagroup/f3dasm>`_.

-----------------

Getting started
---------------


The best way to get started is to:

* Read the :ref:`overview` section, containting a brief introduction to :mod:`l2co` and a statement of need.
* Follow the :ref:`installation-instructions` section, containing a step-by-step guide on how to install the package.

-----------------

Contribute
----------

:mod:`l2co` is an open-source project, and contributions of any kind are welcome and appreciated. If you want to contribute, please send me a message or open an issue on GitHub.


-----------------

License
-------
Copyright 2023, Martin van der Schelling

All rights reserved.

:mod:`l2co` is a free and open-source software published under a `BSD 3-Clause License <https://github.com/bessagroup/l2co/blob/main/LICENSE>`_.
