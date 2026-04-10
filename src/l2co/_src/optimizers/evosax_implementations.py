"""
Module for optimizers.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from functools import partial

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from evosax.algorithms import RandomSearch, algorithms
from evosax.algorithms.base import Params
from evosax.algorithms.base import State as EvoSaxState
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based import population_based_algorithms
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree

# Local
from l2co._src.opt_history import OptHistory
from l2co._src.optimizers.utils import normalize_key
from l2co._src.sampler import random_sampling
from l2co._src.task import Task
from l2co._src.tree_utils import (
    vmapped_loss,
    vmapped_loss_with_rng,
)
from l2co._src.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
)
from l2co._src.update_class import UpdateClass

#                                                                        EvoSax
# =============================================================================


def evosax_distribution_based_fn(
    static: PyTree,
    optimizer: DistributionBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    popsize: int,
    es_params: Params,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for EvoSax distribution-based
    optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : DistributionBasedAlgorithm
        EvoSax distribution-based optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    popsize : int
        Population size.
    es_params : Params
        Evolution strategy parameters.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """

    def init_fn(
        params: InputParameters, key: PRNGKeyArray = None, **sample
    ) -> EvoSaxState:
        """Initialise the distribution-based ES state."""
        params = jax.tree.map(lambda x: x[0], params)
        return optimizer.init(key, params, es_params)

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        """Perform one distribution-based ES step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )

        else:
            loss = vmapped_loss(
                eqx.combine(input_params, static), loss_fn, sample
            )

        new_opt_state, _ = optimizer.tell(
            key=key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=key, state=new_opt_state, params=es_params
        )

        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=opt_hash,
            fevals=popsize,
            grads=grads,
            iterations=1,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def evosax_population_based_fn(
    static: PyTree,
    optimizer: PopulationBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    popsize: int,
    es_params: Params,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for EvoSax population-based
    optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : PopulationBasedAlgorithm
        EvoSax population-based optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    popsize : int
        Population size.
    es_params : Params
        Evolution strategy parameters.
    opt_hash : int
        Hash for optimizer step tracking.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """

    def init_fn(
        params: InputParameters, key: PRNGKeyArray, **sample
    ) -> EvoSaxState:
        """Initialise the population-based ES state."""
        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )
        else:
            loss = vmapped_loss(eqx.combine(params, static), loss_fn, sample)

        return optimizer.init(
            key=key, population=params, fitness=loss, params=es_params
        )

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        """Perform one population-based ES step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )
        else:
            loss = vmapped_loss(
                eqx.combine(input_params, static), loss_fn, sample
            )

        new_opt_state, _ = optimizer.tell(
            key=key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=key, state=new_opt_state, params=es_params
        )

        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=opt_hash,
            fevals=popsize,
            grads=grads,
            iterations=1,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


# =============================================================================


def evosax_distribution_update(
    task: Task,
    optimizer: type[DistributionBasedAlgorithm],
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = (None, None),
    sampler: Callable = random_sampling,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for EvoSax distribution-based optimizers.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : type[DistributionBasedAlgorithm]
        EvoSax distribution-based optimizer class.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
    sampler : Callable, optional
        Sampling function.
    stop_fn : StopFunction or None, optional
        Stopping function.
    **hyperparameters
        Additional optimizer hyperparameters.

    Returns
    -------
    UpdateClass
        UpdateClass instance for optimization.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)

    init_fn, step_fn = evosax_distribution_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        popsize=popsize,
        es_params=optimizer.default_params.replace(**hyperparameters),
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


def evosax_randomsearch_update(
    task: Task,
    optimizer: type[DistributionBasedAlgorithm],
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = (None, None),
    sampler: Callable = random_sampling,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for EvoSax distribution-based optimizers.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : type[DistributionBasedAlgorithm]
        EvoSax distribution-based optimizer class.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
    sampler : Callable, optional
        Sampling function.
    stop_fn : StopFunction or None, optional
        Stopping function.
    **hyperparameters
        Additional optimizer hyperparameters.

    Returns
    -------
    UpdateClass
        UpdateClass instance for optimization.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    sampling_fn = partial(
        random_sampling,
        params=params,
        n_samples=1,
    )

    optimizer = optimizer(
        population_size=popsize, solution=params, sampling_fn=sampling_fn
    )

    init_fn, step_fn = evosax_distribution_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        popsize=popsize,
        es_params=optimizer.default_params.replace(**hyperparameters),
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


def evosax_population_update(
    task: Task,
    optimizer: type[PopulationBasedAlgorithm],
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = (None, None),
    sampler: Callable = random_sampling,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for EvoSax population-based optimizers.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : type[PopulationBasedAlgorithm]
        EvoSax population-based optimizer class.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
    sampler : Callable, optional
        Sampling function.
    stop_fn : StopFunction or None, optional
        Stopping function.
    **hyperparameters
        Additional optimizer hyperparameters.

    Returns
    -------
    UpdateClass
        UpdateClass instance for optimization.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)

    init_fn, step_fn = evosax_population_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        popsize=popsize,
        es_params=optimizer.default_params.replace(**hyperparameters),
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


# =============================================================================


def variable_popsize_optimizer(
    task: Task,
    evosax_class,
    popsize: int | None = None,
    **kwargs,
) -> UpdateClass:
    """Create an EvoSax UpdateClass with automatic population sizing.

    If ``popsize`` is ``None``, it defaults to
    ``int(4 + 3 * log(d))`` where *d* is the problem
    dimensionality.

    Parameters
    ----------
    task : Task
        Task containing the model and loss function.
    evosax_class : type
        EvoSax algorithm class (distribution- or
        population-based).
    popsize : int or None, optional
        Population size. Computed automatically when ``None``.
    **kwargs
        Forwarded to the underlying update function.

    Returns
    -------
    UpdateClass
        Configured optimizer update class.
    """
    if evosax_class in distribution_based_algorithms.values():
        if evosax_class == RandomSearch:
            update_fn = evosax_randomsearch_update
        else:
            update_fn = evosax_distribution_update
    elif evosax_class in population_based_algorithms.values():
        update_fn = evosax_population_update
    else:
        raise ValueError(
            f"EvoSax class '{evosax_class.__name__}' not recognized."
        )

    d = task.dimensionality

    optimizer = Partial(
        update_fn,
        optimizer=evosax_class,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


# =============================================================================

normalized_evosax = {normalize_key(k): v for k, v in algorithms.items()}

# remove svcmaes and svopenes from the mapping as they require special
# treatment
_ = normalized_evosax.pop("svcmaes", None)
_ = normalized_evosax.pop("svopenes", None)
_ = normalized_evosax.pop("evotfes", None)
_ = normalized_evosax.pop("les", None)
_ = normalized_evosax.pop("lga", None)


evosax_mapping = {
    name: Partial(
        variable_popsize_optimizer,
        evosax_class=cls,
    )
    for name, cls in normalized_evosax.items()
}
