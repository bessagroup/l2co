"""
Module for optimizers.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from evosax.algorithms import CMA_ES, PSO, Sep_CMA_ES
from evosax.algorithms.base import Params
from evosax.algorithms.base import State as EvoSaxState
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState as OptaxState

# Local
from l2co._src.opt_history import OptHistory
from l2co._src.sampler import random_sampling
from l2co._src.task import Task
from l2co._src.tree_utils import vmapped_loss
from l2co._src.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
)
from l2co._src.update_class import UpdateClass
from l2co._src.update_state import UpdateState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#                                                                         Optax
# =============================================================================


def optax_fn(
    static: PyTree,
    optimizer: optax.GradientTransformation,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    opt_hash: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for Optax optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformation
        Optax optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
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
        update_state: UpdateState, key: PRNGKeyArray | None = None
    ) -> OptaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        input_params, opt_state, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params
        )

        model = eqx.combine(input_params, static)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, **sample)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=jax.tree.map(lambda x: jnp.expand_dims(x, 0), loss),
            params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), input_params),
            grads=jax.tree.map(lambda x: jnp.expand_dims(x, 0), grads),
            update_step=opt_hash,
            fevals=1,
            iterations=1,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        # history = jax.tree.map(lambda x: jnp.expand_dims(x, 0), history)

        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def optax_extra_args_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    opt_hash: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for Optax optimizers with extra
    arguments.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer instance supporting extra arguments.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
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
        update_state: UpdateState, key: PRNGKeyArray = None, **sample
    ) -> OptaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        input_params, opt_state, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params
        )

        model = eqx.combine(input_params, static)

        fn = Partial(loss_fn, **sample)

        loss, grads = optax.value_and_grad_from_state(fn)(
            model, state=opt_state
        )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            input_params,
            value=loss,
            grad=grads,
            value_fn=fn,
        )

        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=jax.tree.map(lambda x: jnp.expand_dims(x, 0), loss),
            params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), input_params),
            grads=jax.tree.map(lambda x: jnp.expand_dims(x, 0), grads),
            update_step=opt_hash,
            fevals=1,
            iterations=1,
        )

        # Expand dimensions
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        # history = jax.tree.map(lambda x: jnp.expand_dims(x, 0), history)

        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


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
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """

    def init_fn(
        update_state: UpdateState, key: PRNGKeyArray = None, **sample
    ) -> EvoSaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        return optimizer.init(key, params, es_params)

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        input_params, opt_state, key = carry

        loss = vmapped_loss(eqx.combine(input_params, static), loss_fn, sample)

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

        # update_step = jnp.broadcast_to(opt_hash, loss.shape)
        # fevals = jnp.broadcast_to(popsize, loss.shape)
        # iterations = jnp.broadcast_to(1, loss.shape)
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
    pass_rng: bool,
    popsize: int,
    es_params: Params,
    opt_hash: int,
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
    pass_rng : bool
        Whether to pass random key to loss function.
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
        update_state: UpdateState, key: PRNGKeyArray, **sample
    ) -> EvoSaxState:
        population = jax.tree.map(
            lambda x: x[-popsize:], update_state.last_params
        )

        loss = vmapped_loss(eqx.combine(population, static), loss_fn, sample)

        return optimizer.init(
            key=key, population=population, fitness=loss, params=es_params
        )

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        input_params, opt_state, key = carry

        new_key, rng_key = jr.split(key)

        loss = vmapped_loss(eqx.combine(input_params, static), loss_fn, sample)

        new_opt_state, _ = optimizer.tell(
            key=rng_key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=rng_key, state=new_opt_state, params=es_params
        )

        update_step = jnp.broadcast_to(opt_hash, loss.shape)
        fevals = jnp.broadcast_to(popsize, loss.shape)
        iterations = jnp.broadcast_to(1, loss.shape)
        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=update_step,
            fevals=fevals,
            grads=grads,
            iterations=iterations,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


# =============================================================================


def optax_update(
    task: Task,
    optimizer: optax.GradientTransformation,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for Optax optimizers.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : optax.GradientTransformation
        Optax optimizer constructor.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
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

    init_fn, step_fn = optax_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


def optax_update_extra_args(
    task: Task,
    optimizer: optax.GradientTransformationExtraArgs,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for Optax optimizers with extra arguments.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer constructor supporting extra arguments.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
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

    init_fn, step_fn = optax_extra_args_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


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
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


# =============================================================================


def cmaes_variable_popsize(
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    """
    Creates a CMA-ES optimizer with variable population size based
    on dimensionality.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    popsize : int or None, optional
        Population size. If None, computed from dimensionality.
    **kwargs
        Additional arguments.

    Returns
    -------
    UpdateClass
        UpdateClass instance for CMA-ES optimization.
    """
    d = task.dimensionality

    optimizer = Partial(
        evosax_distribution_update,
        optimizer=CMA_ES,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def pso_variable_popsize(
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    """
    Creates a PSO optimizer with variable population size based
    on dimensionality.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    popsize : int or None, optional
        Population size. If None, computed from dimensionality.
    **kwargs
        Additional arguments.

    Returns
    -------
    UpdateClass
        UpdateClass instance for PSO optimization.
    """
    d = task.dimensionality

    optimizer = Partial(
        evosax_population_update,
        optimizer=PSO,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def sepcmaes_variable_popsize(
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    """
    Creates a Sep-CMA-ES optimizer with variable population size based
    on dimensionality.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    popsize : int or None, optional
        Population size. If None, computed from dimensionality.
    **kwargs
        Additional arguments.

    Returns
    -------
    UpdateClass
        UpdateClass instance for Sep-CMA-ES optimization.
    """
    d = task.dimensionality

    optimizer = Partial(
        evosax_distribution_update,
        optimizer=Sep_CMA_ES,
    )

    popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


# =============================================================================


def optimizer_mapping(name: str) -> UpdateClass:
    """
    Maps optimizer name to corresponding UpdateClass or function.

    Parameters
    ----------
    name : str
        Name of the optimizer.

    Returns
    -------
    UpdateClass
        Corresponding UpdateClass or function for the optimizer.

    Raises
    ------
    ValueError
        If optimizer name is not recognized.
    """
    match name:
        case "adam":
            return Partial(optax_update, optimizer=optax.adam, popsize=1)
        case "sgd":
            return Partial(optax_update, optimizer=optax.sgd, popsize=1)
        case "sepcmaes":
            return sepcmaes_variable_popsize
        case "cmaes":
            return cmaes_variable_popsize
        case "pso":
            return pso_variable_popsize
        case "lbfgsb":
            return Partial(
                optax_update_extra_args,
                optimizer=optax.lbfgs,
                popsize=1,
            )
        case _:
            raise ValueError(f"Optimizer '{name}' not recognized.")
