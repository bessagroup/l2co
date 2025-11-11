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
import optax
from evosax.algorithms import CMA_ES, PSO, Sep_CMA_ES
from evosax.algorithms.base import Params
from evosax.algorithms.base import State as EvoSaxState
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState as OptaxState

# Local
from ._typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
)
from .handshake import best, best_one, reset_opt_state
from .opt_history import OptHistory
from .sampler import random_sampling
from .task import Task
from .tree_utils import combined_loss
from .update_class import UpdateClass
from .update_state import UpdateState

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
    pass_rng: bool,
    opt_hash: int,
    n_updates_per_iteration: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    def init_fn(
        update_state: UpdateState, key: PRNGKeyArray = None, **sample
    ) -> OptaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        opt_state = optimizer.init(params)
        return opt_state

    def _single_update(carry, _):
        input_params, opt_state, sample, key = carry

        model = eqx.combine(input_params, static)
        new_key, _ = jr.split(key)

        if pass_rng:
            rng_sample = {"key": new_key}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **sample, **rng_sample)
        loss, grads = eqx.filter_value_and_grad(fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=opt_hash,
            fevals=n_updates_per_iteration,
            iterations=1,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        return (new_params, new_opt_state, sample, new_key), history

    @eqx.filter_jit
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

        (new_params, new_opt_state, _, new_key), history = jax.lax.scan(
            f=_single_update,
            init=(input_params, opt_state, sample, key),
            xs=None,
            length=n_updates_per_iteration,
        )

        history = jax.tree.map(lambda x: jnp.expand_dims(x[-1], 0), history)

        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def optax_extra_args_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    pass_rng: bool,
    opt_hash: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    def init_fn(
        update_state: UpdateState, key: PRNGKeyArray = None, **sample
    ) -> OptaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        opt_state = optimizer.init(params)
        return opt_state

    @eqx.filter_jit
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
        new_key, _ = jr.split(key)

        if pass_rng:
            rng_sample = {"key": new_key}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **sample, **rng_sample)

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
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=opt_hash,
            fevals=1,
            iterations=1,
        )

        # Expand dimensions
        history = jax.tree.map(lambda x: jnp.expand_dims(x, 0), history)

        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


#                                                                        EvoSax
# =============================================================================


def evosax_distribution_based_fn(
    static: PyTree,
    optimizer: DistributionBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    pass_rng: bool,
    popsize: int,
    es_params: Params,
    opt_hash: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    def init_fn(
        update_state: UpdateState, key: PRNGKeyArray = None, **sample
    ) -> EvoSaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        return optimizer.init(key, params, es_params)

    @eqx.filter_jit
    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        input_params, opt_state, key = carry

        new_key, rng_key = jr.split(key)

        if pass_rng:
            rng_sample = {"key": jr.split(new_key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(
            partial(combined_loss, static=static, loss_fn=loss_fn, **sample)
        )(input_params, **rng_sample)

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
    def init_fn(
        update_state: UpdateState, key: PRNGKeyArray, **sample
    ) -> EvoSaxState:
        population = jax.tree.map(
            lambda x: x[-popsize:], update_state.last_params
        )

        if pass_rng:
            rng_sample = {"key": jr.split(key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(
            partial(combined_loss, static=static, loss_fn=loss_fn, **sample)
        )(population, **rng_sample)

        return optimizer.init(
            key=key, population=population, fitness=loss, params=es_params
        )

    @eqx.filter_jit
    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        input_params, opt_state, key = carry

        new_key, rng_key = jr.split(key)

        if pass_rng:
            rng_sample = {"key": jr.split(new_key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(
            partial(combined_loss, static=static, loss_fn=loss_fn, **sample)
        )(input_params, **rng_sample)

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
    n_updates_per_iteration: int = 1,
    stop_fn: StopFunction | None = None,
    handshake_fn_opt_state: Callable | None = None,
    **hyperparameters,
) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        opt_hash=opt_hash,
        n_updates_per_iteration=n_updates_per_iteration,
    )

    handshake_fn = best

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        handshake_fn_opt_state=handshake_fn_opt_state,
    )


def optax_update_extra_args(
    task: Task,
    optimizer: optax.GradientTransformationExtraArgs,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    stop_fn: StopFunction | None = None,
    handshake_fn_opt_state: Callable | None = None,
    **hyperparameters,
) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_extra_args_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        opt_hash=opt_hash,
    )

    handshake_fn = best

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        handshake_fn_opt_state=handshake_fn_opt_state,
    )


def evosax_distribution_update(
    task: Task,
    optimizer: type[DistributionBasedAlgorithm],
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = (None, None),
    sampler: Callable = random_sampling,
    stop_fn: StopFunction | None = None,
    handshake_fn_opt_state: Callable | None = None,
    **hyperparameters,
) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)

    init_fn, step_fn = evosax_distribution_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        popsize=popsize,
        es_params=optimizer.default_params.replace(**hyperparameters),
        opt_hash=opt_hash,
        handshake_fn_opt_state=handshake_fn_opt_state,
    )

    handshake_fn = partial(
        best_one,
        task=task,
        sampler=sampler,
    )

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
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
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)

    init_fn, step_fn = evosax_population_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        popsize=popsize,
        es_params=optimizer.default_params.replace(**hyperparameters),
        opt_hash=opt_hash,
    )

    handshake_fn = partial(
        best_one,
        task=task,
        sampler=sampler,
    )

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


# =============================================================================


def cmaes_variable_popsize(
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_distribution_update,
        optimizer=CMA_ES,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def pso_variable_popsize(
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_population_update,
        optimizer=PSO,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def sepcmaes_variable_popsize(
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_distribution_update,
        optimizer=Sep_CMA_ES,
    )

    popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def adam_variable_updates(
    task: Task, n_updates_per_iteration: int | None = None, **kwargs
):
    d = task.dimensionality

    optimizer = partial(
        optax_update,
        optimizer=optax.adam,
        popsize=1,
    )

    n_updates_per_iteration = int(4 + 3 * jnp.log(d))

    return optimizer(
        **kwargs, task=task, n_updates_per_iteration=n_updates_per_iteration
    )


OPTIMIZER_MAPPING: dict[str, UpdateClass] = {
    "adam": partial(optax_update, optimizer=optax.adam, popsize=1),
    "adam_variable_updates": adam_variable_updates,
    "sgd": partial(optax_update, optimizer=optax.sgd, popsize=1),
    "sepcmaes": sepcmaes_variable_popsize,
    "cmaes": cmaes_variable_popsize,
    "pso": pso_variable_popsize,
    "lbfgsb": partial(
        optax_update_extra_args,
        optimizer=optax.lbfgs,
        popsize=1,
        handshake_fn_opt_state=reset_opt_state,
    ),
}
