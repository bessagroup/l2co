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
    OptStates,
    StepFunction,
)
from .history_state import HistoryState
from .sampler import random_sampling
from .task import Task
from .tree_utils import combined_loss
from .update_state import SAMPLING_STEP, UpdateState, best, best_one

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def optax_fn(static: PyTree,
             optimizer: optax.GradientTransformation,
             loss_fn: LossFunction,
             bounded: tuple[float, float],
             pass_rng: bool,
             opt_hash: int,
             opt_index: int,
             n_optimizers: int,
             **kwargs
             ) -> tuple[InitFunction, StepFunction]:

    def init_fn(update_state: UpdateState, key: PRNGKeyArray = None,
                **sample) -> OptaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        opt_state = optimizer.init(params)
        return opt_state

    @eqx.filter_jit
    def step_fn(carry: tuple[InputParameters, OptStates, PRNGKeyArray],
                sample: dict, *args, **kwargs
                ) -> tuple[tuple[InputParameters, OptStates, PRNGKeyArray],
                           dict]:
        input_params, opt_states, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params)

        model = eqx.combine(input_params, static)
        new_key, _ = jr.split(key)

        if pass_rng:
            rng_sample = {'key': new_key}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **sample, **rng_sample)
        loss, grads = eqx.filter_value_and_grad(fn)(model)

        # Extract and update the active optimizer state
        opt_state: OptaxState = opt_states[opt_index]
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(input_params, updates)

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, new_opt_state)

        history = {
            'loss': loss,  # shape (1,)
            'params': input_params,  # shape (dim,)
            'update_step': opt_hash,  # shape (1,)
        }

        # Expand dimensions
        history = jax.tree.map(
            lambda x: jnp.expand_dims(x, 0), history)

        # Apply box constraints
        new_params = jax.tree.map(
            lambda p: jnp.clip(p, *bounded), new_params
        )

        new_params = jax.tree.map(
            lambda x: jnp.expand_dims(x, 0), new_params
        )

        return (new_params, new_opt_states, new_key), history

    return init_fn, step_fn


def optax_extra_args_fn(static: PyTree,
                        optimizer: optax.GradientTransformationExtraArgs,
                        loss_fn: LossFunction,
                        bounded: tuple[float, float],
                        pass_rng: bool,
                        opt_hash: int,
                        opt_index: int,
                        n_optimizers: int,
                        **kwargs
                        ) -> tuple[InitFunction, StepFunction]:
    def init_fn(update_state: UpdateState,
                key: PRNGKeyArray = None, **sample) -> OptaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        opt_state = optimizer.init(params)
        return opt_state

    @eqx.filter_jit
    def step_fn(carry: tuple[InputParameters, OptStates, PRNGKeyArray],
                sample: dict, *args, **kwargs
                ) -> tuple[tuple[InputParameters, OptStates, PRNGKeyArray],
                           dict]:
        input_params, opt_states, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params)

        model = eqx.combine(input_params, static)
        new_key, _ = jr.split(key)

        if pass_rng:
            rng_sample = {'key': new_key}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **sample, **rng_sample)

        opt_state: OptaxState = opt_states[opt_index]
        loss, grads = optax.value_and_grad_from_state(
            fn)(model, state=opt_state)

        updates, new_opt_state = optimizer.update(
            grads, opt_state, input_params,
            value=loss, grad=grads, value_fn=fn,
        )

        new_params = eqx.apply_updates(input_params, updates)

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, new_opt_state)

        history = {
            'loss': loss,  # shape (1, 1)
            'params': input_params,  # shape (1, dim)
            'update_step': opt_hash,  # shape (1, 1)
        }

        # Expand dimensions
        history = jax.tree.map(
            lambda x: jnp.expand_dims(x, 0), history)

        new_params = jax.tree.map(
            lambda p: jnp.clip(p, *bounded), new_params)

        new_params = jax.tree.map(
            lambda x: jnp.expand_dims(x, 0), new_params
        )

        return (new_params, new_opt_states, new_key), history

    return init_fn, step_fn


def evosax_distribution_based_fn(
    static: PyTree,
        optimizer: DistributionBasedAlgorithm,
        loss_fn: LossFunction,
        bounded: tuple[float, float],
        pass_rng: bool,
        popsize: int,
        es_params: Params,
        opt_hash: int,
        opt_index: int,
        n_optimizers: int,
        **kwargs
) -> tuple[InitFunction, StepFunction]:

    def init_fn(update_state: UpdateState,
                key: PRNGKeyArray = None, **sample) -> EvoSaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        return optimizer.init(key, params, es_params)

    @eqx.filter_jit
    def step_fn(carry: tuple[InputParameters, OptStates, PRNGKeyArray],
                sample: dict, *args, **kwargs) -> tuple[
                    tuple[InputParameters, OptStates, PRNGKeyArray], dict]:
        input_params, opt_states, key = carry

        new_key, rng_key = jr.split(key)

        if pass_rng:
            rng_sample = {'key': jr.split(new_key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(partial(combined_loss, static=static,
                                loss_fn=loss_fn, **sample)
                        )(input_params, **rng_sample)

        opt_state: EvoSaxState = opt_states[opt_index]

        opt_state, _ = optimizer.tell(
            key=rng_key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params
        )

        new_params, opt_state = optimizer.ask(
            key=rng_key,
            state=opt_state,
            params=es_params)

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, opt_state)

        update_step = jnp.broadcast_to(opt_hash, loss.shape)

        history = {
            'loss': loss,
            'params': input_params,
            'update_step': update_step,
        }

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        return (new_params, new_opt_states, new_key), history

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
    opt_index: int,
    n_optimizers: int,
    **kwargs
) -> tuple[InitFunction, StepFunction]:
    def init_fn(update_state: UpdateState, key: PRNGKeyArray,
                **sample) -> EvoSaxState:
        population = jax.tree.map(
            lambda x: x[-popsize:], update_state.last_params)

        if pass_rng:
            rng_sample = {'key': jr.split(key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(partial(combined_loss, static=static,
                                loss_fn=loss_fn, **sample)
                        )(population, **rng_sample)

        return optimizer.init(key=key,
                              population=population,
                              fitness=loss,
                              params=es_params
                              )

    @eqx.filter_jit
    def step_fn(carry: tuple[InputParameters, OptStates, PRNGKeyArray],
                sample: dict, *args, **kwargs) -> tuple[
                    tuple[InputParameters, OptStates, PRNGKeyArray], dict]:
        input_params, opt_states, key = carry

        new_key, rng_key = jr.split(key)

        if pass_rng:
            rng_sample = {'key': jr.split(new_key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(partial(combined_loss, static=static,
                                loss_fn=loss_fn, **sample)
                        )(input_params, **rng_sample)

        opt_state: EvoSaxState = opt_states[opt_index]

        opt_state, _ = optimizer.tell(
            key=rng_key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params
        )

        new_params, opt_state = optimizer.ask(
            key=rng_key,
            state=opt_state,
            params=es_params)

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, opt_state)

        update_step = jnp.broadcast_to(opt_hash, loss.shape)

        history = {
            'loss': loss,
            'params': input_params,
            'update_step': update_step,
        }

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        return (new_params, new_opt_states, new_key), history

    return init_fn, step_fn

# =============================================================================


class UpdateClass(eqx.Module):
    init_fn: InitFunction = eqx.field(static=True)
    handshake_fn: Callable = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    index: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)

    @jax.jit
    def init(self, update_state: UpdateState, key: PRNGKeyArray) -> PyTree:
        """Initialize the optimizer state."""
        return self.init_fn(update_state, key)

    @jax.jit
    def handshake(self, update_state: UpdateState,
                  key: PRNGKeyArray) -> UpdateState:
        case_id = jnp.where(
            (update_state.last_step == SAMPLING_STEP) | (
                update_state.last_step == self.hash),
            0,
            1,
        )

        return jax.lax.switch(
            case_id,
            (
                lambda s: s,
                lambda s: self.handshake_fn(update_state=s, key=key),
            ),
            update_state,
        )

    def step(
        self,
        opt_states: tuple[PyTree],
        update_state: UpdateState,
        history_state: HistoryState,
        n_iterations: int,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, UpdateState]:

        update_state = self.handshake(
            update_state=update_state, key=key)

        params = jax.tree.map(
            lambda x: x[-self.popsize:], update_state.last_params)

        (_, opt_states, _), history = jax.lax.scan(
            f=lambda carry, _: self.step_fn(
                carry, sample={}),
            init=(params, opt_states, key),
            xs=None,
            length=n_iterations+1
        )

        # Remove first element for dim=0 history
        history = jax.tree.map(
            lambda x: x[1:, :], history)

        # flatten the first and second dimension of each key in history
        history_flattened = jax.tree.map(
            lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
            history
        )

        final_update_state = UpdateState.from_history(
            history=history_flattened,
            memory_size=update_state.memory_size,
            last_step=self.hash,)

        update_state = update_state.merge(final_update_state)

        new_history_state = history_state.add(history)

        return update_state, opt_states, new_history_state

# =============================================================================


def optax_update(task: Task,
                 optimizer: optax.GradientTransformation,
                 opt_hash: int,
                 popsize: int,
                 bounded: tuple[float, float] | None = None,
                 opt_index: int = 0,
                 n_optimizers: int = 1,
                 **hyperparameters
                 ) -> UpdateClass:

    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        opt_hash=opt_hash,
        opt_index=opt_index,
        n_optimizers=n_optimizers,)

    handshake_fn = best

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        index=opt_index,
        hash=opt_hash,
    )


def optax_update_extra_args(task: Task,
                            optimizer: optax.GradientTransformationExtraArgs,
                            opt_hash: int,
                            popsize: int,
                            bounded: tuple[float, float] | None = None,
                            opt_index: int = 0,
                            n_optimizers: int = 1,
                            **hyperparameters) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_extra_args_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        opt_hash=opt_hash,
        opt_index=opt_index,
        n_optimizers=n_optimizers,)

    handshake_fn = best

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        index=opt_index,
        hash=opt_hash,
    )


def evosax_distribution_update(task: Task,
                               optimizer: type[DistributionBasedAlgorithm],
                               opt_hash: int,
                               popsize: int,
                               bounded: tuple[float, float] | None = (
                                   None, None),
                               opt_index: int = 0,
                               n_optimizers: int = 1,
                               sampler: Callable = random_sampling,
                               **hyperparameters) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize,
                          solution=params)
    es_params = optimizer.default_params
    es_params = es_params.replace(**hyperparameters)

    init_fn, step_fn = evosax_distribution_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        popsize=popsize,
        es_params=es_params,
        opt_hash=opt_hash,
        opt_index=opt_index,
        n_optimizers=n_optimizers,
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
        index=opt_index,
        hash=opt_hash,

    )


def evosax_population_update(task: Task,
                             optimizer: type[PopulationBasedAlgorithm],
                             opt_hash: int,
                             popsize: int,
                             bounded: tuple[float, float] | None = (
                                 None, None),
                             opt_index: int = 0,
                             n_optimizers: int = 1,
                             sampler: Callable = random_sampling,
                             **hyperparameters) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize,
                          solution=params)
    es_params = optimizer.default_params
    es_params = es_params.replace(**hyperparameters)

    init_fn, step_fn = evosax_population_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=task.loss_fn,
        bounded=bounded,
        pass_rng=task.pass_rng,
        popsize=popsize,
        es_params=es_params,
        opt_hash=opt_hash,
        opt_index=opt_index,
        n_optimizers=n_optimizers,
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
        index=opt_index,
        hash=opt_hash,

    )

# =============================================================================


def cmaes_variable_popsize(
    task: Task, popsize: int | None = None,
        **kwargs) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_distribution_update,
        optimizer=CMA_ES,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def pso_variable_popsize(
    task: Task, popsize: int | None = None,
        **kwargs) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_population_update,
        optimizer=PSO,
    )

    if popsize is None:
        popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def sepcmaes_variable_popsize(
    task: Task, popsize: int | None = None,
        **kwargs) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_distribution_update,
        optimizer=Sep_CMA_ES,
    )

    popsize = int(4 + 3 * jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


OPTIMIZER_MAPPING: dict[str, UpdateClass] = {
    'adam': partial(optax_update, optimizer=optax.adam,
                    popsize=1),
    'sgd': partial(optax_update, optimizer=optax.sgd,
                   popsize=1),
    'sepcmaes': sepcmaes_variable_popsize,
    'cmaes': cmaes_variable_popsize,
    'pso': pso_variable_popsize,
    'lbfgsb': partial(optax_update_extra_args, optimizer=optax.lbfgs,
                      popsize=1)
}
