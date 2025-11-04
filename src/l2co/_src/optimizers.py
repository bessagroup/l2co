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
    OptHistoryType,
    OptStates,
    StepFunction,
    StopFunction,
)
from .history_state import HistoryState
from .sampler import random_sampling
from .stopping_criteria import (
    combined_stopping_criteria,
    covariance_matrix_stopping_criteria,
    gradient_based_stopping_criteria,
)
from .task import Task
from .tree_utils import combined_loss
from .update_state import SAMPLING_STEP, UpdateState, best, best_one

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#                                                                    OptHistory
# =============================================================================


class OptHistory(OptHistoryType):
    loss: jax.Array
    params: PyTree
    grads: PyTree
    update_step: jax.Array
    fevals: jax.Array

    @classmethod
    def init(cls, params: PyTree, popsize: int, last_step: int) -> OptHistory:
        loss = jnp.full((popsize,), jnp.nan)
        params = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), params)
        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), params)
        update_step = jnp.full((popsize,), last_step)
        fevals = jnp.zeros((popsize,), dtype=jnp.int32)

        return cls(
            loss=loss,
            params=params,
            grads=grads,
            update_step=update_step,
            fevals=fevals,
        )

    @classmethod
    def from_init_sample(
        cls, update_state: UpdateState, task: Task, key: PRNGKeyArray
    ):
        # Evaluate the initial parameters
        init_params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        init_loss, init_grads = jax.tree.map(
            lambda x: x[-1], update_state.evaluate_with_grads(task, key)
        )

        loss = jax.tree.map(lambda x: jnp.expand_dims(x, (0, 1)), init_loss)
        params = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), init_params
        )
        grads = jax.tree.map(lambda x: jnp.expand_dims(x, (0, 1)), init_grads)
        update_step = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), update_state.last_step
        )
        fevals = jax.tree.map(
            lambda x: jnp.expand_dims(x, (0, 1)), jnp.array(1)
        )

        return cls(
            loss=loss,
            params=params,
            grads=grads,
            update_step=update_step,
            fevals=fevals,
        )

    def update(self, history: OptHistory) -> OptHistory:
        """Update the recent history with the current history."""

        loss = jnp.concatenate([self.loss[1:], history.loss[None]], axis=0)
        params = jax.tree.map(
            lambda p, new: jnp.concatenate([p[1:], new[None]], axis=0),
            self.params,
            history.params,
        )
        grads = jax.tree.map(
            lambda g, new: jnp.concatenate([g[1:], new[None]], axis=0),
            self.grads,
            history.grads,
        )
        update_step = jnp.concatenate(
            [self.update_step[1:], history.update_step[None]], axis=0
        )
        fevals = jnp.concatenate(
            [self.fevals[1:], history.fevals[None]], axis=0
        )
        return OptHistory(
            loss=loss,
            params=params,
            grads=grads,
            update_step=update_step,
            fevals=fevals,
        )


#                                                                         Optax
# =============================================================================


def optax_fn(
    static: PyTree,
    optimizer: optax.GradientTransformation,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    pass_rng: bool,
    opt_hash: int,
    opt_index: int,
    n_optimizers: int,
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
        input_params, opt_states, sample, key = carry

        model = eqx.combine(input_params, static)
        new_key, _ = jr.split(key)

        if pass_rng:
            rng_sample = {"key": new_key}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **sample, **rng_sample)
        loss, grads = eqx.filter_value_and_grad(fn)(model)
        opt_state: OptaxState = opt_states[opt_index]
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(input_params, updates)
        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, new_opt_state
        )
        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=opt_hash,
            fevals=n_updates_per_iteration,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        return (new_params, new_opt_states, sample, new_key), history

    @eqx.filter_jit
    def step_fn(
        carry: tuple[InputParameters, OptStates, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptStates, PRNGKeyArray], OptHistory]:
        input_params, opt_states, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params
        )

        (new_params, new_opt_states, _, new_key), history = jax.lax.scan(
            f=_single_update,
            init=(input_params, opt_states, sample, key),
            xs=None,
            length=n_updates_per_iteration,
        )

        history = jax.tree.map(lambda x: jnp.expand_dims(x[-1], 0), history)

        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        return (new_params, new_opt_states, new_key), history

    return init_fn, step_fn


def optax_extra_args_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    pass_rng: bool,
    opt_hash: int,
    opt_index: int,
    n_optimizers: int,
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
        carry: tuple[InputParameters, OptStates, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptStates, PRNGKeyArray], OptHistory]:
        input_params, opt_states, key = carry

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

        opt_state: OptaxState = opt_states[opt_index]
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

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, new_opt_state
        )

        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=opt_hash,
            fevals=1,
        )

        # Expand dimensions
        history = jax.tree.map(lambda x: jnp.expand_dims(x, 0), history)

        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        return (new_params, new_opt_states, new_key), history

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
    opt_index: int,
    n_optimizers: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    def init_fn(
        update_state: UpdateState, key: PRNGKeyArray = None, **sample
    ) -> EvoSaxState:
        params = jax.tree.map(lambda x: x[-1], update_state.last_params)
        return optimizer.init(key, params, es_params)

    @eqx.filter_jit
    def step_fn(
        carry: tuple[InputParameters, OptStates, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptStates, PRNGKeyArray], OptHistory]:
        input_params, opt_states, key = carry

        new_key, rng_key = jr.split(key)

        if pass_rng:
            rng_sample = {"key": jr.split(new_key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(
            partial(combined_loss, static=static, loss_fn=loss_fn, **sample)
        )(input_params, **rng_sample)

        opt_state: EvoSaxState = opt_states[opt_index]

        opt_state, _ = optimizer.tell(
            key=rng_key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, opt_state = optimizer.ask(
            key=rng_key, state=opt_state, params=es_params
        )

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, opt_state
        )

        update_step = jnp.broadcast_to(opt_hash, loss.shape)
        fevals = jnp.broadcast_to(popsize, loss.shape)
        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=update_step,
            fevals=fevals,
            grads=grads,
        )

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
        carry: tuple[InputParameters, OptStates, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptStates, PRNGKeyArray], OptHistory]:
        input_params, opt_states, key = carry

        new_key, rng_key = jr.split(key)

        if pass_rng:
            rng_sample = {"key": jr.split(new_key, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(
            partial(combined_loss, static=static, loss_fn=loss_fn, **sample)
        )(input_params, **rng_sample)

        opt_state: EvoSaxState = opt_states[opt_index]

        opt_state, _ = optimizer.tell(
            key=rng_key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, opt_state = optimizer.ask(
            key=rng_key, state=opt_state, params=es_params
        )

        new_opt_states = eqx.tree_at(
            lambda t: t[opt_index], opt_states, opt_state
        )

        update_step = jnp.broadcast_to(opt_hash, loss.shape)
        fevals = jnp.broadcast_to(popsize, loss.shape)
        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=update_step,
            fevals=fevals,
            grads=grads,
        )

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
    hash: int  # = eqx.field(static=True)
    stop_fn: StopFunction = eqx.field(
        static=True, default=lambda _, __, ___: False
    )

    @jax.jit
    def init(self, update_state: UpdateState, key: PRNGKeyArray) -> PyTree:
        """Initialize the optimizer state."""
        return self.init_fn(update_state, key)

    @jax.jit
    def handshake(
        self, update_state: UpdateState, key: PRNGKeyArray
    ) -> UpdateState:
        case_id = jnp.where(
            (update_state.last_step == SAMPLING_STEP)
            | (update_state.last_step == self.hash),
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

    def scan_step(self, carry, *args, **kwargs):
        return self.step_fn(carry, sample={})

    def step(
        self,
        opt_states: tuple[PyTree],
        update_state: UpdateState,
        history_state: HistoryState,
        n_iterations: int,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, UpdateState]:
        update_state = self.handshake(update_state=update_state, key=key)

        params = jax.tree.map(
            lambda x: x[-self.popsize :], update_state.last_params
        )

        window_size = 20

        def scan_step(
            carry: tuple[
                PyTree, tuple[PyTree], PRNGKeyArray, jax.Array, OptHistory
            ],
            _,
        ):
            params, opt_states, key, done, recent_history = carry
            stop_condition = self.stop_fn(
                recent_history, opt_states, self.index
            )

            done = done | stop_condition

            def do_step(
                carry: tuple[
                    PyTree,
                    tuple[PyTree],
                    PRNGKeyArray,
                    jax.Array,
                    OptHistory,
                ],
            ):
                (params, opt_states, key, done, recent_history) = carry
                new_carry, history = self.step_fn(
                    (params, opt_states, key), sample={}
                )

                new_recent_history = recent_history.update(history)

                return (*new_carry, done, new_recent_history), history

            def skip_step(
                carry: tuple[
                    PyTree,
                    tuple[PyTree],
                    PRNGKeyArray,
                    jax.Array,
                    OptHistory,
                ],
            ):
                (params, opt_states, key, done, recent_history) = carry
                dummy_history = recent_history.init(
                    params, self.popsize, self.hash
                )

                return (
                    params,
                    opt_states,
                    key,
                    done,
                    recent_history,
                ), dummy_history

            (new_carry, history_entry) = jax.lax.cond(
                done, skip_step, do_step, carry
            )

            return new_carry, history_entry

        # Initialize moving history buffer with NaNs
        recent_history_init = OptHistory.init(
            params=params,
            popsize=self.popsize,
            last_step=update_state.last_step,
        )

        recent_history_init = jax.tree.map(
            lambda x: jnp.tile(x[None], (window_size, *([1] * x.ndim))),
            recent_history_init,
        )

        init_done = jnp.array(False)
        (_, opt_states, _, _, _), history = jax.lax.scan(
            f=scan_step,
            init=(params, opt_states, key, init_done, recent_history_init),
            xs=None,
            length=n_iterations + 1,
        )

        # Remove first element for dim=0 history
        history = jax.tree.map(lambda x: x[1:, :], history)

        # flatten the first and second dimension of each key in history
        history_flattened = jax.tree.map(
            lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
            history,
        )

        final_update_state = UpdateState.from_history(
            history=history_flattened,
            memory_size=update_state.memory_size,
            last_step=self.hash,
        )

        update_state = update_state.merge(final_update_state)

        new_history_state = history_state.add(history)

        return update_state, opt_states, new_history_state


# =============================================================================


def optax_update(
    task: Task,
    optimizer: optax.GradientTransformation,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    opt_index: int = 0,
    n_optimizers: int = 1,
    n_updates_per_iteration: int = 1,
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
        opt_index=opt_index,
        n_optimizers=n_optimizers,
        n_updates_per_iteration=n_updates_per_iteration,
    )

    handshake_fn = best

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        index=opt_index,
        hash=opt_hash,
        stop_fn=combined_stopping_criteria,
    )


def optax_update_extra_args(
    task: Task,
    optimizer: optax.GradientTransformationExtraArgs,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    opt_index: int = 0,
    n_optimizers: int = 1,
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
        opt_index=opt_index,
        n_optimizers=n_optimizers,
    )

    handshake_fn = best

    return UpdateClass(
        init_fn=init_fn,
        handshake_fn=handshake_fn,
        step_fn=step_fn,
        popsize=popsize,
        index=opt_index,
        hash=opt_hash,
        stop_fn=gradient_based_stopping_criteria,
    )


def evosax_distribution_update(
    task: Task,
    optimizer: type[DistributionBasedAlgorithm],
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = (None, None),
    opt_index: int = 0,
    n_optimizers: int = 1,
    sampler: Callable = random_sampling,
    stop_fn: StopFunction = lambda _, __, ___: False,
    **hyperparameters,
) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)
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
        stop_fn=stop_fn,
    )


def evosax_population_update(
    task: Task,
    optimizer: type[PopulationBasedAlgorithm],
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = (None, None),
    opt_index: int = 0,
    n_optimizers: int = 1,
    sampler: Callable = random_sampling,
    **hyperparameters,
) -> UpdateClass:
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)
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
    task: Task, popsize: int | None = None, **kwargs
) -> UpdateClass:
    d = task.dimensionality

    optimizer = partial(
        evosax_distribution_update,
        optimizer=CMA_ES,
        stop_fn=covariance_matrix_stopping_criteria,
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
        stop_fn=covariance_matrix_stopping_criteria,
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
        optax_update_extra_args, optimizer=optax.lbfgs, popsize=1
    ),
}
