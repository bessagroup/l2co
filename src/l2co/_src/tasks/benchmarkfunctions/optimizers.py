from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.tree as jt
import optax
from evosax import CMA_ES, PSO, EvoState, Strategy
from f3dasm import Block, ExperimentData
from jaxtyping import PyTree

from .tree_utils import (dict_list_to_tree, tree_extend,
                         tree_flatten_population_dim, tree_to_dict_list,
                         tree_trim)


class UpdateStep(Block):
    update_fn: Callable

    def __init__(self, optimizer: optax.GradientTransformation,
                 seed: int,
                 popsize: int,
                 update_fn: Callable,
                 bounded: Optional[Tuple[float, float]] = None,
                 **hyperparameters):

        self.popsize = popsize
        self.bounded = bounded
        self.update_fn = update_fn
        self.seed = jrd.key(seed)
        self.opt_state = None

        self.optimizer = optimizer(**hyperparameters)

    def arm(self, data: ExperimentData) -> None:
        for k, v in data[-self.popsize:]:
            x = v.input_data['x']
        self.params, self.static = eqx.partition(x, eqx.is_inexact_array)

    def call(self, data: ExperimentData, fn: Callable, n_evals: int,
             data_iter: Iterator, has_aux: bool = False,
             **kwargs
             ) -> ExperimentData:
        params_history, loss_history, opt_state, aux_history = self.update_fn(
            params=self.params, static=self.static,
            optimizer=self.optimizer, loss_fn=fn,
            inner_steps=n_evals, data_iter=data_iter,
            seed=self.seed,
            bounded=self.bounded,
            opt_state=self.opt_state,
            has_aux=has_aux,
        )

        self.opt_state = opt_state

        # Trim the first population from the histories
        params_history = tree_trim(tree=params_history, n=self.popsize)
        loss_history = tree_trim(tree=loss_history, n=self.popsize)

        if has_aux:
            aux_history = tree_trim(tree=aux_history, n=self.popsize)

        models = jax.vmap(eqx.combine, in_axes=(0, None))(
            params_history, self.static)

        # self.aux_history = aux_history
        # self.loss_history = loss_history
        # self.params_history = params_history

        return ExperimentData(
            domain=data.domain,
            input_data=tree_to_dict_list(models, name='x'),
            output_data=tree_to_dict_list(loss_history, name='y'),
            project_dir=data.project_dir)


def lbfgs_adapt(params: PyTree, static: PyTree,
                optimizer: optax.GradientTransformationExtraArgs,
                loss_fn: Callable,
                inner_steps: int,
                bounded: Optional[Tuple[float, float]],
                data_iter: Iterator,
                opt_state: Optional[optax.OptState],
                aux: Optional[Dict[str, Any]],
                **kwargs,
                ) -> Tuple[PyTree, PyTree, optax.OptState]:

    if opt_state is None:
        opt_state = optimizer.init(params)

    def scan_fn(carry, _):
        inner_params, opt_state, sample = carry

        loss, grads = optax.value_and_grad_from_state(loss_fn)(
            eqx.combine(inner_params, static), **sample, state=opt_state)

        updates, opt_state = optimizer.update(
            grads, opt_state, inner_params,
            value=loss, grad=grads,
            value_fn=partial(loss_fn, **sample),
        )
        new_params = eqx.apply_updates(inner_params, updates)

        # Apply box constraints if `bounded` is True
        if bounded is not None:
            new_params = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        return (new_params, opt_state, next(data_iter)), (inner_params, loss)

    (final_params, opt_state, final_sample), (
        history_params, history_losses) = jax.lax.scan(
        scan_fn,
        (params, opt_state, next(data_iter)),
        xs=None,
        length=inner_steps
    )

    # Evaluate the final parameters
    final_loss = loss_fn(eqx.combine(final_params, static), **final_sample)

    # Add the final parameters and loss to the histories
    history_params = tree_extend(history_params, final_params)
    history_losses = tree_extend(history_losses, final_loss)

    return history_params, history_losses, opt_state, None

# =============================================================================

# =============================================================================


# def optax_adapt_old(params: PyTree, static: PyTree,
#                 optimizer: optax.GradientTransformation,
#                 loss_fn: Callable,
#                 inner_steps: int,
#                 bounded: Optional[Tuple[float, float]],
#                 data_iter: Iterator,
#                 opt_state: Optional[optax.OptState],
#                 has_aux: bool,
#                 **kwargs) -> Tuple[PyTree, PyTree, optax.OptState]:
#     if opt_state is None:
#         opt_state = optimizer.init(params)

#     def scan_fn(carry, _):
#         inner_params, opt_state, sample = carry

#         if has_aux:
#             (loss, aux), grads = eqx.filter_value_and_grad(
#                 loss_fn, has_aux=has_aux)(eqx.combine(inner_params, static),
#                                           **sample)
#         else:
#             loss, grads = eqx.filter_value_and_grad(
#                 loss_fn, has_aux=has_aux)(eqx.combine(inner_params, static),
#                                           **sample)
#             aux = None

#         updates, opt_state = optimizer.update(grads, opt_state)
#         new_params = eqx.apply_updates(inner_params, updates)

#         # Apply box constraints if `bounded` is True
#         if bounded is not None:
#             new_params = jax.tree_map(
#                 lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

#         return (new_params, opt_state, next(data_iter)), \
#             (inner_params, loss, aux)

#     (final_params, opt_state, final_sample), (
#         history_params, history_losses, history_aux) = jax.lax.scan(
#         scan_fn,
#         (params, opt_state, next(data_iter)),
#         xs=None,
#         length=inner_steps
#     )

#     # Evaluate the final parameters
#     if has_aux:
#         final_loss, final_aux = loss_fn(eqx.combine(final_params, static),
#                                         **final_sample)
#     else:
#         final_loss = loss_fn(eqx.combine(final_params, static), **final_sample)

#     # Add the final parameters and loss to the histories
#     history_params = tree_extend(history_params, final_params)
#     history_losses = tree_extend(history_losses, final_loss)
#     history_aux = tree_extend(history_aux, final_aux) if has_aux else None

#     return history_params, history_losses, opt_state, history_aux

def tree_append(tree: PyTree, new_values: PyTree) -> PyTree:
    return jax.tree_map(lambda x, y: jnp.concatenate([x[None, ...] if x.ndim == y.ndim else x, y[None, ...]]), tree, new_values)


def tree_remove_first(tree: PyTree) -> PyTree:
    return jax.tree_map(lambda x: x[1:], tree)


def optax_adapt(params: PyTree, static: PyTree,
                optimizer: optax.GradientTransformation,
                loss_fn: Callable,
                inner_steps: int,
                bounded: Optional[Tuple[float, float]],
                data_iter: Iterator,
                opt_state: Optional[optax.OptState],
                has_aux: bool,
                **kwargs) -> Tuple[PyTree, PyTree, optax.OptState]:
    if opt_state is None:
        opt_state = optimizer.init(params)

    filtered_loss_fn = eqx.filter_value_and_grad(
        loss_fn, has_aux=has_aux)

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def make_step(params, opt_state, **sample):
        aux = None
        if has_aux:
            (loss, aux), grads = filtered_loss_fn(eqx.combine(params, static),
                                                  **sample)
        else:
            loss, grads = filtered_loss_fn(
                eqx.combine(params, static), **sample)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(params, updates)

        if bounded is not None:
            new_params = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        return new_params, loss, opt_state, aux

    history_params = params
    history_losses = jnp.array([])
    history_aux = {} if has_aux else None

    for step, sample in zip(range(inner_steps), data_iter):
        history_params = tree_append(history_params, params)
        params, loss, opt_state, aux = make_step(params, opt_state, **sample)
        history_losses = tree_append(history_losses, loss)
        if has_aux:
            history_aux = tree_append(history_aux, aux)

    history_params = tree_remove_first(history_params)

    return history_params, history_losses, opt_state, history_aux

# =============================================================================


def evosax_adapt(params: PyTree, static: PyTree,
                 optimizer: Strategy,
                 loss_fn: Callable,
                 inner_steps: int,
                 seed,
                 bounded: Optional[Tuple[float, float]],
                 data_iter: Iterator,
                 opt_state: Optional[EvoState],
                 aux: Optional[Dict[str, Any]],
                 **kwargs
                 ) -> Tuple[PyTree, PyTree, EvoState]:

    es_params = optimizer.default_params
    if bounded is not None:
        es_params = es_params.replace(clip_min=bounded[0], clip_max=bounded[1])

    if opt_state is None:
        opt_state = optimizer.initialize(rng=seed, params=es_params)

    def scan_fn(carry, _):
        inner_params, opt_state, sample, rng = carry

        rng, rng_gen = jrd.split(rng)

        new_params, opt_state = optimizer.ask(
            rng=rng_gen,
            state=opt_state,
            params=es_params)

        inner_models = jax.vmap(eqx.combine, in_axes=(
            0, None))(new_params, static)

        loss = jax.vmap(partial(loss_fn, **sample))(inner_models)

        opt_state = optimizer.tell(
            x=new_params,
            fitness=loss,
            state=opt_state,
            params=es_params
        )

        return (new_params, opt_state, next(data_iter), rng), (
            inner_params, loss)

    (final_params, opt_state, final_sample, _), (
        history_params, history_losses) = jax.lax.scan(
        scan_fn,
        (params, opt_state, next(data_iter), seed),
        xs=None,
        length=inner_steps
    )

    # Evaluate the final parameters
    final_models = jax.vmap(eqx.combine, in_axes=(
        0, None))(final_params, static)

    final_loss = jax.vmap(partial(loss_fn, **final_sample))(final_models)

    # Add the final parameters and loss to the histories
    history_params = tree_extend(
        tree_flatten_population_dim(history_params), final_params,
        expand_axis=False)
    history_losses = tree_extend(
        history_losses.flatten(), final_loss, expand_axis=False)

    return history_params, history_losses, opt_state, None


class EvoSaxUpdate(UpdateStep):
    def __init__(self, optimizer: Strategy,
                 seed: int,
                 popsize: int,
                 update_fn: Callable,
                 bounded: bool = False,
                 **hyperparameters):
        self.optimizer_class = optimizer
        self.update_fn = update_fn
        self.bounded = bounded
        self.popsize = popsize
        self.hyperparameters = hyperparameters
        self.opt_state = None
        self.seed = jrd.key(seed)

    def arm(self, data: ExperimentData) -> None:
        x = [v.input_data for k, v in data[-self.popsize:]]
        pop_x = dict_list_to_tree(x, name='x')

        self.params, self.static = eqx.partition(pop_x, eqx.is_inexact_array)

        self.optimizer = self.optimizer_class(
            popsize=self.popsize,
            pholder_params=jt.map(lambda x: x[0], self.params),
            **self.hyperparameters)

# =============================================================================


OPT_MAPPING: Dict[str, Type[UpdateStep]] = {
    'adam': partial(UpdateStep, optimizer=optax.adam, update_fn=optax_adapt,
                    popsize=1),
    'cmaes': partial(EvoSaxUpdate, optimizer=CMA_ES, update_fn=evosax_adapt),
    'pso': partial(EvoSaxUpdate, optimizer=PSO, update_fn=evosax_adapt),
    'lbfgsb': partial(UpdateStep, optimizer=optax.lbfgs, update_fn=lbfgs_adapt,
                      popsize=1),
}
