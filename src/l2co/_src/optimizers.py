from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.tree as jt
import optax
from evosax import CMA_ES, PSO, EvoState, Strategy
from f3dasm import Block, ExperimentData
from jaxtyping import PyTree

from .tree_utils import (dict_list_to_tree, tree_expand_dims,
                         tree_flatten_population_dim, tree_remove_first,
                         tree_to_dict_list, update_pytree_at_index)


class UpdateStep(Block):
    update_fn: Callable

    def __init__(self, optimizer: optax.GradientTransformation,
                 seed: int,
                 popsize: int,
                 update_fn: Callable,
                 include_input: bool,
                 pass_rng: bool,
                 bounded: Optional[Tuple[float, float]] = None,
                 **hyperparameters):

        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = jrd.key(seed)
        self.opt_state = None
        self.pass_rng = pass_rng

        self.optimizer = optimizer(**hyperparameters)

    def arm(self, data: ExperimentData) -> None:
        for k, v in data[-self.popsize:]:
            x = v.input_data['x']
        self.params, self.static = eqx.partition(x, eqx.is_inexact_array)

    def call(self, data: ExperimentData, fn: Callable, n_iterations: int,
             data_iter: Iterator, has_aux: bool = False,
             **kwargs
             ) -> ExperimentData:
        params_history, loss_history, opt_state, aux_history = self.update_fn(
            params=self.params,
            static=self.static,
            optimizer=self.optimizer,
            loss_fn=fn,
            inner_steps=n_iterations,
            data_iter=data_iter,
            seed=self.seed,
            bounded=self.bounded,
            opt_state=self.opt_state,
            has_aux=has_aux,
            include_input=self.include_input,
            pass_rng=self.pass_rng,
            popsize=self.popsize,
        )

        self.opt_state = opt_state

        if self.include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, self.static), params_history)
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(loss_history, name='y')

        return ExperimentData(
            domain=data.domain,
            input_data=input_data,
            output_data=output_data,
            project_dir=data.project_dir)


def lbfgs_adapt(params: PyTree, static: PyTree,
                optimizer: optax.GradientTransformationExtraArgs,
                loss_fn: Callable,
                inner_steps: int,
                seed: jax.Array,
                bounded: Optional[Tuple[float, float]],
                data_iter: Iterator,
                opt_state: Optional[optax.OptState],
                has_aux: bool,
                include_input: bool,
                pass_rng: bool,
                **kwargs,
                ) -> Tuple[PyTree, PyTree, optax.OptState]:

    def scan_fn(carry, sample):
        inner_params, opt_state, rng = carry

        if pass_rng:
            rng, _ = jrd.split(rng)
            rng_sample = {'key': rng}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **sample, **rng_sample)

        loss, grads = optax.value_and_grad_from_state(fn)(
            eqx.combine(inner_params, static), state=opt_state)

        updates, opt_state = optimizer.update(
            grads, opt_state, inner_params,
            value=loss, grad=grads,
            value_fn=fn,
        )
        new_params = eqx.apply_updates(inner_params, updates)

        # Apply box constraints if `bounded` is True
        if bounded is not None:
            new_params = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': inner_params})

        return (new_params, opt_state, rng), history

    if opt_state is None:
        opt_state = optimizer.init(params)

    samples = unroll_batches(data_iter, inner_steps+1)

    (_, opt_state, _), history = jax.lax.scan(
        f=scan_fn,
        init=(params, opt_state, seed),
        xs=samples,
        length=inner_steps+1
    )

    history_params = jax.tree_map(
        lambda x: x, history['params']) if include_input else None
    history_losses = jax.tree_map(lambda x: x, history['loss'])
    history_aux = jax.tree_map(
        lambda x: x, history['aux']) if has_aux else None

    return history_params, history_losses, opt_state, history_aux

# =============================================================================


def unroll_batches(data_iter: Iterator, inner_steps: int):
    # Collect samples while preserving dictionary structure
    data = [next(data_iter) for _ in range(inner_steps)]

    if not data:
        return {}, {}

    # Stack the batch data directly
    samples = jax.tree_util.tree_map(
        lambda *vals: jnp.stack(vals, axis=0), *data)

    return samples


def optax_adapt(params: PyTree, static: PyTree,
                optimizer: optax.GradientTransformation,
                loss_fn: Callable,
                inner_steps: int,
                seed: jax.Array,
                bounded: Optional[Tuple[float, float]],
                data_iter: Iterator,
                opt_state: Optional[optax.OptState],
                has_aux: bool,
                include_input: bool,
                pass_rng: bool,
                **kwargs) -> Tuple[PyTree, PyTree, optax.OptState]:

    def scan_fn(carry, sample):
        inner_params, opt_state, rng = carry

        model = eqx.combine(inner_params, static)

        if pass_rng:
            rng, _ = jrd.split(rng)
            rng_sample = {'key': rng}
        else:
            rng_sample = {}

        if has_aux:
            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=has_aux)(
                    model, **sample, **rng_sample)
        else:
            loss, grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=has_aux)(
                    model, **sample, **rng_sample)
            aux = None

        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(inner_params, updates)

        # Apply box constraints if `bounded` is True
        if bounded is not None:
            new_params = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': inner_params})

        if has_aux:
            history.update({'aux': aux})

        return (new_params, opt_state, rng), history

    if opt_state is None:
        opt_state = optimizer.init(params)

    samples = unroll_batches(data_iter, inner_steps+1)

    (_, opt_state, _), history = jax.lax.scan(
        scan_fn,
        (params, opt_state, seed),
        xs=samples,
        length=inner_steps+1,
    )

    history_params = jax.tree_map(
        lambda x: x, history['params']) if include_input else None
    history_losses = jax.tree_map(lambda x: x, history['loss'])
    history_aux = jax.tree_map(
        lambda x: x, history['aux']) if has_aux else None

    return history_params, history_losses, opt_state, history_aux


class OptHistory(NamedTuple):
    losses: PyTree
    params: PyTree
    aux: PyTree
    index: int

    @staticmethod
    def append(history: OptHistory, loss: PyTree,
               params: PyTree, aux: PyTree) -> OptHistory:
        new_losses = update_pytree_at_index(
            history.losses, loss, history.index)
        new_params = update_pytree_at_index(
            history.params, params, history.index)
        new_aux = update_pytree_at_index(
            history.aux, aux, history.index) if aux is not None else None

        return OptHistory(new_losses, new_params, new_aux, history.index + 1)


def optax_adapt_for_loop(params: PyTree, static: PyTree,
                         optimizer: optax.GradientTransformation,
                         loss_fn: Callable,
                         inner_steps: int,
                         bounded: Optional[Tuple[float, float]],
                         data_iter: Iterator,
                         loss_iter: Iterator,
                         opt_state: Optional[optax.OptState],
                         has_aux: bool,
                         **kwargs) -> Tuple[PyTree, PyTree, optax.OptState]:

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def make_step(model: eqx.Module, opt_state: optax.OptState,
                  history_state: OptHistory, **batch
                  ) -> Tuple[PyTree, optax.OptState, OptHistory]:
        model = jax.tree_util.tree_unflatten(treedef_model, model)
        opt_state = jax.tree_util.tree_unflatten(
            treedef_opt_state, opt_state)

        aux = None

        if has_aux:
            (loss, aux), grads = filtered_loss_fn(model,
                                                  **batch)
        else:
            loss, grads = filtered_loss_fn(model, **batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)

        if bounded is not None:
            new_model = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_model)

        new_model = jax.tree_util.tree_leaves(new_model)
        opt_state = jax.tree_util.tree_leaves(opt_state)

        new_history_state = OptHistory.append(history=history_state,
                                              loss=loss, params=model,
                                              aux=aux)

        # new_history_state = jax.tree.map(
        #     lambda x, y: tree_append(
        #         x, y), history_state, (loss, new_model, aux)
        # )

        return new_model, opt_state, new_history_state

    if opt_state is None:
        opt_state = optimizer.init(params)

    model = eqx.combine(params, static)

    filtered_loss_fn = eqx.filter_value_and_grad(
        loss_fn, has_aux=has_aux)

    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    history_state = OptHistory(
        losses=jnp.zeros((inner_steps+1,)),  # Fixed shape
        params=tree_expand_dims(model),
        aux={} if has_aux else None,
        index=0
    )

    for _, sample, loss_sample in zip(range(inner_steps+1),
                                      data_iter, loss_iter):
        flat_model, flat_opt_state, history_state = make_step(
            flat_model, flat_opt_state, history_state, **sample, **loss_sample)

    # Remove the first element from history_state.params
    history_state_params = jax.tree.map(
        lambda x: tree_remove_first(x), history_state.params)

    return (history_state_params, history_state.losses,
            opt_state, history_state.aux)

# =============================================================================


def evosax_adapt(params: PyTree, static: PyTree,
                 optimizer: Strategy,
                 loss_fn: Callable,
                 inner_steps: int,
                 seed: jax.Array,
                 bounded: Optional[Tuple[float, float]],
                 data_iter: Iterator,
                 opt_state: Optional[EvoState],
                 has_aux: bool,
                 include_input: bool,
                 pass_rng: bool,
                 popsize: int,
                 **kwargs
                 ) -> Tuple[PyTree, PyTree, EvoState]:

    def scan_fn(carry, sample):
        inner_params, opt_state, rng = carry

        rng, rng_gen = jrd.split(rng)

        new_params, opt_state = optimizer.ask(
            rng=rng_gen,
            state=opt_state,
            params=es_params)

        if pass_rng:
            rng_sample = {'key': jrd.split(rng, num=popsize)}
        else:
            rng_sample = {}

        inner_models = jax.vmap(eqx.combine, in_axes=(
            0, None))(new_params, static)

        loss = jax.vmap(partial(loss_fn, **sample)
                        )(inner_models, **rng_sample)

        opt_state = optimizer.tell(
            x=new_params,
            fitness=loss,
            state=opt_state,
            params=es_params
        )

        history = {'loss': loss}

        if include_input:
            history.update({'params': inner_params})

        return (new_params, opt_state, rng), history

    es_params = optimizer.default_params

    if bounded is not None:
        es_params = es_params.replace(clip_min=bounded[0], clip_max=bounded[1])

    if opt_state is None:
        opt_state = optimizer.initialize(rng=seed, params=es_params)

    samples = unroll_batches(data_iter, inner_steps+1)

    (_, opt_state, _), history = jax.lax.scan(
        f=scan_fn,
        init=(params, opt_state, seed),
        xs=samples,
        length=int(inner_steps+1),
    )

    history_params = jax.tree_map(
        lambda x: x, history['params']) if include_input else None
    history_losses = jax.tree_map(lambda x: x, history['loss'])
    history_aux = jax.tree_map(
        lambda x: x, history['aux']) if has_aux else None

    return (tree_flatten_population_dim(history_params),
            history_losses.flatten(), opt_state, history_aux)


class EvoSaxUpdate(UpdateStep):
    def __init__(self, optimizer: Strategy,
                 seed: int,
                 popsize: int,
                 update_fn: Callable,
                 include_input: bool,
                 bounded: Optional[Tuple[float, float]] = None,
                 pass_rng: bool = False,
                 **hyperparameters):
        self.optimizer_class = optimizer
        self.update_fn = update_fn
        self.include_input = include_input
        self.bounded = bounded
        self.popsize = popsize
        self.hyperparameters = hyperparameters
        self.pass_rng = pass_rng
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
    'adam_forloop': partial(UpdateStep, optimizer=optax.adam,
                            update_fn=optax_adapt_for_loop,
                            popsize=1),
    'cmaes': partial(EvoSaxUpdate, optimizer=CMA_ES, update_fn=evosax_adapt),
    'pso': partial(EvoSaxUpdate, optimizer=PSO, update_fn=evosax_adapt),
    'lbfgsb': partial(UpdateStep, optimizer=optax.lbfgs, update_fn=lbfgs_adapt,
                      popsize=1),
}
