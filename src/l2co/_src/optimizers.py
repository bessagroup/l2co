from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.tree as jt
import optax
from evosax import CMA_ES, PSO, EvoParams, EvoState, Strategy
from f3dasm import Block, ExperimentData
from jaxtyping import PyTree
from tqdm import tqdm

from .batching import unroll_batches
from .tree_utils import (dict_list_to_tree, tree_flatten_population_dim,
                         tree_to_dict_list, update_pytree_at_index)


# =============================================================================
class UpdateStep(Block):
    update_fn: Callable

    def __init__(self, optimizer: optax.GradientTransformation,
                 seed: int,
                 popsize: int,
                 update_fn: Callable,
                 include_input: bool,
                 pass_rng: bool,
                 use_scan: bool,
                 bounded: Optional[Tuple[float, float]] = None,
                 **hyperparameters):

        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = jrd.key(seed)
        self.opt_state = None
        self.pass_rng = pass_rng
        self.use_scan = use_scan

        self.optimizer = optimizer(**hyperparameters)

    def arm(self, data: ExperimentData) -> None:
        for k, v in data[-self.popsize:]:
            x = v.input_data['x']
        self.params, self.static = eqx.partition(x, eqx.is_inexact_array)

    def call(self, data: ExperimentData, fn: Callable, n_iterations: int,
             data_iter: Iterator, has_aux: bool = False,
             **kwargs
             ) -> ExperimentData:
        history, opt_state = self.update_fn(
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
            use_scan=self.use_scan,
        )

        self.opt_state = opt_state

        if self.include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, self.static), history['params'])
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(history['loss'], name='y')

        return ExperimentData(
            domain=data.domain,
            input_data=input_data,
            output_data=output_data,
            project_dir=data.project_dir)


class EvoSaxUpdate(UpdateStep):
    def __init__(self, optimizer: Strategy,
                 seed: int,
                 popsize: int,
                 update_fn: Callable,
                 include_input: bool,
                 pass_rng: bool,
                 use_scan: bool,
                 bounded: Optional[Tuple[float, float]] = None,
                 **hyperparameters):
        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = jrd.key(seed)
        self.opt_state = None
        self.pass_rng = pass_rng
        self.use_scan = use_scan

        self.optimizer_class = optimizer
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData) -> None:
        x = [v.input_data for k, v in data[-self.popsize:]]
        pop_x = dict_list_to_tree(x, name='x')

        self.params, self.static = eqx.partition(pop_x, eqx.is_inexact_array)

        self.optimizer = self.optimizer_class(
            popsize=self.popsize,
            pholder_params=jt.map(lambda x: x[0], self.params),
            **self.hyperparameters)

#                                                                Scan functions
# =============================================================================


ScanFunction = Callable[
    [Tuple[PyTree, optax.OptState, jax.Array, Dict[str, jax.Array]]],
    Tuple[Tuple[PyTree, optax.OptState, jax.Array],
          Dict[str, jax.Array]]]


def lbfgs_scan(static: PyTree,
               optimizer: optax.GradientTransformationExtraArgs,
               loss_fn: Callable,
               bounded: Optional[Tuple[float, float]],
               has_aux: bool,
               include_input: bool,
               pass_rng: bool,
               **kwargs
               ) -> ScanFunction:

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

    return scan_fn


def optax_scan(static: PyTree,
               optimizer: optax.GradientTransformation,
               loss_fn: Callable,
               bounded: Optional[Tuple[float, float]],
               has_aux: bool,
               include_input: bool,
               pass_rng: bool,
               **kwargs
               ) -> ScanFunction:

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

    return scan_fn


def evosax_scan(static: PyTree,
                optimizer: Strategy,
                loss_fn: Callable,
                bounded: Optional[Tuple[float, float]],
                has_aux: bool,
                include_input: bool,
                pass_rng: bool,
                popsize: int,
                es_params: EvoParams,
                **kwargs
                ) -> ScanFunction:

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

    return scan_fn

#                                                           Make step functions
# =============================================================================


MakeStepFunction = Callable[
    [eqx.Module, optax.OptState, jax.Array, Dict[str, jax.Array]],
    Tuple[PyTree, optax.OptState, jax.Array, Dict[str, jax.Array]]]


def lbfgs_make_step(static: PyTree,
                    optimizer: optax.GradientTransformationExtraArgs,
                    loss_fn: Callable,
                    bounded: Optional[Tuple[float, float]],
                    has_aux: bool,
                    include_input: bool,
                    pass_rng: bool,
                    **kwargs) -> MakeStepFunction:

    @eqx.filter_jit
    def make_step(params: eqx.Module, opt_state: optax.OptState,
                  rng: jax.Array, **batch
                  ) -> Tuple[PyTree, optax.OptState, jax.Array,
                             Dict[str, jax.Array]]:

        if pass_rng:
            rng, _ = jrd.split(rng)
            rng_sample = {'key': rng}
        else:
            rng_sample = {}

        fn = partial(loss_fn, **batch, **rng_sample)

        if has_aux:
            raise ValueError('Auxiliary data is not supported for L-BFGS')

        else:
            loss, grads = optax.value_and_grad_from_state(fn)(
                eqx.combine(params, static), state=opt_state)

        updates, opt_state = optimizer.update(
            grads, opt_state, params,
            value=loss, grad=grads,
            value_fn=fn,
        )
        new_params = eqx.apply_updates(params, updates)

        if bounded is not None:
            new_params = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': params})

        return new_params, opt_state, rng, history

    return make_step


def optax_make_step(static: PyTree,
                    optimizer: optax.GradientTransformationExtraArgs,
                    loss_fn: Callable,
                    bounded: Optional[Tuple[float, float]],
                    has_aux: bool,
                    include_input: bool,
                    pass_rng: bool,
                    **kwargs) -> MakeStepFunction:
    @eqx.filter_jit
    def make_step(params: eqx.Module, opt_state: optax.OptState,
                  rng: jax.Array, **batch
                  ) -> Tuple[PyTree, optax.OptState, jax.Array,
                             Dict[str, jax.Array]]:

        if pass_rng:
            rng, _ = jrd.split(rng)
            rng_sample = {'key': rng}
        else:
            rng_sample = {}

        if has_aux:
            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=has_aux)(eqx.combine(params, static),
                                          **batch, **rng_sample)
        else:
            loss, grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=has_aux)(eqx.combine(
                    params, static), **batch, **rng_sample)

        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = eqx.apply_updates(params, updates)

        if bounded is not None:
            new_params = jax.tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': params})

        if has_aux:
            history.update({'aux': aux})

        return new_params, opt_state, rng, history
    return make_step


def evosax_make_step(static: PyTree,
                     optimizer: optax.GradientTransformationExtraArgs,
                     loss_fn: Callable,
                     bounded: Optional[Tuple[float, float]],
                     has_aux: bool,
                     include_input: bool,
                     pass_rng: bool,
                     popsize: int,
                     es_params: EvoParams,
                     **kwargs) -> MakeStepFunction:
    @eqx.filter_jit
    def make_step(params: eqx.Module, opt_state: optax.OptState,
                  rng: jax.Array, **batch
                  ) -> Tuple[PyTree, optax.OptState, jax.Array,
                             Dict[str, jax.Array]]:
        rng, rng_gen = jrd.split(rng)

        batched_loss_fn = partial(loss_fn, **batch)

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

        if has_aux:
            loss, aux = jax.vmap(batched_loss_fn)(inner_models, **rng_sample)
        else:
            loss = jax.vmap(batched_loss_fn)(inner_models, **rng_sample)

        opt_state = optimizer.tell(
            x=new_params,
            fitness=loss,
            state=opt_state,
            params=es_params
        )

        history = {'loss': loss}

        if include_input:
            history.update({'params': params})

        if has_aux:
            history.update({'aux': aux})

        return new_params, opt_state, rng, history
    return make_step


#                                                              L-BFGS optimizer
# =============================================================================


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
                use_scan: bool,
                **kwargs,
                ) -> Tuple[PyTree, optax.OptState]:

    rng, _ = jrd.split(seed)

    if opt_state is None:
        opt_state = optimizer.init(params)

    if use_scan:
        samples = unroll_batches(data_iter, inner_steps+1)

        scan_fn = lbfgs_scan(
            static=static,
            optimizer=optimizer,
            loss_fn=loss_fn,
            bounded=bounded,
            has_aux=has_aux,
            include_input=include_input,
            pass_rng=pass_rng,
        )

        (_, opt_state, _), history = jax.lax.scan(
            f=scan_fn,
            init=(params, opt_state, rng),
            xs=samples,
            length=inner_steps+1
        )

    else:
        make_step = lbfgs_make_step(
            static=static,
            optimizer=optimizer,
            loss_fn=loss_fn,
            bounded=bounded,
            has_aux=has_aux,
            include_input=include_input,
            pass_rng=pass_rng,
        )

        history_list = []
        iter_loop = zip(range(inner_steps+1), data_iter)

        for _, sample in tqdm(iter_loop, total=inner_steps+1):
            params, opt_state, rng, history = make_step(
                params=params, opt_state=opt_state,
                rng=rng, **sample)

            history_list.append(history)

        history = jax.tree.map(lambda *xs: jnp.vstack(xs), *history_list)

    return history, opt_state

# =============================================================================


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
                use_scan: bool,
                verbose: bool = False,
                **kwargs) -> Tuple[PyTree, optax.OptState]:

    rng, _ = jrd.split(seed)

    if opt_state is None:
        opt_state = optimizer.init(params)

    if use_scan:
        samples = unroll_batches(data_iter, inner_steps+1)

        scan_fn = optax_scan(
            static=static,
            optimizer=optimizer,
            loss_fn=loss_fn,
            bounded=bounded,
            has_aux=has_aux,
            include_input=include_input,
            pass_rng=pass_rng,
        )

        (_, opt_state, _), history = jax.lax.scan(
            scan_fn,
            (params, opt_state, rng),
            xs=samples,
            length=inner_steps+1,
        )

    else:
        make_step = optax_make_step(
            static=static,
            optimizer=optimizer,
            loss_fn=loss_fn,
            bounded=bounded,
            has_aux=has_aux,
            include_input=include_input,
            pass_rng=pass_rng,
        )

        history_list = []
        if verbose:
            iter_loop = tqdm(zip(range(inner_steps+1), data_iter),
                             total=inner_steps+1)
        else:
            iter_loop = zip(range(inner_steps+1), data_iter)

        for _, sample in iter_loop:
            params, opt_state, rng, history = make_step(
                params=params, opt_state=opt_state,
                rng=rng, **sample)

            history_list.append(history)

        history: dict = jax.tree.map(lambda *xs: jnp.vstack(xs), *history_list)

    return history, opt_state


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
                 use_scan: bool,
                 **kwargs
                 ) -> Tuple[PyTree, EvoState]:

    rng, rng_init = jrd.split(seed)

    es_params = optimizer.default_params

    if bounded is not None:
        es_params = es_params.replace(clip_min=bounded[0], clip_max=bounded[1])

    if opt_state is None:
        opt_state = optimizer.initialize(rng=rng_init, params=es_params)

    if use_scan:
        samples = unroll_batches(data_iter, inner_steps+1)

        scan_fn = evosax_scan(
            static=static,
            optimizer=optimizer,
            loss_fn=loss_fn,
            bounded=bounded,
            has_aux=has_aux,
            include_input=include_input,
            pass_rng=pass_rng,
            popsize=popsize,
            es_params=es_params
        )

        (_, opt_state, _), history = jax.lax.scan(
            f=scan_fn,
            init=(params, opt_state, rng),
            xs=samples,
            length=inner_steps+1,
        )

    else:
        make_step = evosax_make_step(
            static=static,
            optimizer=optimizer,
            loss_fn=loss_fn,
            bounded=bounded,
            has_aux=has_aux,
            include_input=include_input,
            pass_rng=pass_rng,
            popsize=popsize,
            es_params=es_params
        )

        history_list = []
        iter_loop = zip(range(inner_steps+1), data_iter)

        for _, sample in tqdm(iter_loop, total=inner_steps+1):
            params, opt_state, rng, history = make_step(
                params=params, opt_state=opt_state,
                rng=rng, **sample)

            history_list.append(history)

        history = jax.tree.map(lambda *xs: jnp.vstack(xs), *history_list)

    # Flatten population dimension
    history['params'] = tree_flatten_population_dim(
        history['params']) if include_input else None

    history['aux'] = tree_flatten_population_dim(
        history['aux']) if has_aux else None

    history['loss'] = history['loss'].flatten()

    return history, opt_state

# =============================================================================


# =============================================================================
OPT_MAPPING: Dict[str, Type[UpdateStep]] = {
    'adam': partial(UpdateStep, optimizer=optax.adam, update_fn=optax_adapt,
                    use_scan=True,
                    popsize=1),
    'adam_forloop': partial(UpdateStep, optimizer=optax.adam,
                            update_fn=optax_adapt,
                            popsize=1,
                            use_scan=False),
    'cmaes': partial(EvoSaxUpdate, optimizer=CMA_ES, update_fn=evosax_adapt,
                     use_scan=True),
    'cmaes_forloop': partial(EvoSaxUpdate, optimizer=CMA_ES,
                             update_fn=evosax_adapt, use_scan=False),
    'pso': partial(EvoSaxUpdate, optimizer=PSO, update_fn=evosax_adapt,
                   use_scan=True),
    'pso_forloop': partial(EvoSaxUpdate, optimizer=PSO,
                           update_fn=evosax_adapt, use_scan=False),
    'lbfgsb': partial(UpdateStep, optimizer=optax.lbfgs, update_fn=lbfgs_adapt,
                      popsize=1, use_scan=True),
    'lbfgsb_forloop': partial(UpdateStep, optimizer=optax.lbfgs,
                              update_fn=lbfgs_adapt,
                              popsize=1, use_scan=False),
}
