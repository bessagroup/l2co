from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
from evosax import CMA_ES, PSO, EvoParams, EvoState, Strategy
from f3dasm import Block, ExperimentData
from jaxtyping import PyTree
from tqdm import tqdm

from ._typing import InitFunction, LossFunction, PostFunction, ScanFunction
from .batching import NoneIterator
from .tree_utils import (dict_list_to_tree, tree_flatten_population_dim,
                         tree_to_dict_list)


# =============================================================================
class UpdateStep(Block):

    def __init__(self, optimizer: optax.GradientTransformation,
                 seed: int,
                 popsize: int,
                 update_fn: Callable[
                     [],
                     Tuple[ScanFunction, InitFunction, PostFunction]],
                 include_input: bool,
                 pass_rng: bool,
                 bounded: Optional[Tuple[float, float]] = None,
                 verbose: bool = False,
                 **hyperparameters):

        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = jr.key(seed)
        self.opt_state = None
        self.pass_rng = pass_rng
        self.verbose = verbose
        self.optimizer = optimizer(**hyperparameters)

    def arm(self, data: ExperimentData, fn: LossFunction, has_aux: bool = False
            ) -> None:
        for k, v in data[-self.popsize:]:
            x = v.input_data['x']
        self.params, self.static = eqx.partition(x, eqx.is_inexact_array)

        self.make_step, self.init_fn, self.post_fn = self.update_fn(
            static=self.static,
            params=self.params,
            optimizer=self.optimizer,
            loss_fn=fn,
            bounded=self.bounded,
            has_aux=has_aux,
            include_input=self.include_input,
            pass_rng=self.pass_rng,
        )

    def call(self, data: ExperimentData, n_iterations: int,
             data_iter: Iterator, **kwargs) -> ExperimentData:

        rng, _ = jr.split(self.seed)

        if self.opt_state is None:
            self.opt_state = self.init_fn()

        # Option 1: There is no batching, hence we fix the sample and
        # use scan
        if isinstance(data_iter, NoneIterator):
            (_, opt_state, _), history = jax.lax.scan(
                f=lambda carry, _: self.make_step(
                    carry, sample=data_iter.data),
                init=(self.params, self.opt_state, rng),
                xs=None,
                length=n_iterations+1
            )

        # Option 2: There is training data and batching
        else:
            history_list = []
            iter_loop = tqdm(zip(range(n_iterations+1), data_iter),
                             total=n_iterations+1) if self.verbose else zip(
                                 range(n_iterations+1), data_iter)

            for _, sample in iter_loop:
                (self.params, self.opt_state, rng), history = self.make_step(
                    (self.params, self.opt_state, rng), sample)

                history_list.append(history)

            history = jax.tree.map(lambda *xs: jnp.vstack(xs), *history_list)

        input_data, output_data = self.post_fn(history)

        return ExperimentData(
            domain=data.domain,
            input_data=input_data,
            output_data=output_data,
            project_dir=data.project_dir)


class EvoSaxUpdate(UpdateStep):
    def __init__(self, optimizer: Strategy,
                 seed: int,
                 popsize: int,
                 update_fn: Callable[[], Tuple[ScanFunction, InitFunction]],
                 include_input: bool,
                 pass_rng: bool,
                 bounded: Optional[Tuple[float, float]] = None,
                 verbose: bool = False,
                 **hyperparameters):
        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = jr.key(seed)
        self.opt_state = None
        self.pass_rng = pass_rng
        self.optimizer_class = optimizer
        self.verbose = verbose
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData, fn: LossFunction,
            has_aux: bool = False) -> None:

        self.has_aux = has_aux

        x = [v.input_data for k, v in data[-self.popsize:]]
        pop_x = dict_list_to_tree(x, name='x')

        self.params, self.static = eqx.partition(pop_x, eqx.is_inexact_array)

        self.optimizer = self.optimizer_class(
            popsize=self.popsize,
            pholder_params=jt.map(lambda x: x[0], self.params),
            **self.hyperparameters)

        rng, rng_init = jr.split(self.seed)

        es_params = self.optimizer.default_params

        if self.bounded is not None:
            es_params = es_params.replace(
                clip_min=self.bounded[0], clip_max=self.bounded[1])

        self.make_step, self.init_fn, self.post_fn = self.update_fn(
            static=self.static,
            optimizer=self.optimizer,
            loss_fn=fn,
            bounded=self.bounded,
            has_aux=has_aux,
            include_input=self.include_input,
            pass_rng=self.pass_rng,
            popsize=self.popsize,
            es_params=es_params,
            seed=self.seed,
        )

#                                                                Scan functions
# =============================================================================


def lbfgs_scan(static: PyTree,
               params: PyTree,
               optimizer: optax.GradientTransformationExtraArgs,
               loss_fn: LossFunction,
               bounded: Optional[Tuple[float, float]],
               has_aux: bool,
               include_input: bool,
               pass_rng: bool,
               **kwargs
               ) -> Tuple[ScanFunction, InitFunction, PostFunction]:

    @jax.jit
    def scan_fn(carry, sample, *args, **kwargs):
        inner_params, opt_state, rng = carry

        if pass_rng:
            rng, _ = jr.split(rng)
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

    def init_fn() -> optax.OptState:
        return optimizer.init(params)

    def post_fn(history: PyTree) -> Tuple[List[Dict[str, PyTree] | None,
                                               List[Dict[str, PyTree]]]]:
        if include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, static), history['params'])
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(history['loss'], name='y')
        return input_data, output_data

    return scan_fn, init_fn, post_fn


def optax_scan(static: PyTree,
               params: PyTree,
               optimizer: optax.GradientTransformation,
               loss_fn: LossFunction,
               bounded: Optional[Tuple[float, float]],
               has_aux: bool,
               include_input: bool,
               pass_rng: bool,
               **kwargs
               ) -> Tuple[ScanFunction, InitFunction, PostFunction]:

    @jax.jit
    def scan_fn(carry, sample, *args, **kwargs):
        inner_params, opt_state, rng = carry

        model = eqx.combine(inner_params, static)

        if pass_rng:
            rng, _ = jr.split(rng)
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

    def init_fn() -> optax.OptState:
        return optimizer.init(params)

    def post_fn(history: PyTree) -> Tuple[List[Dict[str, PyTree] | None,
                                               List[Dict[str, PyTree]]]]:
        if include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, static), history['params'])
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(history['loss'], name='y')
        return input_data, output_data

    return scan_fn, init_fn, post_fn


def evosax_scan(static: PyTree,
                optimizer: Strategy,
                loss_fn: LossFunction,
                bounded: Optional[Tuple[float, float]],
                has_aux: bool,
                include_input: bool,
                pass_rng: bool,
                popsize: int,
                es_params: EvoParams,
                seed: jax.Array,
                **kwargs
                ) -> Tuple[ScanFunction, InitFunction, PostFunction]:

    @jax.jit
    def scan_fn(carry, sample, *args, **kwargs):
        inner_params, opt_state, rng = carry

        rng, rng_gen = jr.split(rng)

        new_params, opt_state = optimizer.ask(
            rng=rng_gen,
            state=opt_state,
            params=es_params)

        if pass_rng:
            rng_sample = {'key': jr.split(rng, num=popsize)}
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

    def init_fn() -> EvoState:
        return optimizer.initialize(rng=seed, params=es_params)

    def post_fn(history: PyTree
                ) -> Tuple[List[Dict[str, PyTree] | None,
                                List[Dict[str, PyTree]]]]:
        # Flatten population dimension
        history['params'] = tree_flatten_population_dim(
            history['params']) if include_input else None

        history['aux'] = tree_flatten_population_dim(
            history['aux']) if has_aux else None

        history['loss'] = history['loss'].flatten()

        if include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, static), history['params'])
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(history['loss'], name='y')
        return input_data, output_data

    return scan_fn, init_fn, post_fn

# =============================================================================


# =============================================================================
OPT_MAPPING: Dict[str, Type[UpdateStep]] = {
    'adam': partial(
        UpdateStep,
        optimizer=optax.adam, update_fn=optax_scan,
        popsize=1),
    'cmaes': partial(
        EvoSaxUpdate,
        optimizer=CMA_ES, update_fn=evosax_scan,
    ),
    'pso': partial(
        EvoSaxUpdate,
        optimizer=PSO, update_fn=evosax_scan,
    ),
    'lbfgsb': partial(
        UpdateStep,
        optimizer=optax.lbfgs, update_fn=lbfgs_scan,
        popsize=1),
}
