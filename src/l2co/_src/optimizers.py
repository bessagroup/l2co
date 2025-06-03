
# Standard
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from evosax import CMA_ES, PSO, EvoParams, EvoState, Sep_CMA_ES, Strategy
from f3dasm import Block, ExperimentData
from jax.tree_util import tree_map
from jaxtyping import PyTree
from tqdm import tqdm

# Local
from ._typing import InitFunction, LossFunction, PostFunction, ScanFunction
from .batching import batch_iter, unroll_batches
from .task import Task
from .tree_utils import (dict_list_to_tree, tree_flatten_population_dim,
                         tree_to_dict_list)

# =============================================================================


#                                                                       Modules
# =============================================================================


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

SAMPLER_HASH = '0000000000000000'


class EvaluateModel(Block):
    def __init__(self, task: Task, seed: int):
        self.task = task
        self.seed = seed

    def call(self, data: ExperimentData) -> ExperimentData:
        _input_data = [v.input_data for k, v in data]
        _output_data = [v.output_data for k, v in data]
        models = dict_list_to_tree(_input_data, name='x')

        inner_steps = len(data)

        if self.task.pass_rng:
            rng_sample = {'key': jr.split(
                jr.key(self.seed), num=inner_steps)}
        else:
            rng_sample = {}

        data_iter = batch_iter(dataset=self.task.loaded_dataset,
                               n_evals=inner_steps,
                               batch_size=self.task.batch_size,
                               key=self.task.dataset['seed'])

        # Option 1: There is training data but no batching
        if isinstance(data_iter, dict):
            loss_fn = jax.vmap(partial(self.task.loss_fn, **data_iter))

            if self.task.has_aux:
                y, aux = loss_fn(models, **rng_sample)
            else:
                y = loss_fn(models, **rng_sample)

        else:
            samples = unroll_batches(data_iter, inner_steps)

            loss_fn = jax.vmap(self.task.loss_fn)

            if self.task.has_aux:
                y, aux = loss_fn(models, **samples, **rng_sample)
            else:
                y = loss_fn(models, **samples, **rng_sample)

        if len(_output_data) > 0:
            output_data = [{**out1, **out2}
                           for out1, out2 in zip(
                               _output_data, tree_to_dict_list(y, name='y'))]
        else:
            output_data = tree_to_dict_list(y, name='y')

        return ExperimentData(domain=data.domain, input_data=_input_data,
                              output_data=output_data,
                              project_dir=data.project_dir)


class UpdateStep(Block):
    """
    Update step for optimization.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        The optimizer to use.
    seed : int
        Random seed for reproducibility.
    popsize : int
        Population size for optimization.
    update_fn : Callable
        Function to define the update step.
    include_input : bool
        Whether to include input in the history.
    pass_rng : bool
        Whether to pass random number generator.
    bounded : Optional[Tuple[float, float]], optional
        Bounds for the parameters, by default None.
    task : Optional[Task], optional
        Task to optimize, by default None.
    verbose : bool, optional
        Whether to display progress, by default False.
    **hyperparameters : dict
        Additional hyperparameters for the optimizer.
    """

    def __init__(self, optimizer: Callable[..., optax.GradientTransformation],
                 seed: int,
                 popsize: int,
                 update_fn: Callable[
                     [],
                     Tuple[ScanFunction, InitFunction, PostFunction]],
                 include_input: bool,
                 opt_hash: str,
                 name: str,
                 bounded: Optional[Tuple[float, float]] = None,
                 task: Optional[Task] = None,
                 verbose: bool = False,
                 **hyperparameters):

        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = seed
        self.opt_state = None
        self.verbose = verbose
        self.task = task
        self.name = name
        self.opt_hash = opt_hash
        self.optimizer = optimizer(**hyperparameters)

    def arm(self, data: ExperimentData, previous_step: Optional[UpdateStep],
            sampler: Block, seed: int) -> None:
        """
        Prepare the optimizer for training.

        Parameters
        ----------
        data : ExperimentData
            Experiment data for optimization.

        Notes
        -----
        The passed f3dasm.ExperimentData needs an input data parameter 'x'
        that is a PyTree of the parameters to be optimized.

        The loss function provided should be a pure function that can be jitted
        and accepts as first argument the parameters to be optimized.
        The loss function should return a scalar loss value.
        """

        x0_strategy = get_handshake(previous_step=previous_step,
                                    current_step=self)

        # Create or select the initial solutions
        x0, self.has_created_flag = create_x0(
            data=data, model=self.task.model,
            mode=x0_strategy, n_samples=self.popsize,
            sampler=sampler, seed=seed
        )

        if self.has_created_flag:
            data_generator = EvaluateModel(task=self.task, seed=self.seed)
            data_generator.arm(x0)
            x0 = data_generator.call(data=x0)

        # Set the params
        for k, v in x0[-self.popsize:]:
            x = v.input_data['x']
        self.params, self.static = eqx.partition(x, eqx.is_inexact_array)

        self.make_step, self.init_fn, self.post_fn = self.update_fn(
            static=self.static,
            params=self.params,
            optimizer=self.optimizer,
            loss_fn=self.task.loss_fn,
            bounded=self.bounded,
            has_aux=self.task.has_aux,
            include_input=self.include_input,
            pass_rng=self.task.pass_rng,
            opt_hash=self.opt_hash,
        )

        if self.opt_state is None:
            self.opt_state = self.init_fn()

        self.opt_state = get_state_handshake(previous_step=previous_step,
                                             current_step=self)

    def call(self, data: ExperimentData, n_iterations: int,
             data_iter: Iterator, use_scan: bool = True,
             **kwargs) -> ExperimentData:
        """
        Perform optimization steps.

        Parameters
        ----------
        data : ExperimentData
            Experiment data for optimization.
        n_iterations : int
            Number of iterations to perform.
        data_iter : Iterator
            Iterator for batching data.
        use_scan : bool, optional
            Whether to use JAX scan for optimization, by default True.
        **kwargs : dict
            Additional arguments for the optimization.

        Returns
        -------
        ExperimentData
            Updated experiment data after optimization.
        """
        key = jr.key(self.seed)
        rng, _ = jr.split(key)

        # Option 1: There is no batching, hence we fix the sample and
        # use scan
        if use_scan:
            (_, self.opt_state, _), history = jax.lax.scan(
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

            history = tree_map(lambda *xs: jnp.vstack(xs), *history_list)

        input_data, output_data = self.post_fn(history)

        for i in range(self.popsize):
            output_data[i]['opt_step'] = SAMPLER_HASH

        return ExperimentData(
            domain=data.domain,
            input_data=input_data,
            output_data=output_data,
            project_dir=data.project_dir)


class EvoSaxUpdate(UpdateStep):
    """
    Update step for evolutionary strategies using EvoSax.

    Parameters
    ----------
    optimizer : Strategy
        Evolutionary strategy to use.
    seed : int
        Random seed for reproducibility.
    popsize : int
        Population size for optimization.
    update_fn : Callable
        Function to define the update step.
    include_input : bool
        Whether to include input in the history.
    pass_rng : bool
        Whether to pass random number generator.
    bounded : Optional[Tuple[float, float]], optional
        Bounds for the parameters, by default None.
    verbose : bool, optional
        Whether to display progress, by default False.
    task : Optional[Task], optional
        Task to optimize, by default None.
    **hyperparameters : dict
        Additional hyperparameters for the optimizer.
    """

    def __init__(self, optimizer: Type[Strategy],
                 seed: int,
                 popsize: int,
                 update_fn: Callable[[], Tuple[ScanFunction, InitFunction]],
                 include_input: bool,
                 name: str,
                 opt_hash: str,
                 bounded: Optional[Tuple[float, float]] = None,
                 verbose: bool = False,
                 task: Optional[Task] = None,
                 **hyperparameters):
        self.popsize = popsize
        self.bounded = bounded
        self.include_input = include_input
        self.update_fn = update_fn
        self.seed = seed
        self.opt_state = None
        self.optimizer_class = optimizer
        self.verbose = verbose
        self.task = task
        self.hyperparameters = hyperparameters
        self.name = name
        self.opt_hash = opt_hash

    def arm(self, data: ExperimentData, previous_step: Optional[UpdateStep],
            sampler: Block, seed: int) -> None:
        """
        Prepare the evolutionary strategy for training.

        Parameters
        ----------
        data : ExperimentData
            Experiment data for optimization.
        fn : LossFunction
            Loss function to optimize.
        has_aux : bool, optional
            Whether the loss function returns auxiliary data, by default False.

        Notes
        -----
        The passed f3dasm.ExperimentData needs an input data parameter 'x'
        that is a PyTree of the parameters to be optimized.

        The loss function provided should be a pure function that can be jitted
        and accepts as first argument the parameters to be optimized.
        The loss function should return a scalar loss value.
        """

        x0_strategy = get_handshake(previous_step=previous_step,
                                    current_step=self)

        # Create or select the initial solutions
        x0, self.has_created_flag = create_x0(
            data=data, model=self.task.model,
            mode=x0_strategy, n_samples=self.popsize,
            sampler=sampler, seed=seed
        )

        if self.has_created_flag:
            data_generator = EvaluateModel(task=self.task, seed=self.seed)
            data_generator.arm(x0)
            x0 = data_generator.call(data=x0)

        x = [v.input_data for k, v in x0[-self.popsize:]]
        pop_x = dict_list_to_tree(x, name='x')

        self.params, self.static = eqx.partition(pop_x, eqx.is_inexact_array)

        self.optimizer = self.optimizer_class(
            popsize=self.popsize,
            pholder_params=tree_map(lambda x: x[0], self.params),
            **self.hyperparameters)

        es_params = self.optimizer.default_params

        if self.bounded is not None:
            es_params = es_params.replace(
                clip_min=self.bounded[0], clip_max=self.bounded[1])

        self.make_step, self.init_fn, self.post_fn = self.update_fn(
            static=self.static,
            optimizer=self.optimizer,
            loss_fn=self.task.loss_fn,
            bounded=self.bounded,
            has_aux=self.task.has_aux,
            include_input=self.include_input,
            pass_rng=self.task.pass_rng,
            popsize=self.popsize,
            es_params=es_params,
            opt_hash=self.opt_hash,
            key=jr.key(self.seed),
        )

        if self.opt_state is None:
            self.opt_state = self.init_fn()

        self.opt_state = get_state_handshake(previous_step=previous_step,
                                             current_step=self)

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
               opt_hash: str,
               **kwargs
               ) -> Tuple[ScanFunction, InitFunction, PostFunction]:
    """
    Define the scan function for L-BFGS optimization.

    Parameters
    ----------
    static : PyTree
        Static parameters for the model.
    params : PyTree
        Initial parameters for optimization.
    optimizer : optax.GradientTransformationExtraArgs
        Optimizer for L-BFGS.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : Optional[Tuple[float, float]]
        Bounds for the parameters.
    has_aux : bool
        Whether the loss function returns auxiliary data.
    include_input : bool
        Whether to include input in the history.
    pass_rng : bool
        Whether to pass random number generator.
    **kwargs : dict
        Additional arguments for the scan function.

    Returns
    -------
    Tuple[ScanFunction, InitFunction, PostFunction]
        Functions for scanning, initialization, and post-processing.
    """
    @eqx.filter_jit
    def scan_fn(carry: Tuple[PyTree, optax.OptState, jax.Array],
                sample: dict, *args, **kwargs
                ) -> Tuple[Tuple[PyTree, optax.OptState, jax.Array], dict]:
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
            new_params = tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': inner_params})

        return (new_params, opt_state, rng), history

    def init_fn() -> optax.OptState:
        return optimizer.init(params)

    def post_fn(history: PyTree
                ) -> Tuple[List[Dict[str, PyTree]] | None, List[
                    Dict[str, PyTree]]]:
        if include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, static), history['params'])
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(history['loss'], name='y')
        _output_data = [{'opt_step': opt_hash, **out} for out in output_data]
        return input_data, _output_data

    return scan_fn, init_fn, post_fn


def optax_scan(static: PyTree,
               params: PyTree,
               optimizer: optax.GradientTransformation,
               loss_fn: LossFunction,
               bounded: Optional[Tuple[float, float]],
               has_aux: bool,
               include_input: bool,
               pass_rng: bool,
               opt_hash: str,
               **kwargs
               ) -> Tuple[ScanFunction, InitFunction, PostFunction]:
    """
    Define the scan function for Optax optimization.

    Parameters
    ----------
    static : PyTree
        Static parameters for the model.
    params : PyTree
        Initial parameters for optimization.
    optimizer : optax.GradientTransformation
        Optimizer for Optax.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : Optional[Tuple[float, float]]
        Bounds for the parameters.
    has_aux : bool
        Whether the loss function returns auxiliary data.
    include_input : bool
        Whether to include input in the history.
    pass_rng : bool
        Whether to pass random number generator.
    **kwargs : dict
        Additional arguments for the scan function.

    Returns
    -------
    Tuple[ScanFunction, InitFunction, PostFunction]
        Functions for scanning, initialization, and post-processing.
    """
    @eqx.filter_jit
    def scan_fn(carry: Tuple[PyTree, optax.OptState, jax.Array],
                sample: dict, *args, **kwargs
                ) -> Tuple[Tuple[PyTree, optax.OptState, jax.Array], dict]:
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
            new_params = tree_map(
                lambda p: jnp.clip(p, bounded[0], bounded[1]), new_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': inner_params})

        if has_aux:
            history.update({'aux': aux})

        return (new_params, opt_state, rng), history

    def init_fn() -> optax.OptState:
        return optimizer.init(params)

    def post_fn(history: PyTree
                ) -> Tuple[List[Dict[str, PyTree]] | None, List[
                    Dict[str, PyTree]]]:
        if include_input:
            models = jax.lax.map(lambda p: eqx.combine(
                p, static), history['params'])
            input_data = tree_to_dict_list(models, name='x')
        else:
            input_data = None

        output_data = tree_to_dict_list(history['loss'], name='y')
        _output_data = [{'opt_step': opt_hash, **out} for out in output_data]
        return input_data, _output_data

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
                key: jax.Array,
                opt_hash: str,
                **kwargs
                ) -> Tuple[ScanFunction, InitFunction, PostFunction]:
    """
    Define the scan function for EvoSax optimization.

    Parameters
    ----------
    static : PyTree
        Static parameters for the model.
    optimizer : Strategy
        Evolutionary strategy to use.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : Optional[Tuple[float, float]]
        Bounds for the parameters.
    has_aux : bool
        Whether the loss function returns auxiliary data.
    include_input : bool
        Whether to include input in the history.
    pass_rng : bool
        Whether to pass random number generator.
    popsize : int
        Population size for optimization.
    es_params : EvoParams
        Parameters for the evolutionary strategy.
    key : jax.Array
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments for the scan function.

    Returns
    -------
    Tuple[ScanFunction, InitFunction, PostFunction]
        Functions for scanning, initialization, and post-processing.
    """
    def combined_loss(params: PyTree, **rng_sample: dict) -> jax.Array:
        return loss_fn(eqx.combine(params, static), **rng_sample)

    @eqx.filter_jit
    def scan_fn(carry: Tuple[PyTree, EvoState, jax.Array],
                sample: dict, *args, **kwargs) -> Tuple[
                    Tuple[PyTree, EvoState, jax.Array], dict]:
        inner_params, opt_state, rng = carry

        rng, rng_gen = jr.split(rng)

        if pass_rng:
            rng_sample = {'key': jr.split(rng, num=popsize)}
        else:
            rng_sample = {}

        loss = jax.vmap(partial(combined_loss, **sample)
                        )(inner_params, **rng_sample)

        opt_state = optimizer.tell(
            x=inner_params,
            fitness=loss,
            state=opt_state,
            params=es_params
        )

        new_params, opt_state = optimizer.ask(
            rng=rng_gen,
            state=opt_state,
            params=es_params)

        history = {'loss': loss}

        if include_input:
            history.update({'params': inner_params})

        return (new_params, opt_state, rng), history

    def init_fn() -> EvoState:
        return optimizer.initialize(rng=key, params=es_params)

    def post_fn(history: PyTree
                ) -> Tuple[List[Dict[str, PyTree]] | None, List[
                    Dict[str, PyTree]]]:
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
        _output_data = [{'opt_step': opt_hash, **out} for out in output_data]
        return input_data, _output_data

    return scan_fn, init_fn, post_fn

# =============================================================================


def get_handshake(previous_step: Optional[UpdateStep],
                  current_step: UpdateStep) -> str:
    """
    Return the handshake method to use between two optimizers

    Parameters
    ----------
    previous_step : Optional[UpdateStep]
        name of the first optimizer
    current_step : UpdateStep
        name of the optimizer to switch from

    Returns
    -------
    str
        method to determine the starting population of the second optimizer

    Notes
    -----
    The handshake method is either 'best', 'last', 'new' or 'random'
    """

    if previous_step is None:
        return 'new'

    previous_step_name = previous_step.name
    current_step_name = current_step.name

    if previous_step_name.lower() == current_step_name.lower():
        return 'last'

    if current_step_name.lower() in ['lbfgsb',
                                     'lbfgs', 'adam', 'randomsearch']:
        return 'best'

    if previous_step_name.lower() == 'randomsearch':
        return 'best'

    if current_step_name.lower() in ['cmaes', 'pso', 'sepcmaes']:
        return 'best_one'

    if (previous_step_name.lower() == 'cmaes'
            and current_step_name.lower() == 'pso'):
        return 'last'

    if (previous_step_name.lower() == 'pso'
            and current_step_name.lower() == 'cmaes'):
        return 'last'

    return 'new'


def get_state_handshake(previous_step: Optional[UpdateStep],
                        current_step: UpdateStep):
    return current_step.opt_state


def create_x0(data: Optional[ExperimentData], model: PyTree,
              mode: str, n_samples: int, sampler: Block,
              seed: int) -> Tuple[ExperimentData, bool]:
    """
    Create the initial solutions for optimizing a problem according to some
    initial strategy.

    Parameters
    ----------
    data : Optional[ExperimentData]
        The ExperimentData object to create the initial solutions from.
        If None, a new ExperimentData object will be created.
    model : PyTree
        The structure of the model to create input parameters for
    mode : str
        The strategy to which we create the initial samples
    n_samples : int
        number of initial solutions to create
    sampler : Block
        Sampler object to use as sampling technique
    seed : int
        Seed for the pseudo-random number generator

    Returns
    -------
    ExperimentData
        The initial solutions in ExperimentData format
    """

    has_created_flag = False

    if mode == 'new':
        _data = sampler.call(data=data, model=model,
                             num_samples=n_samples, seed=seed)
        has_created_flag = True

    elif mode == 'last':
        _data = data[-n_samples:]

    elif mode == 'best':
        _data = data.sort(
            criterion=lambda es: es.output_data['y'],
            reverse=True)[-n_samples:].reset_index()

    elif mode == 'best_one':
        _data = data.sort(
            criterion=lambda es: es.output_data['y'],
            reverse=True)[-1:].reset_index()

    # Check if enough samples are returned
    if len(_data) < n_samples:
        _additional_samples, has_created_flag = create_x0(
            data=_data, model=model, mode='new',
            n_samples=n_samples - len(_data),
            sampler=sampler, seed=seed)
        _data += _additional_samples

    return _data, has_created_flag


def choose_es(task: Task, popsize: Optional[int] = None, **kwargs) -> Strategy:
    """
    Choose an evolutionary strategy based on task dimensionality.

    Parameters
    ----------
    task : Task
        Task to optimize.
    popsize : Optional[int], optional
        Population size for optimization, by default None.
    **kwargs : dict
        Additional arguments for the strategy.

    Returns
    -------
    Strategy
        Selected evolutionary strategy.
    """
    d = task.dimensionality
    if d > 400:
        optimizer = partial(
            EvoSaxUpdate,
            optimizer=Sep_CMA_ES, update_fn=evosax_scan, name='sepcmaes'
        )
    else:
        optimizer = partial(
            EvoSaxUpdate,
            optimizer=CMA_ES, update_fn=evosax_scan, name='cmaes'
        )

    popsize = int(4+3*jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


def sepcmaes_variable_popsize(
        task: Task, popsize: Optional[int] = None, **kwargs) -> Strategy:
    d = task.dimensionality

    optimizer = partial(
        EvoSaxUpdate,
        optimizer=Sep_CMA_ES, update_fn=evosax_scan, name='sepcmaes'
    )

    popsize = int(4+3*jnp.log(d))

    return optimizer(**kwargs, task=task, popsize=popsize)


# =============================================================================

OPT_MAPPING: Dict[str, Type[UpdateStep]] = {
    'adam': partial(
        UpdateStep,
        optimizer=optax.adam, update_fn=optax_scan,
        popsize=1, name='adam'),
    'cmaes': choose_es,
    'sepcmaes': sepcmaes_variable_popsize,
    'pso': partial(
        EvoSaxUpdate,
        optimizer=PSO, update_fn=evosax_scan, name='pso'
    ),
    'lbfgsb': partial(
        UpdateStep,
        optimizer=optax.lbfgs, update_fn=lbfgs_scan,
        popsize=1, name='lbfgsb'),
}
