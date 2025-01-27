# Standard
import time
from functools import partial
from itertools import product
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Type)

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as onp
import optax
import xarray as xr
from evosax import CMA_ES, PSO, Strategy
from f3dasm import Block, ExperimentData
from f3dasm.design import Domain
from opfunu import BenchmarkFunc
from SALib.sample import latin

from ..._logging import logger

#                                                                    Schedulers
# =============================================================================


class OptimizerScheduler(NamedTuple):
    optimizer: str
    n_evals: int
    x0_strategy: str
    hyperparameters: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config):
        return cls(
            optimizer=config.optimizer,
            n_evals=config.n_evals,
            x0_strategy=config.x0_strategy,
            hyperparameters=config.hyperparameters
        )


def get_scheduler(
        optimizer_1: Dict[str, Any], optimizer_2: Dict[str, Any],
        switch: float, total_evals: int,
        first_x0: str = 'new') -> List[OptimizerScheduler]:
    """
    Get the scheduler for two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : _type_
        name of the second optimizer
    iterations : int
        total number of iterations across both optimizers
    switch : float, optional
        fraction of the iterations to use for the first optimizer
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'

    Returns
    -------
    List[Tuple[str, int, str]]
        List of schedulers for the two optimizers
    """

    i_0 = int(total_evals // (1/switch))
    i_1 = total_evals - i_0

    if optimizer_1['optimizer'] == optimizer_2['optimizer']:
        return [OptimizerScheduler(optimizer=optimizer_1['optimizer'],
                                   n_evals=total_evals,
                                   x0_strategy=first_x0,
                hyperparameters=optimizer_1['hyperparameters'])]

    return [OptimizerScheduler(optimizer=optimizer_1['optimizer'],
                               n_evals=i_0,
                               x0_strategy=first_x0,
                               hyperparameters=optimizer_1['hyperparameters']),
            OptimizerScheduler(
                optimizer=optimizer_2['optimizer'],
        n_evals=i_1,
        x0_strategy=get_handshake(
                    optimizer_1['optimizer'], optimizer_2['optimizer']),
        hyperparameters=optimizer_2['hyperparameters'])]


def get_handshake(optimizer_1: str, optimizer_2: str) -> str:
    """
    Return the handshake method to use between two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : str
        name of the optimizer to switch from

    Returns
    -------
    str
        method to determine the starting population of the second optimizer

    Notes
    -----
    The handshake method is either 'best', 'last', 'new' or 'random'
    """
    if optimizer_1.lower() == optimizer_2.lower():
        return 'last'

    if optimizer_2.lower() in ['lbfgsb', 'lbfgs', 'adam', 'randomsearch']:
        return 'best'

    if optimizer_1.lower() == 'randomsearch':
        return 'best'

    if optimizer_1.lower() == 'cmaes' and optimizer_2.lower() == 'pso':
        return 'last'

    if optimizer_1.lower() == 'pso' and optimizer_2.lower() == 'cmaes':
        return 'last'

    return 'new'


def get_all_combinations(
        optimizers: List[Dict[str, Any]], total_evals: int,
        first_x0: str = 'new',
        switch: float = 0.5) -> List[List[OptimizerScheduler]]:
    """
    Get all combination of two optimizer names

    Parameters
    ----------
    optimizers : List[str]
        list of optimizer names
    iterations : int
        total number of iterations across both optimizers
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'
    switch : float, optional
        fraction of the iterations to use for the first
        optimizer, by default 0.5

    Returns
    -------
    List[OptimizerScheduler]
        List of optimizer schedulers
    """
    return [get_scheduler(
        optimizer_1=opt_1, optimizer_2=opt_2,
        total_evals=total_evals, first_x0=first_x0,
        switch=switch)
        for opt_1, opt_2 in product(optimizers, repeat=2)]


def name_from_schedule(schedules: Iterable[OptimizerScheduler]) -> str:
    return '+'.join([s.optimizer for s in schedules])


#                                                            Converter function
# =============================================================================

def jnp_to_dict(jnp_array: jnp.ndarray, name: str = 'x') -> dict:
    if len(jnp_array.shape) == 1:
        return [{name: jnp_array}]
    return [{name: row} for row in jnp_array]


def jnp_to_dict_output(jnp_array: jnp.ndarray, name: str = 'y') -> dict:
    return [{name: row} for row in jnp_array]


def experimentdata_to_xarray(data: ExperimentData) -> xr.DataArray:
    return xr.DataArray(
        jnp.array([es.output_data['y'] for _, es in data]),
        dims='iterations', coords={'iterations': range(len(data))})

#                                                               Sampling Blocks
# =============================================================================


class LatinSampler(Block):
    def __init__(self, num_vars: int, lower_bound: float,
                 upper_bound: float, seed: int):
        self.seed = seed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.problem = {
            "num_vars": num_vars,
            "bounds": [[lower_bound, upper_bound]
                       for _ in range(num_vars)]
        }

    def call(self, data: ExperimentData, num_samples: int, **kwargs
             ) -> ExperimentData:

        samples = jnp.array(latin.sample(
            problem=self.problem, N=num_samples, seed=self.seed))
        return ExperimentData(input_data=jnp_to_dict(samples),
                              domain=data.domain,
                              project_dir=data.project_dir)


class RandomSampler(Block):
    def __init__(self, num_vars: int, seed: int, lower_bound: float = 0.0,
                 upper_bound: float = 1.0, max_samples: int = 10**3):
        self.seed = jrd.key(seed)
        self.num_vars = num_vars
        self.max_samples = max_samples
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def call(self, data: ExperimentData, num_samples: int, **kwargs
             ) -> ExperimentData:

        samples_pool = jrd.uniform(
            key=self.seed, shape=(self.max_samples, self.num_vars),
            minval=self.lower_bound, maxval=self.upper_bound)

        # samples = jrd.uniform(sample_seed, (num_samples, self.num_vars))
        return ExperimentData(
            input_data=jnp_to_dict(samples_pool[:num_samples]),
            domain=data.domain,
            project_dir=data.project_dir)

#                                                              Optimizer Blocks
# =============================================================================


class OptaxUpdate(Block):
    def __init__(self, optimizer: optax.GradientTransformation,
                 dimensionality: int,
                 bounds: jnp.ndarray,
                 seed: Optional[int] = None,
                 popsize: int = 1,
                 **hyperparameters):
        self.optimizer = optimizer(**hyperparameters)
        self.bounds = bounds

    def arm(self, data: ExperimentData) -> None:
        for k, v in data[-1:]:
            x = v.input_data['x']
        self.opt_state = self.optimizer.init(x)
        self.init_carry = (x, self.opt_state)

    def call(self, data: ExperimentData, fn: Callable, n_evals: int, **kwargs
             ) -> ExperimentData:
        value_and_grad_fun = jax.value_and_grad(fn)

        # Perform the scan
        _, (params_history, value_history) = jax.lax.scan(
            partial(optax_step, bounds=self.bounds,
                    value_and_grad_fun=value_and_grad_fun,
                    optimizer=self.optimizer),
            self.init_carry, xs=None,
            length=n_evals
        )

        return ExperimentData(
            domain=data.domain, input_data=jnp_to_dict(params_history),
            output_data=jnp_to_dict_output(value_history))


def optax_step(carry, _, bounds: jnp.ndarray, value_and_grad_fun: Callable,
               optimizer: optax.GradientTransformation):
    params, state = carry
    params = jnp.clip(params, bounds[:, 0], bounds[:, 1])

    value, grad = value_and_grad_fun(params)
    updates, state = optimizer.update(grad, state)

    params = optax.apply_updates(params, updates)

    params = jnp.clip(params, bounds[:, 0], bounds[:, 1])

    return (params, state), (params, value)  # Track loss value


class LBFGSBUpdate(Block):
    def __init__(self, optimizer: optax.GradientTransformation,
                 dimensionality: int,
                 bounds: jnp.ndarray,
                 seed: Optional[int] = None,
                 popsize: int = 1,
                 **hyperparameters):
        self.optimizer = optimizer(**hyperparameters)
        self.bounds = bounds

    def arm(self, data: ExperimentData) -> None:
        for k, v in data[-1:]:
            x = v.input_data['x']
        self.opt_state = self.optimizer.init(x)
        self.init_carry = (x, self.opt_state)

    def call(self, data: ExperimentData, fn: Callable, n_evals: int, **kwargs
             ) -> ExperimentData:
        value_and_grad_fun = optax.value_and_grad_from_state(fn)

        # Perform the scan
        _, (params_history, value_history) = jax.lax.scan(
            partial(lbfgs_step, bounds=self.bounds,
                    value_and_grad_fun=value_and_grad_fun,
                    fn=fn, optimizer=self.optimizer),
            self.init_carry, xs=None, length=n_evals
        )

        return ExperimentData(
            domain=data.domain, input_data=jnp_to_dict(params_history),
            output_data=jnp_to_dict_output(value_history))

    def exit(self):
        pass


def lbfgs_step(carry, _, bounds: jnp.ndarray, value_and_grad_fun: Callable,
               fn: Callable, optimizer: optax.GradientTransformationExtraArgs):
    params, state = carry
    params = jnp.clip(params, bounds[:, 0], bounds[:, 1])
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = optimizer.update(
        grad, state, params, value=value, grad=grad, value_fn=fn
    )
    params = optax.apply_updates(params, updates)

    params = jnp.clip(params, bounds[:, 0], bounds[:, 1])

    return (params, state), (params, value)  # Track loss value


class EvoSaxUpdate(Block):
    def __init__(self, optimizer: Strategy, popsize: int, dimensionality: int,
                 seed: int, bounds: jnp.ndarray, **hyperparameters):
        self.optimizer = optimizer(
            popsize=popsize, num_dims=dimensionality, **hyperparameters)
        self.popsize = popsize
        self.bounds = bounds
        self.seed = jrd.PRNGKey(seed)

    def arm(self, data: ExperimentData) -> None:
        # Initialize optimizer parameters and state
        self.es_params = self.optimizer.default_params
        self.opt_state = self.optimizer.initialize(self.seed, self.es_params)

        # Incorporate the last population from the data
        x = jnp.array([v.input_data['x'] for _, v in data[-self.popsize:]])
        y = jnp.array([v.output_data['y'] for _, v in data[-self.popsize:]])

        self.opt_state = self.optimizer.tell(
            x, y, self.opt_state, self.es_params)

        self.init_carry = (self.opt_state, self.seed)

    def call(self, data: ExperimentData, fn: Callable, n_evals: int, **kwargs
             ) -> ExperimentData:
        # value_fn = jax.vmap(fn)

        # Perform the scan
        _, (params_history, value_history) = jax.lax.scan(
            partial(evosax_step, bounds=self.bounds, value_fn=jax.vmap(fn),
                    optimizer=self.optimizer, es_params=self.es_params),
            self.init_carry, xs=None, length=n_evals
        )

        return ExperimentData(
            domain=data.domain, input_data=jnp_to_dict(
                params_history.reshape(-1, params_history.shape[-1])),
            output_data=jnp_to_dict_output(value_history.reshape(-1)))


def evosax_step(carry, _, bounds: jnp.ndarray, value_fn: Callable,
                optimizer: Strategy, es_params: Dict[str, Any]):
    state, rng = carry

    # Split RNG for reproducibility
    rng, rng_gen = jrd.split(rng)

    # Retrieve new candidates (ask step)
    params, state = optimizer.ask(
        rng_gen, state, es_params)

    value = value_fn(params)

    state = optimizer.tell(params, value, state, es_params)

    # Clip candidates to stay within bounds
    params = jnp.clip(params, bounds[:, 0], bounds[:, 1])

    return (state, rng), (params, value)


OPT_MAPPING: Dict[str, Type[Block]] = {
    'adam': partial(OptaxUpdate, optimizer=optax.adam),
    'cmaes': partial(EvoSaxUpdate, optimizer=CMA_ES),
    'pso': partial(EvoSaxUpdate, optimizer=PSO),
    'lbfgsb': partial(LBFGSBUpdate, optimizer=optax.lbfgs),
}

# =============================================================================


def run_optimization_schedule_multiple_realizations(
        function_name: str,
        optimizer_schedules: Iterable[OptimizerScheduler],
        dimensionality: int,
        noise: float,
        seed: int,
        n_realizations: int,
) -> xr.DataArray:
    t1 = time.perf_counter()
    bounds = jnp.tile(jnp.array([[0.0, 1.0]]), (dimensionality, 1))

    # Benchmark function
    benchmark_func = BenchmarkFunc(
        fn_name=function_name, dimensionality=dimensionality,
        scale_bounds=bounds, seed=seed, offset=True, noise=noise)

    results_list: List[xr.DataArray] = []

    for s in range(n_realizations):
        data = run_optimization_schedule(
            benchmark_func=benchmark_func,
            optimizer_schedules=optimizer_schedules,
            seed=seed + s,
            bounds=bounds,
        )

        results_list.append(experimentdata_to_xarray(data))

    results = xr.concat(results_list, xr.DataArray(range(n_realizations),
                                                   dims='realization'))

    name = onp.array([name_from_schedule(optimizer_schedules)], dtype=object)
    logger.info((
        f'Optimized {dimensionality}D {function_name} function '
        f'with {name_from_schedule(optimizer_schedules)}'
        f' (took {time.perf_counter() - t1:.2f} seconds)')
    )
    return results.expand_dims(
        dim={'output_dim': ['y']}).to_dataset(
            name='output').assign_coords({'optimizer': name})


def run_optimization_schedule(
        benchmark_func: Block,
        optimizer_schedules: Iterable[OptimizerScheduler],
        seed: int,
        bounds: jnp.ndarray,
) -> ExperimentData:

    dimensionality = len(bounds)

    # Start with empty ExperimentData
    data = None

    # Sampler
    sampler = RandomSampler(
        lower_bound=bounds[0][0], upper_bound=bounds[0][1],
        num_vars=dimensionality, seed=seed)

    for schedule in optimizer_schedules:

        update_step = OPT_MAPPING[schedule.optimizer](
            **schedule.hyperparameters, seed=seed,
            bounds=bounds, dimensionality=dimensionality)

        if 'popsize' in schedule.hyperparameters:
            popsize = schedule.hyperparameters['popsize']
        else:
            popsize = 1

        data = run_optimization(
            benchmark_func=benchmark_func,
            update_step=update_step,
            sampler=sampler,
            seed=seed,
            n_evals=schedule.n_evals,
            popsize=popsize,
            x0_strategy=schedule.x0_strategy,
            data=data,

        )

    return data


def run_optimization(
    benchmark_func: Block,
    update_step: Block,
    sampler: Block,
    seed: int,
    n_evals: int,
    popsize: int = 1,
    x0_strategy: str = 'new',
    data: Optional[ExperimentData] = None,
    **hyperparameters
) -> ExperimentData:

    if data is None:
        # Create data
        domain = Domain()
        domain.add_parameter(name='x')
        domain.add_output(name='y')
        data = ExperimentData(domain=domain)

    # Create x0
    x0 = create_x0(data=data, mode=x0_strategy, n_samples=popsize,
                   sampler=sampler)

    if x0_strategy == 'new':
        n_evals -= popsize
        data += x0

    # Evaluate samples
    benchmark_func.arm(data=x0)
    x0 = benchmark_func.call(data=x0)

    # Arm update step
    update_step.arm(x0)

    n_updates = n_evals // popsize + (n_evals % popsize > 0)

    data += update_step.call(
        data=data, fn=benchmark_func, n_evals=n_updates)

    # Remove overiterations
    data.remove_rows_bottom(number_of_rows=popsize * n_updates - n_evals)

    return data


def create_x0(data: ExperimentData, mode: str, n_samples: int,
              sampler: Block) -> ExperimentData:

    if mode == 'new':
        data += sampler.call(data=data, num_samples=n_samples)

    elif mode == 'last':
        data = data[-n_samples:]

    elif mode == 'best':
        data = data.sort(
            criterion=lambda es: es.output_data['y'],
            reverse=True)[-n_samples:].reset_index()

    return data
