# Standard
import gc
import json
from pathlib import Path
from time import time
from typing import Callable, Iterator, List, Optional, Tuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.tree as jt
import numpy as onp
import xarray as xr
from f3dasm import Block, ExperimentData, ExperimentSample
from f3dasm.datageneration import DataGenerator
from f3dasm.design import Domain
from jaxtyping import PyTree
from SALib.sample import latin

from ..._logging import logger
from ...data_loader import OptimizationDynamics
from ...l2co_model import L2CO_model
from ...optimizerschedule import OptimizationSchedule, OptimizationStep
from .optimizers import OPT_MAPPING
from .task import Task
from .tree_utils import tree_to_dict_list

#                                                            Converter function
# =============================================================================


def experimentdata_to_xarray(data: ExperimentData,
                             include_input: bool = False) -> xr.Dataset:
    y_arr = jnp.atleast_2d([es.output_data['y']
                            for _, es in data])

    y = xr.DataArray(
        y_arr,
        dims=['output_dim', 'iterations'],
        coords={'iterations': range(y_arr.shape[1]),
                'output_dim': ['y']})
    if include_input:

        x_arr = jnp.array([es.input_data['x']
                          for _, es in data])

        x = xr.DataArray(
            x_arr.T,
            dims=['input_dim', 'iterations'],
            coords={'iterations': range(x_arr.shape[0]),
                    'input_dim': [f"x{i}" for i in range(x_arr.shape[1])]})

        return xr.Dataset({'input': x,
                           'output': y})

    else:
        return xr.Dataset({'output': y})

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
        return ExperimentData(
            input_data=tree_to_dict_list(samples, name='x'),
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

        return ExperimentData(
            input_data=tree_to_dict_list(samples_pool[:num_samples], name='x'),
            domain=data.domain,
            project_dir=data.project_dir)


class RandomWeightSampler(Block):
    def __init__(self, model: PyTree, seed: jax.Array,
                 lower_bound: float = 0.0,
                 upper_bound: float = 1.0, max_samples: int = 100):
        self.model = model
        self.seed = jrd.key(seed)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_samples = max_samples
        self.keys = jrd.split(self.seed, self.max_samples)

    def call(self, data: ExperimentData, num_samples: int,
             **kwargs) -> ExperimentData:
        params, static = eqx.partition(self.model, eqx.is_inexact_array)
        sample_list = []
        for key, _ in zip(self.keys, range(num_samples)):
            sample = jt.map(lambda x: jrd.uniform(
                key, shape=x.shape, minval=self.lower_bound,
                maxval=self.upper_bound), params)

            sample_list.append({'x': eqx.combine(sample, static)})

        return ExperimentData(domain=data.domain, input_data=sample_list,
                              project_dir=data.project_dir)


class NormalWeightSampler(Block):
    def __init__(self, model: PyTree, seed: jax.Array,
                 max_samples: int = 100):
        self.model = model
        self.seed = jrd.key(seed)
        self.max_samples = max_samples
        self.keys = jrd.split(self.seed, self.max_samples)

    def call(self, data: ExperimentData, num_samples: int,
             **kwargs) -> ExperimentData:
        params, static = eqx.partition(self.model, eqx.is_inexact_array)
        sample_list = []
        for key, _ in zip(self.keys, range(num_samples)):
            sample = jt.map(lambda x: jrd.normal(
                key, shape=x.shape), params)

            sample_list.append({'x': eqx.combine(sample, static)})

        return ExperimentData(domain=data.domain, input_data=sample_list,
                              project_dir=data.project_dir)

#                                                                Evaluate Block
# =============================================================================


class EvaluateModel(Block):
    def __init__(self, loss_fn: Callable,
                 iter_data: Iterator):
        self.iter_data = iter_data
        self.loss_fn = jax.jit(loss_fn)

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        open_es = [es for _, es in data if es.is_status('open')]
        if not open_es:
            return data

        for es in open_es:
            es.mark('in_progress')
            y = self.loss_fn(es.input_data['x'], **next(self.iter_data))
            es.store(name='y', object=y)
            es.mark('finished')

        return data

# =============================================================================


def run_optimization_schedule_multiple_realizations(
        task: Task,
        optimization_schedule: OptimizationSchedule,
        seed: int,
        n_realizations: int = 1,
        include_input: bool = False,
        bounded: Optional[Tuple[float, float]] = None,
) -> xr.DataArray:
    results_list: List[xr.DataArray] = []

    for s in range(n_realizations):
        if bounded is None:
            sampler = NormalWeightSampler(
                model=task.model,
                seed=seed + s,
            )
        else:
            sampler = RandomWeightSampler(
                model=task.model,
                seed=seed + s, lower_bound=bounded[0], upper_bound=bounded[1])

        data = run_optimization_schedule(
            loss_fn=task.loss_fn,
            optimization_schedule=optimization_schedule,
            seed=seed + s,
            bounded=bounded,
            sampler=sampler,
            learning_data=task.learning_data,
        )

        results_list.append(experimentdata_to_xarray(
            data, include_input=include_input))

    # results = xr.concat(results_list, xr.DataArray(range(n_realizations),
    #                                                dims='realization'))

    results = xr.concat(results_list, dim='realization')

    return results


def run_optimization_schedule(
        loss_fn: Callable,
        optimization_schedule: OptimizationSchedule,
        seed: int,
        bounded: Optional[Tuple[float, float]],
        sampler: Block,
        learning_data: Iterator,


) -> ExperimentData:

    # Start with empty ExperimentData
    data = None

    for step in optimization_schedule:

        evaluate_model = EvaluateModel(
            loss_fn=loss_fn, iter_data=learning_data)
        update_step = OPT_MAPPING[step.optimizer](
            **step.hyperparameters, seed=seed,
            bounded=bounded)

        data = run_optimization(
            data_generator=evaluate_model,
            update_step=update_step,
            sampler=sampler,
            n_evals=step.n_evals,
            popsize=update_step.popsize,
            x0_strategy=step.x0_strategy,
            data=data,
            learning_data=learning_data,

        )

    return data


def run_optimization(
    data_generator: Block,
    update_step: Block,
    sampler: Block,
    n_evals: int,
    learning_data: Iterator,
    popsize: int,
    x0_strategy: str,
    data: Optional[ExperimentData] = None,
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
    data_generator.arm(data=x0)
    x0 = data_generator.call(data=x0)

    # Arm update step
    update_step.arm(x0)

    n_updates = n_evals // popsize + (n_evals % popsize > 0)

    data += update_step.call(
        data=data, fn=data_generator.loss_fn, n_evals=n_updates,
        data_iter=learning_data)

    # Remove overiterations
    data.remove_rows_bottom(number_of_rows=popsize * n_updates - n_evals)
    logger.debug(f"Optimization finished with {n_evals} evaluations")

    return data


def create_x0(data: ExperimentData, mode: str, n_samples: int,
              sampler: Block) -> ExperimentData:

    if mode == 'new':
        data = sampler.call(data=data, num_samples=n_samples)

    elif mode == 'last':
        data = data[-n_samples:]

    elif mode == 'best':
        data = data.sort(
            criterion=lambda es: es.output_data['y'],
            reverse=True)[-n_samples:].reset_index()

    return data


class RunOptimization(DataGenerator):
    def __init__(self,
                 n_realizations: int,
                 seed: int,
                 bounded: Optional[Tuple[float, float]] = None,
                 include_input: bool = False,
                 clear_jax_cache: bool = True,
                 ):
        self.bounded = bounded
        self.n_realizations = n_realizations
        self.include_input = include_input
        self.clear_jax_cache = clear_jax_cache
        self.rng = onp.random.default_rng(seed=seed)

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        t1 = time()
        es = experiment_sample.input_data
        task: Task = es['task']
        schedule: OptimizationSchedule = es['optimizer schedule']
        seed: int = self.rng.integers(0, 2**30)

        data_xarr = run_optimization_schedule_multiple_realizations(
            task=task,
            optimization_schedule=schedule,
            seed=seed,
            n_realizations=self.n_realizations,
            include_input=self.include_input,
            bounded=self.bounded,
        )

        experiment_sample.store(
            name='loss history',
            object=data_xarr, to_disk=True)

        if self.clear_jax_cache:
            eqx.clear_caches()
            jax.clear_caches()
            gc.collect()

        logger.info((
            f"Finished optimization task {task.name} with {schedule.name} "
            f" (took {time() - t1:.2f} seconds)"
        ))

        return experiment_sample

# =============================================================================


def run_optimization_no_schedule_multiple_realizations(
        task: Task,
        seed: int,
        trained_l2co_model: L2CO_model,
        n_evals_per_switch: int,
        n_switches: int = 1,
        n_realizations: int = 1,
        include_input: bool = False,
        bounded: Optional[Tuple[float, float]] = None,
) -> Tuple[xr.DataArray, List[OptimizationSchedule]]:
    schedule_list: List[OptimizationSchedule] = []
    results_list: List[xr.DataArray] = []

    for s in range(n_realizations):
        if bounded is None:
            sampler = NormalWeightSampler(
                model=task.model,
                seed=seed + s,
            )
        else:
            sampler = RandomWeightSampler(
                model=task.model,
                seed=seed + s, lower_bound=bounded[0], upper_bound=bounded[1])

        data, schedule = run_optimization_no_schedule(
            trained_l2co_model=trained_l2co_model,
            seed=seed + s,
            bounded=bounded,
            sampler=sampler,
            task=task,
            n_switches=n_switches,
            n_evals_per_switch=n_evals_per_switch,
        )

        results_list.append(experimentdata_to_xarray(
            data, include_input=include_input))

        schedule_list.append(schedule)

    results = xr.concat(results_list, xr.DataArray(range(n_realizations),
                                                   dims='realization'))

    return results, schedule_list


def run_optimization_no_schedule(
        trained_l2co_model: L2CO_model,
        seed: int,
        bounded: Optional[Tuple[float, float]],
        sampler: Block,
        task: Task,
        n_switches: int,
        n_evals_per_switch: int,
) -> Tuple[ExperimentData, OptimizationSchedule]:

    # Start with empty ExperimentData
    data = None
    step_list = []
    # Create first optimization dynamics
    optimization_dynamics = OptimizationDynamics.from_task_and_run(
        task=task
    )

    # Predict the first optimizer
    step: OptimizationStep = trained_l2co_model.predict(
        data=optimization_dynamics)[0]

    logger.debug(
        f"[window 0/{n_switches}] Predicted optimizer: {step.optimizer}")

    step.n_evals = n_evals_per_switch
    previous_step = None
    step_list.append(step)

    for s in range(n_switches):
        evaluate_model = EvaluateModel(
            loss_fn=task.loss_fn, iter_data=task.learning_data)

        if previous_step is None or \
                previous_step.optimizer != step.optimizer:
            update_step = OPT_MAPPING[step.optimizer](
                **step.hyperparameters, seed=seed,
                bounded=bounded)

        data = run_optimization(
            data_generator=evaluate_model,
            update_step=update_step,
            sampler=sampler,
            n_evals=step.n_evals,
            popsize=update_step.popsize,
            x0_strategy=step.x0_strategy,
            data=data,
            learning_data=task.learning_data,

        )

        if s == n_switches - 1:
            break

        previous_step = step.copy()

        optimization_dynamics = OptimizationDynamics.from_task_and_run(
            task=task,
            previous_step=previous_step,
            data=data[-n_evals_per_switch:],
        )

        step: OptimizationStep = trained_l2co_model.predict(
            data=optimization_dynamics)[0]

        logger.debug(
            f"[window {s}/{n_switches}] Predicted optimizer: {step.optimizer}")

        step.n_evals = n_evals_per_switch
        step_list.append(step)

    schedule = OptimizationSchedule(
        steps=step_list, n_evals=n_evals_per_switch*n_switches)

    logger.debug(f"Optimization finished with {n_evals_per_switch*n_switches}")

    return data, schedule


class RunOptimizationL2CO(DataGenerator):
    def __init__(self,
                 trained_l2co_model: L2CO_model,
                 n_realizations: int,
                 seed: int,
                 n_evals_per_switch: int,
                 n_switches: int,
                 bounded: Optional[Tuple[float, float]] = None,
                 include_input: bool = False,
                 clear_jax_cache: bool = True,

                 ):
        self.bounded = bounded
        self.n_realizations = n_realizations
        self.include_input = include_input
        self.clear_jax_cache = clear_jax_cache
        self.rng = onp.random.default_rng(seed=seed)
        self.n_evals_per_switch = n_evals_per_switch
        self.n_switches = n_switches
        self.trained_l2co_model = trained_l2co_model

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        t1 = time()
        es = experiment_sample.input_data
        task: Task = es['task']
        # schedule: OptimizationSchedule = es['optimizer schedule']
        seed: int = self.rng.integers(0, 2**30)

        data_xarr, schedule_list = \
            run_optimization_no_schedule_multiple_realizations(
                task=task,
                seed=seed,
                trained_l2co_model=self.trained_l2co_model,
                n_evals_per_switch=self.n_evals_per_switch,
                n_switches=self.n_switches,
                n_realizations=self.n_realizations,
                include_input=self.include_input,
                bounded=self.bounded,
            )

        experiment_sample.store(
            name='loss history',
            object=data_xarr, to_disk=True)

        experiment_sample.store(
            name='l2co schedules',
            object=schedule_list,
            to_disk=True,
            store_function=save_schedules,
            load_function=load_schedules,
        )

        if self.clear_jax_cache:
            eqx.clear_caches()
            jax.clear_caches()
            gc.collect()

        logger.info((
            f"Finished optimization task {task.name} with L2CO"
            f" (took {time() - t1:.2f} seconds)"
        ))

        return experiment_sample


def save_schedules(object: List[OptimizationSchedule], path: str):
    _path = Path(path)
    with open(_path.with_suffix('.json'), "w") as f:
        json.dump([s.to_dict() for s in object], f)

    return _path.with_suffix('.json')


def load_schedules(filepath: str) -> List[OptimizationSchedule]:
    with open(filepath, "r") as f:
        schedules = json.load(f)

    return [OptimizationSchedule.from_dict(s) for s in schedules]
