# Standard
import gc
import json
from functools import partial
from pathlib import Path
from time import time
from typing import Callable, Iterator, List, Optional, Tuple, Type

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as onp
import xarray as xr
from f3dasm import Block, ExperimentData, ExperimentSample
from f3dasm.datageneration import DataGenerator
from f3dasm.design import Domain
from jaxtyping import PyTree

# Local
from ._logging import logger
from .batching import NoneIterator, batch_iter, unroll_batches
from .data_loader import OptimizationDynamics
from .l2co_model import L2CO_model
from .optimizers import OPT_MAPPING
from .optimizerschedule import (OptimizationSchedule, OptimizationStep,
                                step_n_iterations, step_n_iterations_equal)
from .sampler import NormalWeightSampler, RandomWeightSampler
from .task import Task
from .tree_utils import dict_list_to_tree, tree_to_dict_list
from .validation import validation_fn

# =============================================================================


def experimentdata_to_xarray(data: ExperimentData,
                             popsizes: List[int],
                             budgets: List[float] | float,
                             n_iterations: int,
                             include_input: bool = False) -> xr.Dataset:
    y_arr = jnp.vstack([es.output_data['y']
                        for _, es in data]).T

    y = xr.DataArray(
        y_arr,
        dims=['output_dim', 'budget'],
        coords={'output_dim': ['y']})

    if include_input:
        x_arr = jnp.vstack([es.input_data['x']
                            for _, es in data]).T

        x = xr.DataArray(
            x_arr,
            dims=['input_dim', 'budget'],
            coords={'input_dim': [f"x{i}" for i in range(x_arr.shape[0])]})

        ds = xr.Dataset({'input': x,
                         'output': y,
                         })

    else:
        ds = xr.Dataset({
            'output': y,
        })

    ds.attrs['popsizes'] = popsizes
    ds.attrs['budgets'] = budgets
    ds.attrs['n_iterations'] = n_iterations
    return ds


def schedule_to_xarray(data: OptimizationSchedule) -> xr.DataArray:
    x = xr.DataArray(
        [json.dumps(data.to_dict())],
        dims=['schedule']
    )
    return x

#                                                               Sampling Blocks
# =============================================================================


#                                                                Evaluate Block
# =============================================================================

class EvaluateModel(Block):
    def __init__(self, task: Task, seed: int = 0):
        self.task = task
        self.seed = seed

    def call(self, data: ExperimentData) -> ExperimentData:
        x = [v.input_data for k, v in data]
        models = dict_list_to_tree(x, name='x')

        inner_steps = len(data)

        if self.task.pass_rng:
            rng_sample = {'key': jrd.split(
                jrd.key(self.seed), num=inner_steps)}
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

        return ExperimentData(domain=data.domain, input_data=x,
                              output_data=tree_to_dict_list(y, name='y'),
                              project_dir=data.project_dir)


# =============================================================================


def run_optimization_schedule_multiple_realizations(
        task: Task,
        optimization_schedule: OptimizationSchedule,
        budget_fn: Callable[[], int],
        seed: int,
        partition_factor: float,
        n_realizations: int = 1,
        include_input: bool = False,
        bounded: Optional[Tuple[float, float]] = None,
        sampler: Optional[Block] = None,
        verbose: bool = False,
        use_scan: Optional[bool] = None,
        track_params: bool = False,
) -> xr.Dataset:
    results_list: List[xr.DataArray] = []

    n_iterations: int = int(budget_fn(task=task) * partition_factor)

    for s in range(n_realizations):
        if sampler is None:
            if bounded is None:
                sampler = NormalWeightSampler()
            else:
                sampler = RandomWeightSampler(
                    lower_bound=bounded[0],
                    upper_bound=bounded[1])

        data, popsizes = run_optimization_schedule(
            task=task,
            optimization_schedule=optimization_schedule,
            n_iterations=n_iterations,
            seed=seed + s,
            bounded=bounded,
            sampler=sampler,
            include_input=track_params,
            verbose=verbose,
            use_scan=use_scan,
        )

        results_list.append(experimentdata_to_xarray(
            data=data,
            popsizes=popsizes,
            budgets=optimization_schedule.budgets,
            n_iterations=n_iterations,
            include_input=include_input))

    results = xr.concat(results_list, dim='realization')

    return results


def run_optimization_schedule(
        task: Task,
        optimization_schedule: OptimizationSchedule,
        n_iterations: int,
        seed: int,
        bounded: Optional[Tuple[float, float]],
        sampler: Block,
        include_input: bool,
        verbose: bool,
        use_scan: Optional[bool],
) -> Tuple[ExperimentData, List[int]]:

    # Start with empty ExperimentData
    data = None
    loaded_dataset = task.loaded_dataset
    popsizes = []

    for step, step_iterations in zip(
        optimization_schedule,
        step_n_iterations(
            n_iterations=n_iterations,
            budgets=optimization_schedule.budgets)):

        learning_data = batch_iter(
            dataset=loaded_dataset, n_evals=step_iterations,
            batch_size=task.batch_size, key=seed)

        if use_scan is None:
            _use_scan = True if isinstance(
                learning_data, NoneIterator) else False
        else:
            _use_scan = use_scan

        update_step = OPT_MAPPING[step.optimizer](
            **step.hyperparameters, seed=seed,
            bounded=bounded,
            include_input=include_input,
            pass_rng=task.pass_rng,
            task=task,
            verbose=verbose)

        data = run_optimization(
            task=task,
            update_step=update_step,
            sampler=sampler,
            n_iterations=step_iterations,
            learning_data=learning_data,
            popsize=update_step.popsize,
            x0_strategy=step.x0_strategy,
            seed=seed,
            use_scan=_use_scan,
            data=data,

        )

        popsizes.append(update_step.popsize)

    return data, popsizes


def run_optimization(
    task: Task,
    update_step: Block,
    sampler: Block,
    n_iterations: int,
    learning_data: Iterator,
    popsize: int,
    x0_strategy: str,
    seed: int,
    use_scan: bool,
    data: Optional[ExperimentData] = None,
) -> ExperimentData:

    if data is None:
        # Create data
        domain = Domain()
        domain.add_parameter(name='x')
        domain.add_output(name='y')
        data = ExperimentData(domain=domain)

    # Create x0
    x0 = create_x0(data=data, model=task.model, mode=x0_strategy,
                   n_samples=popsize,
                   sampler=sampler, seed=seed)

    if x0_strategy in ['new', 'best_one']:
        data_generator = EvaluateModel(task=task, seed=seed)
        data_generator.arm(x0)
        x0 = data_generator.call(data=x0)
        n_iterations -= 1
        data += x0

    # Arm update step
    update_step.arm(data=x0, fn=task.loss_fn, has_aux=task.has_aux)

    iterates = update_step.call(
        data=data, n_iterations=n_iterations,
        data_iter=learning_data,
        use_scan=use_scan)

    data += iterates[popsize:]

    logger.debug(f"Optimization finished with {n_iterations} evaluations")

    return data


def create_x0(data: ExperimentData, model: PyTree, mode: str, n_samples: int,
              sampler: Block, seed: int) -> ExperimentData:

    if mode == 'new':
        _data = sampler.call(data=data, model=model,
                             num_samples=n_samples, seed=seed)

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
        _data += create_x0(
            data=data, model=model, mode='new',
            n_samples=n_samples - len(_data),
            sampler=sampler, seed=seed)

    return _data


class RunOptimization(DataGenerator):
    def __init__(self,
                 budget_fn: Callable[[], int],
                 n_realizations: int,
                 partition_factor: float,
                 seed: int,
                 bounded: Optional[Tuple[float, float]] = None,
                 include_input: bool = False,
                 clear_jax_cache: bool = True,
                 sampler_cls: Optional[Type[Block]] = None,
                 verbose: bool = False,
                 use_scan: Optional[bool] = None,
                 track_params: bool = False
                 ):
        self.budget_fn = budget_fn
        self.bounded = bounded
        self.n_realizations = n_realizations
        self.partition_factor = partition_factor
        self.include_input = include_input
        self.clear_jax_cache = clear_jax_cache
        self.rng = onp.random.default_rng(seed=seed)
        self.verbose = verbose
        self.sampler_cls = sampler_cls
        self.use_scan = use_scan
        self.track_params = track_params

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        es = experiment_sample.input_data
        task: Task = es['task']
        schedule: OptimizationSchedule = es['optimizer schedule']
        seed: int = self.rng.integers(0, 2**30)

        if schedule.name.startswith('l2co'):
            n_switches = schedule.steps[0].hyperparameters['n_switches']
            l2co_path = Path(schedule.steps[0].
                             hyperparameters['model_dir']) / str(
                schedule.steps[0].hyperparameters['l2co_path']) / 'model'

            trained_l2co_model = L2CO_model.load(l2co_path)

            experiment_sample = self.run_optimization_l2co(
                experiment_sample=experiment_sample,
                task=task,
                trained_l2co_model=trained_l2co_model,
                seed=seed,
                n_switches=n_switches,
            )

        else:
            experiment_sample = self.run_optimization(
                experiment_sample=experiment_sample,
                task=task,
                schedule=schedule,
                seed=seed,
            )

        if self.clear_jax_cache:
            eqx.clear_caches()
            jax.clear_caches()
            gc.collect()

        return experiment_sample

    def run_optimization(
        self, experiment_sample: ExperimentSample, task: Task,
        schedule: OptimizationSchedule,
            seed: int) -> ExperimentSample:
        t1 = time()
        data_xarr = run_optimization_schedule_multiple_realizations(
            task=task,
            optimization_schedule=schedule,
            budget_fn=self.budget_fn,
            seed=seed,
            partition_factor=self.partition_factor,
            n_realizations=self.n_realizations,
            include_input=self.include_input,
            bounded=self.bounded,
            sampler=self.sampler_cls,
            verbose=self.verbose,
            use_scan=self.use_scan,
            track_params=self.track_params
        )

        experiment_sample.store(
            name='loss history',
            object=data_xarr, to_disk=True)

        logger.info((
            f"Finished optimization task {task.name} with {schedule.name} with"
            f" {int(self.budget_fn(task=task) * self.partition_factor)} "
            f"iterations for {self.n_realizations} realizations"
            f" (took {time() - t1:.2f} seconds)"
        ))

        return experiment_sample

    def run_optimization_l2co(
        self, experiment_sample: ExperimentSample,
            task: Task, trained_l2co_model: L2CO_model, seed: int,
            n_switches: int) -> ExperimentSample:
        t1 = time()
        data_xarr, schedule_xarr = \
            run_optimization_no_schedule_multiple_realizations(
                task=task,
                budget_fn=self.budget_fn,
                partition_factor=self.partition_factor,
                seed=seed,
                trained_l2co_model=trained_l2co_model,
                n_switches=n_switches,
                n_realizations=self.n_realizations,
                include_input=self.include_input,
                bounded=self.bounded,
                sampler=self.sampler_cls,
                verbose=self.verbose,
                use_scan=self.use_scan,
                track_params=self.track_params,
            )

        experiment_sample.store(
            name='loss history',
            object=data_xarr, to_disk=True)

        experiment_sample.store(
            name='l2co schedules',
            object=schedule_xarr,
            to_disk=True)

        logger.info((
            f"Finished L2CO optimization: task {task.name} with"
            f" {int(self.budget_fn(task=task) * self.partition_factor)} "
            f"iterations for {self.n_realizations} realizations"
            f" (took {time() - t1:.2f} seconds)"
        ))

        return experiment_sample

# =============================================================================


def run_optimization_no_schedule_multiple_realizations(
        task: Task,
        budget_fn: Callable[[], int],
        partition_factor: float,
        seed: int,
        trained_l2co_model: L2CO_model,
        n_switches: int = 1,
        n_realizations: int = 1,
        include_input: bool = False,
        bounded: Optional[Tuple[float, float]] = None,
        sampler: Optional[Block] = None,
        verbose: bool = False,
        use_scan: Optional[bool] = None,
        track_params: bool = False
) -> Tuple[xr.DataArray, List[OptimizationSchedule]]:
    schedules_list: List[xr.DataArray] = []
    results_list: List[xr.DataArray] = []

    n_iterations: int = int(budget_fn(task=task) * partition_factor)

    for s in range(n_realizations):
        if sampler is None:
            if bounded is None:
                sampler = NormalWeightSampler()
            else:
                sampler = RandomWeightSampler(
                    lower_bound=bounded[0],
                    upper_bound=bounded[1])

        data, schedule, popsizes = run_optimization_no_schedule(
            trained_l2co_model=trained_l2co_model,
            seed=seed + s,
            bounded=bounded,
            sampler=sampler,
            task=task,
            n_switches=n_switches,
            n_iterations=n_iterations,
            include_input=track_params,
            verbose=verbose,
            use_scan=use_scan,
        )

        ds = experimentdata_to_xarray(
            data=data,
            popsizes=popsizes,
            budgets=schedule.budgets,
            n_iterations=n_iterations,
            include_input=include_input)

        results_list.append(validation_fn(ds))

        schedules_list.append(schedule_to_xarray(
            data=schedule))

    results = xr.concat(results_list, dim='realization')
    schedules = xr.concat(schedules_list, dim='realization')

    return results, schedules


def run_optimization_no_schedule(
        task: Task,
        trained_l2co_model: L2CO_model,
        n_switches: int,
        n_iterations: int,
        seed: int,
        bounded: Optional[Tuple[float, float]],
        sampler: Block,
        include_input: bool,
        verbose: bool,
        use_scan: Optional[bool],
) -> Tuple[ExperimentData, OptimizationSchedule, List[int]]:

    budgets = step_n_iterations_equal(
        n_iterations=n_iterations, n_switches=n_switches)

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

    previous_step = None
    step_list.append(step)
    loaded_dataset = task.loaded_dataset
    popsizes = []

    for s, n_iterations_per_switch in enumerate(budgets):

        learning_data = batch_iter(
            dataset=loaded_dataset, n_evals=n_iterations_per_switch,
            batch_size=task.batch_size, key=seed)

        if use_scan is None:
            _use_scan = True if isinstance(
                learning_data, NoneIterator) else False
        else:
            _use_scan = use_scan

        if previous_step is None or previous_step.optimizer != step.optimizer:
            update_step = OPT_MAPPING[step.optimizer](
                **step.hyperparameters, seed=seed,
                bounded=bounded,
                include_input=include_input,
                pass_rng=task.pass_rng,
                task=task,
                verbose=verbose)

        data = run_optimization(
            task=task,
            update_step=update_step,
            sampler=sampler,
            n_iterations=n_iterations_per_switch,
            learning_data=learning_data,
            popsize=update_step.popsize,
            x0_strategy=step.x0_strategy,
            seed=seed,
            use_scan=_use_scan,
            data=data,
        )

        # Append population size
        popsizes.append(update_step.popsize)

        if s == n_switches - 1:
            break

        previous_step = step.copy()

        optimization_dynamics = OptimizationDynamics.from_task_and_run(
            task=task,
            previous_step=previous_step,
            data=data[-(n_iterations_per_switch*update_step.popsize):],
        )

        step: OptimizationStep = trained_l2co_model.predict(
            data=optimization_dynamics)[0]

        logger.debug(
            f"[window {s}/{n_switches}] Predicted optimizer: {step.optimizer}")

        step_list.append(step)

    schedule = OptimizationSchedule(
        steps=step_list, budgets=[1/n_switches]*n_switches)

    logger.debug(
        f"Optimization finished with {n_iterations_per_switch*n_switches}"
        f"iterations!")

    return data, schedule, popsizes


def run_optimization_random_schedule_multiple_realizations(
        task: Task,
        budget_fn: Callable[[], int],
        partition_factor: float,
        seed: int,
        trained_l2co_model: L2CO_model,
        n_switches: int = 1,
        n_realizations: int = 1,
        include_input: bool = False,
        bounded: Optional[Tuple[float, float]] = None,
        sampler: Optional[Block] = None,
        verbose: bool = False,
        use_scan: Optional[bool] = None,
        track_params: bool = False
) -> Tuple[xr.DataArray, List[OptimizationSchedule]]:
    schedule_list: List[OptimizationSchedule] = []
    results_list: List[xr.DataArray] = []

    n_iterations: int = int(budget_fn(task=task) * partition_factor)

    for s in range(n_realizations):
        if sampler is None:
            if bounded is None:
                sampler = NormalWeightSampler()
            else:
                sampler = RandomWeightSampler(
                    lower_bound=bounded[0],
                    upper_bound=bounded[1])

        data, schedule, popsizes = run_optimization_random_schedule(
            trained_l2co_model=trained_l2co_model,
            seed=seed + s,
            bounded=bounded,
            sampler=sampler,
            task=task,
            n_switches=n_switches,
            n_iterations=n_iterations,
            include_input=track_params,
            verbose=verbose,
            use_scan=use_scan,
        )

        ds = experimentdata_to_xarray(
            data=data,
            popsizes=popsizes,
            budgets=schedule.budgets,
            n_iterations=n_iterations,
            include_input=include_input)

        results_list.append(validation_fn(ds))

        schedule_list.append(schedule)

    results = xr.concat(results_list, dim='realization')

    return results, schedule_list


def run_optimization_random_schedule(
        task: Task,
        trained_l2co_model: L2CO_model,
        n_switches: int,
        n_iterations: int,
        seed: int,
        bounded: Optional[Tuple[float, float]],
        sampler: Block,
        include_input: bool,
        verbose: bool,
        use_scan: Optional[bool],
) -> Tuple[ExperimentData, OptimizationSchedule, List[int]]:

    gen = onp.random.default_rng(seed)

    budgets = step_n_iterations_equal(
        n_iterations=n_iterations, n_switches=n_switches)

    # Start with empty ExperimentData
    data = None

    step_list = []

    # Predict the first optimizer
    step: OptimizationStep = gen.choice(
        list(trained_l2co_model.stage0_optimizers.items()))

    logger.debug(
        f"[window 0/{n_switches}] Predicted optimizer: {step.optimizer}")

    previous_step = None
    step_list.append(step)
    loaded_dataset = task.loaded_dataset
    popsizes = []

    for s, n_iterations_per_switch in enumerate(budgets):

        learning_data = batch_iter(
            dataset=loaded_dataset, n_evals=n_iterations_per_switch,
            batch_size=task.batch_size, key=seed)

        if use_scan is None:
            _use_scan = True if isinstance(
                learning_data, NoneIterator) else False
        else:
            _use_scan = use_scan

        if previous_step is None or previous_step.optimizer != step.optimizer:
            update_step = OPT_MAPPING[step.optimizer](
                **step.hyperparameters, seed=seed,
                bounded=bounded,
                include_input=include_input,
                pass_rng=task.pass_rng,
                task=task,
                verbose=verbose)

        data = run_optimization(
            task=task,
            update_step=update_step,
            sampler=sampler,
            n_iterations=n_iterations_per_switch,
            learning_data=learning_data,
            popsize=update_step.popsize,
            x0_strategy=step.x0_strategy,
            seed=seed,
            use_scan=_use_scan,
            data=data,
        )

        # Append population size
        popsizes.append(update_step.popsize)

        if s == n_switches - 1:
            break

        previous_step = step.copy()

        step: OptimizationStep = gen.choice(
            list(
                trained_l2co_model.stage1_models_optimizers[
                    previous_step.name].values()))

        logger.debug(
            f"[window {s}/{n_switches}] Predicted optimizer: {step.optimizer}")

        step_list.append(step)

    schedule = OptimizationSchedule(
        steps=step_list, budgets=[1/n_switches]*n_switches)

    logger.debug(
        f"Optimization finished with {n_iterations_per_switch*n_switches}"
        f"iterations! (random choices)")

    return data, schedule, popsizes


def save_schedules(object: List[OptimizationSchedule], path: str):
    _path = Path(path)
    with open(_path.with_suffix('.json'), "w") as f:
        json.dump([s.to_dict() for s in object], f)

    return _path.with_suffix('.json')


def load_schedules(filepath: str) -> List[OptimizationSchedule]:
    with open(filepath, "r") as f:
        schedules = json.load(f)

    return [OptimizationSchedule.from_dict(s) for s in schedules]
