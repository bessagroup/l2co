# Standard
import gc
import json
from pathlib import Path
from time import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type

# Third-party
import equinox as eqx
import jax
import numpy as onp
import xarray as xr
from f3dasm import Block, ExperimentData, ExperimentSample
from f3dasm.datageneration import DataGenerator
from f3dasm.design import Domain
from jaxtyping import PyTree

# Local
from ._logging import logger
from .batching import NoneIterator, batch_iter
from .data_loader import OptimizationDynamics
from .l2co_model import L2CO_model
from .optimizers import OPT_MAPPING, UpdateStep
from .optimizerschedule import (OptimizationSchedule, OptimizationStep,
                                step_n_iterations, step_n_iterations_equal)
from .sampler import NormalWeightSampler, RandomWeightSampler
from .task import Task
from .tree_utils import experimentdata_to_xarray, schedule_to_xarray
from .validation import validation_fn

#                                                                Evaluate Block
# =============================================================================


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

    if sampler is None:
        if bounded is None:
            sampler = NormalWeightSampler()
        else:
            sampler = RandomWeightSampler(
                lower_bound=bounded[0],
                upper_bound=bounded[1])

    for s in range(n_realizations):

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
    previous_step = None
    optimizer_dict: Dict[OptimizationStep, UpdateStep] = {}

    for i, (step, step_iterations) in enumerate(zip(
        optimization_schedule,
        step_n_iterations(
            n_iterations=n_iterations,
            budgets=optimization_schedule.budgets))):

        learning_data = batch_iter(
            dataset=loaded_dataset, n_evals=step_iterations,
            batch_size=task.batch_size, key=seed)

        if step in optimizer_dict:
            update_step = optimizer_dict[step]
        else:
            update_step = OPT_MAPPING[step.optimizer](
                **step.hyperparameters, seed=seed,
                bounded=bounded,
                include_input=include_input,
                task=task,
                verbose=verbose)

            optimizer_dict[step] = update_step

        # Check if we have to use scan or not
        if use_scan is None:
            _use_scan = True if isinstance(
                learning_data, NoneIterator) else False
        else:
            _use_scan = use_scan

        data = run_optimization(
            task=task,
            update_step=update_step,
            previous_step=previous_step,
            sampler=sampler,
            n_iterations=step_iterations,
            learning_data=learning_data,
            use_scan=_use_scan,
            seed=seed + i,
            data=data,

        )

        popsizes.append(update_step.popsize)
        previous_step = update_step

    return data, popsizes


def run_optimization(
    task: Task,
    update_step: UpdateStep,
    previous_step: Optional[UpdateStep],
    sampler: Block,
    n_iterations: int,
    learning_data: Iterator | NoneIterator,
    use_scan: bool,
    seed: int,
    data: Optional[ExperimentData] = None,
) -> ExperimentData:

    # If no data is given, create data
    if data is None:
        domain = Domain()
        domain.add_parameter(name='x')
        domain.add_output(name='y')
        data = ExperimentData(domain=domain)

    # Arm update step
    update_step.arm(data=data, previous_step=previous_step, sampler=sampler,
                    seed=seed)

    iterates = update_step.call(
        data=data, n_iterations=n_iterations,
        data_iter=learning_data,
        use_scan=use_scan)

    # If the optimizer created new x0 samples, include them but remove the
    # last iterate to keep it even
    if update_step.has_created_flag:
        data += iterates[:-update_step.popsize]
    # If not, remove the x0 samples
    else:
        data += iterates[update_step.popsize:]

    logger.debug(f"Optimization finished with {n_iterations} evaluations")

    return data


def create_x0(data: Optional[ExperimentData], model: PyTree,
              mode: str, n_samples: int, sampler: Block,
              seed: int) -> ExperimentData:
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
        self.seed = seed
        self.verbose = verbose
        self.sampler_cls = sampler_cls
        self.use_scan = use_scan
        self.track_params = track_params

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        es = experiment_sample.input_data
        task: Task = es['task']
        schedule: OptimizationSchedule = es['optimizer schedule']

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
                seed=self.seed,
                n_switches=n_switches,
            )

        else:
            experiment_sample = self.run_optimization(
                experiment_sample=experiment_sample,
                task=task,
                schedule=schedule,
            )

        if self.clear_jax_cache:
            eqx.clear_caches()
            jax.clear_caches()
            gc.collect()

        return experiment_sample

    def run_optimization(
            self, experiment_sample: ExperimentSample, task: Task,
            schedule: OptimizationSchedule) -> ExperimentSample:
        t1 = time()
        data_xarr = run_optimization_schedule_multiple_realizations(
            task=task,
            optimization_schedule=schedule,
            budget_fn=self.budget_fn,
            seed=self.seed,
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
            task: Task, trained_l2co_model: L2CO_model,
            n_switches: int) -> ExperimentSample:
        t1 = time()
        data_xarr, schedule_xarr = \
            run_optimization_no_schedule_multiple_realizations(
                task=task,
                budget_fn=self.budget_fn,
                partition_factor=self.partition_factor,
                seed=self.seed,
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
