# Standard
import hashlib
from typing import Iterable, Optional, Tuple

# Third-party
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from f3dasm import Block, ExperimentData
from f3dasm.design import Domain

# Local
from .batching import batch_iter
from .data_loader import DataLoader
from .naming import (BUDGET, OPT_HISTORY, OUTPUT_DIM, OUTPUT_MEAN, OUTPUT_MIN,
                     PROBLEM, STAIRCASE, TRAJECTORY, UPDATE_STEP)
from .optimizers import OPT_MAPPING, EvaluateModel, UpdateStep
from .optimizerschedule import OptimizationStep, step_n_iterations_equal
from .performance import output_cummin, scale_by_problem
from .run_optimization import run_optimization
from .sampler import SAMPLER_HASH
from .task import Task
from .tree_utils import experimentdata_to_xarray
from .validation import validation_fn

# =============================================================================


class BenchEnv(gym.Env):
    def __init__(
        self,
            optimizers: Iterable[OptimizationStep],
            task: Task,
            seed: int,
            sampler: Block,
            n_iterations: int,
            n_choices: int,
            bounded: Optional[Tuple[float, float]],
            *, verbose: bool = False,
    ):

        # Seed
        self._seed = seed

        # Iterations
        self._n_choices = n_choices
        self._n_iterations = n_iterations
        self._n_iterations_per_choice = step_n_iterations_equal(
            n_iterations=n_iterations,
            n_switches=n_choices)
        self._i = 0
        self._current_iterations = 0

        # Optimizers
        self._optimizers = optimizers
        self._previous_step = None
        self._bounded = bounded
        self._verbose = verbose

        # Optimization data
        self._data = None
        self._popsizes = []
        self._budgets = None

        # Sampler
        self._sampler = sampler

        # Task
        self._task = task

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                'trajectory': gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(n_iterations,),
                    dtype=np.float32),
                'staircase': gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(n_iterations,),
                    dtype=np.float32),

            }
        )

        # Action space
        self.action_space = gym.spaces.Discrete(len(optimizers))
        self._action_to_optimizer = self._set_optimizers()

    def _set_optimizers(self):
        return {
            idx: OPT_MAPPING[step.optimizer](
                **step.hyperparameters,
                seed=self._seed,
                bounded=self._bounded,
                include_input=True,
                opt_hash=step.hash,
                task=self._task,
                verbose=self._verbose,)
            for idx, step in enumerate(self._optimizers)}

    @property
    def _observation(self):

        if self._data is None:
            return {
                TRAJECTORY: np.ones((self._n_iterations,)),
                STAIRCASE: np.ones((self._n_iterations,)),
            }

        xarr = experimentdata_to_xarray(self._data, popsizes=self._popsizes,
                                        budgets=self._budgets,
                                        n_iterations=self._current_iterations)

        x = validation_fn(x=xarr, budget_steps=self._current_iterations)

        ds = xr.Dataset(
            {TRAJECTORY: scale_by_problem(x[OUTPUT_MEAN]
                                          ).isel({OUTPUT_DIM: 0}),
             'staircase': scale_by_problem(output_cummin(
                 x[OUTPUT_MIN])).ffill(BUDGET).isel({OUTPUT_DIM: 0})
             })

        # Reindex budget dimension
        ds[BUDGET] = range(self._n_iterations - ds[BUDGET].size,
                           self._n_iterations)
        ds = ds.reindex({BUDGET: range(self._n_iterations)}, fill_value=1.0)

        return {
            'trajectory': ds[TRAJECTORY],
            'staircase': ds[STAIRCASE],
        }

    @property
    def _reward(self):
        if self._terminated:
            best_loss = float(
                self._data.sort(lambda x: x.output_data['y']
                                ).reset_index().data[0].output_data['y'])
            return -best_loss
        else:
            return -1e3

    @property
    def _terminated(self):
        return (self._i == self._n_choices)

    @property
    def _truncated(self):
        return False

    @property
    def _info(self):
        return {}

    @property
    def _display_observation(self):
        obs = self._observation
        fig, ax = plt.subplots()
        ax.plot(obs['trajectory'], label='Trajectory', color='k')
        ax.plot(obs['staircase'], label='Staircase',
                linestyle='--', color='red')
        return fig

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        self._data = None
        self._previous_step = None
        self._action_to_optimizer = self._set_optimizers()
        self._i = 0
        self._current_iterations = 0
        return (self._observation, self._info)

    def _run_x0(self):
        # Create the ExperimentData object
        domain = Domain()
        domain.add_parameter(name='x')
        domain.add_output(name='y')
        domain.add_output(name='opt_step')
        self._data = ExperimentData(domain=domain)

        # Sample one initial solution
        self._sampler.arm(data=self._data)
        self._data = self._sampler.call(
            data=self._data, model=self._task.model, num_samples=1,
            seed=self._seed)

        # Evaluate the model to get the initial loss
        data_generator = EvaluateModel(task=self._task, seed=self._seed)
        data_generator.arm(self._data)
        self._data = data_generator.call(data=self._data)

        self._previous_step = 'sampler'

    def _run_optimization(self, update_step: UpdateStep, n_iterations: int):
        if self._data is None:
            self._run_x0()

        learning_data = batch_iter(dataset=self._task.loaded_dataset,
                                   n_evals=n_iterations,
                                   batch_size=self._task.batch_size,
                                   key=self._seed)

        self._data = run_optimization(
            task=self._task,
            update_step=update_step,
            previous_step=self._previous_step,
            sampler=self._sampler,
            n_iterations=n_iterations,
            learning_data=learning_data,
            seed=self._seed + self._i + 1,
            data=self._data,
            use_scan=True
        )

        self._popsizes.append(update_step.popsize)
        self._budgets = [1.0 / (self._i + 1)] * (self._i+1)
        self._current_iterations += n_iterations
        self._previous_step = update_step

    def step(self, action: int):
        # Select update_step from action
        update_step = self._action_to_optimizer[action]

        # Run optimization with the selected update_step
        self._run_optimization(
            update_step=update_step,
            n_iterations=self._n_iterations_per_choice[self._i],
        )

        # Increment the _i counter
        self._i += 1

        # Get the observation, reward, termination, truncation, and info
        return (self._observation,
                self._reward,
                self._terminated,
                self._truncated,
                self._info)


# =============================================================================

def func_hash(x: np.ndarray) -> int:
    return int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)


STANDARD_OPTIMIZER_MAPPING = {
    '1.0': 0,
    SAMPLER_HASH: 1
}


class BenchEnv2(gym.Env):
    def __init__(self, d: DataLoader,
                 optimizers: Iterable[OptimizationStep],
                 window_size: float):
        self._d = d
        self._optimizers = optimizers
        self._window_size = window_size
        self._n_choices = np.arange(
            window_size, 1.0 + window_size, window_size)
        self._action_to_optimizer = {
            idx: opt.hash for idx, opt in enumerate(optimizers)}

        self._params = d.to_all_params(window_size=window_size,
                                       include_last=True)

        standard_mapping = STANDARD_OPTIMIZER_MAPPING.copy()
        standard_mapping.update(
            {v: 2+i for i, v in
             enumerate(self._action_to_optimizer.values())})

        for n in self._n_choices:
            self._params[f"{n:.2f}"][UPDATE_STEP] = xr.apply_ufunc(
                lambda x: standard_mapping.get(x, -1),
                self._params[f"{n:.2f}"][UPDATE_STEP],
                vectorize=True,
                output_dtypes=[int],
            )

        self.action_space = gym.spaces.Discrete(len(optimizers))

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                'trajectory': gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._d.data[BUDGET].size,),
                    dtype=np.float32),
                'staircase': gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._d.data[BUDGET].size,),
                    dtype=np.float32),
                'update_step': gym.spaces.Box(
                    low=0, high=len(optimizers) + 1,
                    shape=(self._d.data[BUDGET].size,),
                    dtype=np.int32),
            }
        )

        performance = self._d.performance_per_realization.isel(
            {BUDGET: -1}, drop=True)
        opt_history = self._d.select_t(
            t_start=0.0, t=1.0, select_one_before=True).hash_opt_step

        self._performance = performance.assign_coords(
            {OPT_HISTORY: opt_history}
        ).groupby((OPT_HISTORY, PROBLEM, OUTPUT_DIM),
                  restore_coord_dims=True).first().isel({OUTPUT_DIM: 0})

    @property
    def _observation(self):

        if self._current_step == 0.0:
            return {
                'trajectory': np.ones(self._d.data[BUDGET].size),
                'staircase': np.ones(self._d.data[BUDGET].size),
                'update_step': np.zeros(self._d.data[BUDGET].size, dtype=int),
            }

        history_hash = func_hash(np.array(self._current_history))

        x = self._params[f"{self._current_step:.2f}"].isel(
            {PROBLEM: self._problem_id, OUTPUT_DIM: 0}).sel(
                {OPT_HISTORY: history_hash})

        return {
            'trajectory': x[TRAJECTORY],
            'staircase': x[STAIRCASE],
            'update_step': x[UPDATE_STEP],
        }

    @property
    def _reward(self):
        return 0 if not self._terminated else self._reward_function()

    @property
    def _terminated(self) -> bool:
        return self._current_step == 1.0

    @property
    def _truncated(self) -> bool:
        return False

    @property
    def _info(self) -> dict:
        return {}

    def _reward_function(self) -> float:
        history_hash = func_hash(np.array(self._current_history))
        return 1.0 - float(
            self._performance.isel({PROBLEM: self._problem_id}).sel(
                {OPT_HISTORY: history_hash}))

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_step = 0.0
        self._current_history = [SAMPLER_HASH]
        self._problem_id = np.random.default_rng(
            seed).integers(0, self._d.data[PROBLEM].size)
        return self._observation, self._info

    def step(self, action: int):
        self._current_history.extend(
            [self._action_to_optimizer[action]] * int(
                self._window_size * self._d.data[BUDGET].size))
        self._current_step += self._window_size

        # Get the observation, reward, termination, truncation, and info
        return (self._observation,
                self._reward,
                self._terminated,
                self._truncated,
                self._info)
