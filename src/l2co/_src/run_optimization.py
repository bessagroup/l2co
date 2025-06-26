"""
Module for running optimization tasks using a JAX-based environment
and policy model.
"""

#                                                                       Modules
# =============================================================================

# Standard
import gc
from functools import partial
from time import time
from typing import Callable, Optional, Tuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from jaxtyping import PRNGKeyArray

# Local
from ._logging import logger
from .environment import Env, EnvParams, EnvState
from .optimizerschedule import OptimizationSchedule
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class RolloutWrapper:
    """
    Wrapper for performing rollouts in the optimization environment.

    Parameters
    ----------
    env : Env
        The optimization environment.
    env_params : EnvParams
        Parameters for the environment.
    policy_model : eqx.Module
        Policy model for decision-making.

    Methods
    -------
    single_rollout(key_input)
        Perform a single rollout in the environment.
    """

    def __init__(self, env: Env, env_params: EnvParams,
                 policy_model: eqx.Module):
        self.env = env
        self.env_params = env_params
        self.policy_model = policy_model

    @partial(jax.jit, static_argnames=("self",))
    def single_rollout(self, key_input: PRNGKeyArray):
        key_reset, key_episode = jr.split(key_input)
        obs, state = self.env.reset(key_reset, self.env_params)

        def policy_step(state_input: Tuple[jax.Array, EnvState, PRNGKeyArray,
                                           jax.Array, jax.Array], _):
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jr.split(key, 3)

            logits = self.policy_model(obs, state.time)
            action = jr.categorical(key_net, logits)
            action = jnp.squeeze(action)
            next_obs, next_state, reward, done, _ = self.env.step(
                key_step, state, action, self.env_params)

            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)

            carry = [
                next_obs,
                next_state,
                key,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                key_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )

        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        last_state = carry_out[1]
        return obs, action, reward, next_obs, done, cum_return, last_state


class RunOptimization(DataGenerator):
    """
    Class for running optimization tasks.

    Parameters
    ----------
    budget_fn : Callable[[], int]
        Function to determine the budget for the task.
    n_realizations : int
        Number of realizations to run.
    seed : int
        Random seed for reproducibility.
    sampler : Callable
        Sampler for generating initial parameters.
    memory_size : int
        Size of the memory buffer.
    iteration_stepsize : int, optional
        Step size for iterations, by default 1.
    bounded : Optional[Tuple[float, float]], optional
        Bounds for the optimization, by default (None, None).
    clear_jax_cache : bool, optional
        Whether to clear the JAX cache after execution, by default True.
    verbose : bool, optional
        Whether to enable verbose logging, by default False.

    Methods
    -------
    execute(experiment_sample)
        Execute the optimization task.
    run_optimization(experiment_sample, task, schedule)
        Run the optimization process.
    """

    def __init__(self,
                 budget_fn: Callable[[], int],
                 n_realizations: int,
                 seed: int,
                 sampler: Callable,
                 memory_size: int,
                 iteration_stepsize: int = 1,
                 bounded: Optional[Tuple[float, float]] = (None, None),
                 clear_jax_cache: bool = True,
                 verbose: bool = False,
                 ):
        self.budget_fn = budget_fn
        self.n_realizations = n_realizations
        self.seed = seed
        self.sampler = sampler
        self.memory_size = memory_size
        self.iteration_stepsize = iteration_stepsize
        self.bounded = bounded
        self.clear_jax_cache = clear_jax_cache
        self.verbose = verbose

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        es = experiment_sample.input_data
        task: Task = es['task']
        schedule: OptimizationSchedule = es['optimizer schedule']

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
            self,
            experiment_sample: ExperimentSample,
            task: Task,
            schedule: OptimizationSchedule) -> ExperimentSample:
        t1 = time()

        total_iterations = self.budget_fn(task=task)
        env = Env()
        env_params = EnvParams.init(
            task=task,
            schedules=schedule.steps,
            sampler=self.sampler,
            n_iterations=self.iteration_stepsize,
            total_iterations=total_iterations,
            memory_size=self.memory_size,
            bounded=self.bounded,
        )

        key = jr.key(self.seed)

        policy_model = schedule.to_policy()
        rollout = RolloutWrapper(env, env_params, policy_model)

        obs, action, reward, next_obs, done, cum_return, last_state = jax.vmap(
            rollout.single_rollout)(jr.split(key, self.n_realizations))

        data_xarr = last_state.history_state.to_xarray_realizations()

        experiment_sample.store(
            name="loss history",
            object=data_xarr, to_disk=True)

        logger.info((
            f"Finished optimization task {task.name} with {schedule.name} with"
            f" {total_iterations} "
            f"iterations for {self.n_realizations} realizations"
            f" (took {time() - t1:.2f} seconds)"
        ))

        return experiment_sample
