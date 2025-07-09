"""
Module for running optimization tasks using a JAX-based environment
and policy model.
"""

#                                                                       Modules
# =============================================================================

import gc
from functools import partial
from pathlib import Path
from time import time
from typing import Any, Callable, Optional, Tuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
# Standard
import optax
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from jaxtyping import PRNGKeyArray, PyTree

# Local
from ._logging import logger
from .environment import Env, EnvParams, EnvState
from .l2co_model import L2CO_model
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

    def __init__(self, env: Env, env_params: EnvParams):
        self.env = env
        self.env_params = env_params

    def batch_rollout(self, model: eqx.Module,
                      n_realizations: int, key: PRNGKeyArray):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        return jax.vmap(self.single_rollout,
                        in_axes=(None, None, 0),
                        out_axes=0)(params, static,
                                    jr.split(key, n_realizations))

    @partial(jax.jit, static_argnames=("self", "static",))
    def single_rollout(self, params: PyTree, static: PyTree,
                       key: PRNGKeyArray):
        policy_model = eqx.combine(params, static)
        key_reset, key_episode = jr.split(key)
        obs, state = self.env.reset(key_reset, self.env_params)

        def policy_step(state_input: Tuple[jax.Array, EnvState, PRNGKeyArray,
                                           jax.Array, jax.Array], _):
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jr.split(key, 3)

            logits = policy_model(obs, state.time)
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
            y = [obs, action, reward, next_obs, logits, done]
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
        obs, action, reward, next_obs, logits, done = scan_out
        cum_return = carry_out[-2]
        last_state = carry_out[1]
        return (obs, action, reward, next_obs, logits, done,
                cum_return, last_state)

    def train(self, model: eqx.Module, optimizer: optax.GradientTransformation,
              n_episodes: int, *, key: PRNGKeyArray):
        params, static = eqx.partition(model, eqx.is_inexact_array)

        scan_fn = partial(policy_train_step,
                          static=static,
                          rollout=self,
                          optimizer=optimizer)

        carry_init = (params, optimizer.init(params), key)

        (carry_final, losses) = jax.lax.scan(
            scan_fn,
            carry_init,
            xs=None,
            length=n_episodes,
        )
        params, opt_state, _ = carry_final

        return eqx.combine(params, static), opt_state, losses


def policy_train_step(
    carry: Tuple[PyTree, optax.OptState, PRNGKeyArray],
    _: Any,
    static: PyTree,
    rollout: RolloutWrapper,
    optimizer: optax.GradientTransformation,
) -> Tuple[Tuple[PyTree, optax.OptState, PRNGKeyArray], jax.Array]:

    params, opt_state, key = carry
    # model = eqx.combine(params, rollout.static)
    key, subkey = jr.split(key)

    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(
        loss_fn)(params, static, rollout, key=subkey)

    # Update model
    updates, opt_state = optimizer.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)

    new_carry = (params, opt_state, key)
    return new_carry, loss


def loss_fn(params: PyTree, static: PyTree, rollout: RolloutWrapper,
            key: PRNGKeyArray) -> jax.Array:
    # One rollout using the current model
    (obs, action, reward, next_obs,
     logits, done, cum_return, _) = rollout.single_rollout(
        params, static, key=key)

    # Compute returns
    returns = compute_returns(reward, gamma=0.99)  # [T]

    # Per-timestep loss using logits and actions
    per_step_loss = jax.vmap(policy_loss)(logits, action, returns)

    # Total loss over trajectory
    return jnp.sum(per_step_loss)


def compute_returns(rewards: jax.Array, gamma: float = 0.99) -> jax.Array:
    def scan_fn(G, r):
        G = r + gamma * G
        return G, G

    reversed_rewards = jnp.flip(rewards)
    _, reversed_returns = jax.lax.scan(scan_fn, init=0.0, xs=reversed_rewards)
    returns = jnp.flip(reversed_returns)
    return returns


def policy_loss(logits: jax.Array, action: int, G: float) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits)
    log_prob = log_probs[action]
    return -log_prob * G


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
                 memory_size: float,
                 iteration_stepsize: float = 0.1,
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

        if schedule.name.startswith('l2co'):
            l2co_path = Path(
                schedule.steps[0].hyperparameters['model_dir']) / str(
                schedule.steps[0].hyperparameters['l2co_path'])

            trained_l2co_model = L2CO_model.load(l2co_path)

            self.run_optimization_l2co(
                experiment_sample=experiment_sample,
                task=task,
                model=trained_l2co_model,
                train=schedule.steps[0].hyperparameters.get('train', False),
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
            self,
            experiment_sample: ExperimentSample,
            task: Task,
            schedule: OptimizationSchedule) -> ExperimentSample:
        t1 = time()

        total_iterations = self.budget_fn(task=task)

        iteration_stepsize = int(self.iteration_stepsize * total_iterations)
        memory_size = int(self.memory_size * total_iterations)

        env = Env()
        env_params = EnvParams.init(
            task=task,
            schedules=schedule.steps,
            sampler=self.sampler,
            n_iterations=iteration_stepsize,
            total_iterations=total_iterations,
            memory_size=memory_size,
            bounded=self.bounded,
        )

        key = jr.key(self.seed)

        policy_model = schedule.to_policy()
        rollout = RolloutWrapper(env, env_params)

        (obs, action, reward, next_obs, logits, done, cum_return,
         last_state) = rollout.batch_rollout(
            model=policy_model,
            n_realizations=self.n_realizations,
            key=key,
        )

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

    def run_optimization_l2co(
            self,
            experiment_sample: ExperimentSample,
            task: Task,
            model: L2CO_model,
            train: bool = False) -> ExperimentSample:
        t1 = time()

        total_iterations = self.budget_fn(task=task)

        iteration_stepsize = int(self.iteration_stepsize * total_iterations)
        memory_size = int(self.memory_size * total_iterations)

        env = Env()
        env_params = EnvParams.init(
            task=task,
            schedules=model.optimizers,
            sampler=self.sampler,
            n_iterations=iteration_stepsize,
            total_iterations=total_iterations,
            memory_size=memory_size,
            bounded=self.bounded,
        )

        key = jr.key(self.seed)

        rollout = RolloutWrapper(env, env_params)

        if train:
            train_key, _ = jr.split(key)
            model, _, _ = rollout.train(
                model=model,
                optimizer=optax.adam(learning_rate=0.01),
                n_episodes=100,
                key=train_key,
            )

        (obs, action, reward, next_obs, logits, done, cum_return,
         last_state) = rollout.batch_rollout(
            model=model,
            n_realizations=self.n_realizations,
            key=key,
        )

        data_xarr = last_state.history_state.to_xarray_realizations()

        experiment_sample.store(
            name="loss history",
            object=data_xarr, to_disk=True)

        logger.info((
            f"Finished optimization task {task.name} with L2CO model with"
            f" {total_iterations} "
            f"iterations for {self.n_realizations} realizations"
            f" (took {time() - t1:.2f} seconds)"
        ))

        return experiment_sample
