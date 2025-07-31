"""
Module for running optimization tasks using a JAX-based environment
and policy model.
"""

#                                                                       Modules
# =============================================================================

# Standard
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
import optax
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from jaxtyping import PRNGKeyArray, PyTree
from tensorflow_probability.substrates import jax as tfp

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
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape

    @staticmethod
    @eqx.filter_jit
    def select_action(
        model: eqx.Module,
        obs: jax.Array,
        state: EnvState,
        rng: PRNGKeyArray,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:

        value, logits = jax.vmap(model, in_axes=(0, None))(obs, state.time)
        pi = tfp.distributions.Categorical(logits=logits)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)

        # Return value flattened to (num_envs,)
        return action, log_prob, value[:, 0]

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def batch_reset(self, key: PRNGKeyArray, num_envs: int,
                    fix_initial_conditions: bool = False):
        if fix_initial_conditions:
            return self.batch_reset_fixed(key, num_envs)
        else:
            return jax.vmap(self.env.reset, in_axes=(0, None))(
                jr.split(key, num_envs), self.env_params
            )

    @partial(jax.jit, static_argnums=(0, 2))
    def batch_reset_fixed(self, key: PRNGKeyArray, num_envs: int):
        obs, state = self.env.reset(key, self.env_params)
        obs = jnp.repeat(obs[None, :], repeats=num_envs, axis=0)
        state = jax.tree.map(lambda x: jnp.repeat(
            x[None, ...], num_envs, axis=0), state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    # @partial(jax.jit, static_argnums=(0, 3, 4))
    @eqx.filter_jit
    def batch_evaluate(self, key: PRNGKeyArray,
                       model: eqx.Module,
                       num_envs: int,
                       fix_initial_conditions: bool = False):
        """Rollout an episode with lax.scan."""

        # Reset the environment
        obs, state = self.batch_reset(
            key, num_envs, fix_initial_conditions)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng, cum_reward, valid_mask = state_input
            next_rng, rng_step, rng_net = jr.split(rng, 3)

            action, _, _ = self.select_action(model, obs, state, rng_net)

            next_o, next_s, reward, done, _ = self.batch_step(
                jr.split(rng_step, num_envs),
                state,
                action.squeeze(),
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_o,
                next_s,
                next_rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_o, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                key,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )
        obs, action, reward, next_o, done = scan_out
        cum_return = carry_out[-2].squeeze()
        last_state = carry_out[1]
        return (obs, action, reward, next_o, done,
                cum_return, last_state)

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate_precomputed(
            self, key: PRNGKeyArray,
            actions: jax.Array,
            fix_initial_conditions: bool = False):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        num_envs = actions.shape[0]

        obs, state = self.batch_reset(
            key, num_envs, fix_initial_conditions)

        def policy_step(state_input, action):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng, cum_reward, valid_mask = state_input
            next_rng, rng_step, _ = jr.split(rng, 3)

            next_o, next_s, reward, done, _ = self.batch_step(
                jr.split(rng_step, num_envs),
                state,
                action.squeeze(),
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_o,
                next_s,
                next_rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_o, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                key,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            actions.T,
            # self.env_params.max_steps_in_episode,
        )
        obs, action, reward, next_o, done = scan_out
        cum_return = carry_out[-2].squeeze()
        last_state = carry_out[1]
        return (obs, action, reward, next_o, done,
                cum_return, last_state)

    def batch_rollout(self, model: eqx.Module,
                      n_realizations: int, key: PRNGKeyArray,
                      fix_initial_conditions: bool = False):
        params, static = eqx.partition(model, eqx.is_inexact_array)

        key_reset, key_episode = jr.split(key)
        key_episode = jr.split(key_episode, n_realizations)
        if fix_initial_conditions:
            in_axes = (None, None, None, 0)
        else:
            key_reset = jr.split(key_reset, n_realizations)
            in_axes = (None, None, 0, 0)

        return jax.vmap(self.single_rollout,
                        in_axes=in_axes,
                        out_axes=0)(params, static, key_reset, key_episode)

    def batch_rollout_precomputed(self, schedules: jax.Array,
                                  key: PRNGKeyArray,
                                  fix_initial_conditions: bool = False):
        k1, k2 = jr.split(key)
        key_episode = jr.split(k2, schedules.shape[0])

        if fix_initial_conditions:
            key_reset = k1
            in_axes = (0, None, 0)
        else:
            key_reset = jr.split(k1, schedules.shape[0])
            in_axes = (0, 0, 0)

        return jax.vmap(self.precomputed_rollout,
                        in_axes=in_axes,
                        out_axes=0)(schedules, key_reset, key_episode)

    @partial(jax.jit, static_argnames=("self"))
    def precomputed_rollout(
            self, schedule: jax.Array,
            key_reset: PRNGKeyArray, key_episode: PRNGKeyArray):
        obs, state = self.env.reset(key_reset, self.env_params)

        def policy_step(state_input: Tuple[jax.Array, EnvState, PRNGKeyArray,
                                           jax.Array, jax.Array], action: int):
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jr.split(key, 3)

            # logits = policy_model(obs, state.time)
            # action = jr.categorical(key_net, logits)
            # action = jnp.squeeze(action)
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
            schedule,
            self.env_params.max_steps_in_episode,
        )

        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        last_state = carry_out[1]
        return (obs, action, reward, next_obs, done,
                cum_return, last_state)

    @partial(jax.jit, static_argnames=("self", "static",))
    def single_rollout(self, params: PyTree, static: PyTree,
                       key_reset: PRNGKeyArray, key_episode: PRNGKeyArray):
        policy_model = eqx.combine(params, static)
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
              n_episodes: int, *, key: PRNGKeyArray,
              fix_initial_conditions: bool = False
              ) -> Tuple[eqx.Module, optax.OptState, jax.Array]:
        params, static = eqx.partition(model, eqx.is_inexact_array)
        key_reset, key_episode = jr.split(key)

        scan_fn = partial(policy_train_step,
                          static=static,
                          rollout=self,
                          optimizer=optimizer,
                          fix_initial_conditions=fix_initial_conditions,
                          )

        carry_init = (params, optimizer.init(params), key_reset, key_episode)

        (carry_final, losses) = jax.lax.scan(
            scan_fn,
            carry_init,
            xs=None,
            length=n_episodes,
        )
        params, opt_state, key_reset, key_episode = carry_final

        return eqx.combine(params, static), opt_state, losses


def policy_train_step(
    carry: Tuple[PyTree, optax.OptState, PRNGKeyArray, PRNGKeyArray],
    _: Any,
    static: PyTree,
    rollout: RolloutWrapper,
    optimizer: optax.GradientTransformation,
    fix_initial_conditions: bool
) -> Tuple[Tuple[PyTree, optax.OptState, PRNGKeyArray, PRNGKeyArray],
           jax.Array]:

    params, opt_state, key_reset, key_episode = carry

    if not fix_initial_conditions:
        key_reset, _ = jr.split(key_reset)

    key_episode, _ = jr.split(key_episode)

    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(
        loss_fn)(params, static, rollout, key_reset, key_episode)

    # Update model
    updates, opt_state = optimizer.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)

    new_carry = (params, opt_state, key_reset, key_episode)
    return new_carry, loss


def loss_fn(params: PyTree, static: PyTree, rollout: RolloutWrapper,
            key_reset: PRNGKeyArray, key_episode: PRNGKeyArray) -> jax.Array:
    # One rollout using the current model
    (obs, action, reward, next_obs,
     logits, done, cum_return, _) = rollout.single_rollout(
        params, static, key_reset, key_episode)

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
