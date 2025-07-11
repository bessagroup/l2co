"""
Module for creating the reinforcement learning environment.
"""

#                                                                       Modules
# =============================================================================

# Standard
from functools import partial
from typing import Any, Callable, Iterable, Optional, Tuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

# Local
from .history_state import HistoryState
from .optimizers import OPTIMIZER_MAPPING, UpdateClass
from .optimizerschedule import OptimizationStep
from .task import Task
from .tree_utils import (bfill, encode, ffill, output_cummin, staircase,
                         trajectory)
from .update_state import UpdateState, new

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class EnvState(eqx.Module):
    update_state: UpdateState
    opt_states: Tuple[PyTree]
    history_state: HistoryState
    time: float


class EnvParams(eqx.Module):
    task: Task = None
    optimizers: Tuple[UpdateClass] = None
    sampler: Callable = None
    n_iterations: int = 1
    total_iterations: int = 100
    memory_size: int = 10
    bounded: Optional[Tuple[float, float]] = (None, None)

    @property
    def max_steps_in_episode(self) -> int:
        return self.total_iterations // self.n_iterations

    @property
    def encode_mapping(self) -> jax.Array:
        return jnp.array(
            [0, 1] + [v.hash for v in self.optimizers], dtype=jnp.uint32
        )

    @classmethod
    def init(cls, task: Task, schedules: Iterable[OptimizationStep],
             sampler: Callable,
             n_iterations: int, total_iterations: int, memory_size: int,
             bounded: Tuple[float, float]):

        optimizers = tuple(
            OPTIMIZER_MAPPING[step.optimizer](
                **step.hyperparameters,
                task=task,
                opt_hash=step.hash,
                bounded=bounded,
                opt_index=idx,
                n_optimizers=len(schedules),
            ) for idx, step in enumerate(schedules)
        )

        return cls(
            task=task,
            optimizers=optimizers,
            sampler=sampler,
            n_iterations=n_iterations,
            total_iterations=total_iterations,
            memory_size=memory_size,
            bounded=bounded,
        )


class Env():
    @partial(jax.jit, static_argnames=("self", "params",))
    def reset(self, key: jax.Array,
              params: EnvParams) -> Tuple[jax.Array, EnvState]:

        obs, state = self.reset_env(key, params)
        return obs, state

    @partial(jax.jit, static_argnames=("self", "params",))
    def step(self, key: jax.Array, state: EnvState, action: int,
             params: EnvParams):
        obs, state, reward, done, info = self.step_env(
            key, state, action, params
        )

        return obs, state, reward, done, info

    def reset_env(self, key: jax.Array,
                  params: EnvParams) -> Tuple[jax.Array, EnvState]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        key : jax.Array
            Random key for reproducibility.
        params : EnvParams
            Parameters for the environment.

        Returns
        -------
        Tuple[jax.Array, EnvState]
            Initial observation and environment state.
        """
        model_params, _ = eqx.partition(
            params.task.model, eqx.is_inexact_array)

        update_state = UpdateState.init(
            params=model_params,
            memory_size=params.memory_size,
            key=key,
        )

        update_state = new(update_state=update_state,
                           task=params.task, sampler=params.sampler)

        history_state = HistoryState.init(
            n_iterations=params.total_iterations+1)

        # Add initial state to history
        history_state = history_state.add({
            'loss': jax.tree.map(
                lambda x: jnp.expand_dims(x, (0, 1)), update_state.best_loss),
            'update_step': jax.tree.map(
                lambda x: jnp.expand_dims(x, (0, 1)), update_state.last_step),
        })

        opt_states = tuple(opt.init(update_state) for opt in params.optimizers)

        state = EnvState(
            update_state=update_state,
            opt_states=opt_states,
            history_state=history_state,
            time=0.0,
        )

        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams,
                key=None) -> jax.Array:
        return jnp.stack(
            (
                trajectory(state.history_state.output_mean),
                staircase(state.history_state.output_min),
                encode(state.history_state.update_step, params.encode_mapping),
            ),
            axis=0,
            dtype=jnp.float32,
        ).T

    def step_env(
        self, key: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """
        Perform a single step in the environment.

        Parameters
        ----------
        key : jax.Array
            Random key for reproducibility.
        state : EnvState
            Current state of the environment.
        action : int
            Action to perform.
        params : EnvParams
            Parameters for the environment.

        Returns
        -------
        Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]
            Observation, updated state, reward, done flag, and additional info.
        """

        # === Static dispatch via lax.switch === #
        def make_branch(optim: UpdateClass):
            return lambda _: optim.step(
                opt_states=state.opt_states,
                update_state=state.update_state,
                history_state=state.history_state,
                n_iterations=params.n_iterations,
            )

        branches = [make_branch(optim) for optim in params.optimizers]

        # Run the optimizer step based on the action
        update_state, opt_states, history_state = jax.lax.switch(
            action,
            branches,
            operand=None,
        )

        # Update the time
        time = state.time + (params.n_iterations / params.total_iterations)

        # Create the new state
        state = EnvState(
            opt_states=opt_states,
            update_state=update_state,
            history_state=history_state,
            time=time,
        )

        done = state.time >= 1.0

        # reward = self.compute_reward(
        #     history_state=history_state,
        #     n_iterations=params.n_iterations,
        # )

        reward = jax.lax.cond(
            done,
            lambda x: self.compute_reward2(history_state=x),
            lambda x: 0.0,
            history_state,
        )

        info = {}

        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            info,
        )

    def compute_reward2(
        self,
        history_state: HistoryState,
    ) -> jax.Array:
        x = history_state.output_min
        x = ffill(bfill(x))
        x = output_cummin(x)

        return -(x.min() + 0.1 * (x.argmin() / x.shape[0]))

    def compute_reward(
        self,
        history_state: HistoryState,
        n_iterations: int
    ) -> jax.Array:
        """
        Compute the reward as the difference in output_min over n_iterations.

        Parameters
        ----------
        history_state : HistoryState
            The updated history state after applying the step.
        n_iterations : int
            The number of steps ago to compare against.

        Returns
        -------
        jax.Array
            The reward value (scalar).
        """
        output = staircase(history_state.output_min)  # shape: (T,)
        last = output[-1]
        prev = output[-(n_iterations + 1)]
        reward = prev - last  # improvement in loss (positive if improved)
        return reward
