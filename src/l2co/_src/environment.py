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
from .tree_utils import staircase, trajectory, tree_where
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

    @classmethod
    def init(cls, task: Task, schedules: Iterable[OptimizationStep],
             sampler: Callable,
             n_iterations: int, total_iterations: int, memory_size: int,
             bounded: Tuple[float, float]):
        optimizers = tuple(
            OPTIMIZER_MAPPING[step.optimizer](
                **step.hyperparameters,
                task=task,
                opt_hash=idx+1,
                bounded=bounded,
                opt_index=jnp.array(idx, dtype=jnp.int32),
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
    def __init__(self,):
        super().__init__()

    @partial(jax.jit, static_argnames=("self", "params",))
    def reset(self, key: jax.Array,
              params: EnvParams) -> Tuple[jax.Array, EnvState]:

        obs, state = self.reset_env(key, params)
        return obs, state

    @partial(jax.jit, static_argnames=("self", "params",))
    @eqx.debug.assert_max_traces(max_traces=100)
    def step(self, key: jax.Array, state: EnvState, action: int,
             params: EnvParams):
        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = tree_where(done, state_re, state_st)
        obs = jnp.where(done, obs_re, obs_st)

        return obs, state, reward, done, info

    def reset_env(self, key: jax.Array,
                  params: EnvParams) -> Tuple[jax.Array, EnvState]:
        model_params, _ = eqx.partition(
            params.task.model, eqx.is_inexact_array)

        update_state = UpdateState.init(
            params=model_params,
            memory_size=params.memory_size,
            key=key,
        )

        update_state = new(update_state=update_state,
                           task=params.task, sampler=params.sampler)

        history_state = HistoryState.init(n_iterations=params.total_iterations)

        opt_states = tuple(opt.init(update_state) for opt in params.optimizers)

        state = EnvState(
            update_state=update_state,
            opt_states=opt_states,
            history_state=history_state,
            time=0.0,
        )

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        return jnp.stack(
            (
                trajectory(state.history_state.output_mean),
                staircase(state.history_state.output_min),
                state.history_state.update_step,
            ),
            axis=0,
            dtype=jnp.float32,
        )

    def step_env(
            self, key: jax.Array, state: EnvState,
            action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:

        # === Static dispatch via lax.switch === #
        def make_branch(optim: UpdateClass):
            return lambda _: optim.step(
                opt_states=state.opt_states,
                update_state=state.update_state,
                history_state=state.history_state,
                n_iterations=params.n_iterations,
            )

        branches = [make_branch(optim) for optim in params.optimizers]

        update_state, opt_states, history_state = jax.lax.switch(
            action,
            branches,
            operand=None,
        )

        done = state.time > 1.0
        time = state.time + (params.n_iterations / params.total_iterations)

        reward = 0.0

        info = {}

        state = EnvState(
            opt_states=opt_states,
            update_state=update_state,
            history_state=history_state,
            time=time,
        )

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            info,
        )
