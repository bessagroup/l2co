"""
Module containing the PPO implementation for reinforcement learning.

"""

#                                                                       Modules
# =============================================================================


# Standard
from collections import defaultdict
from functools import partial
from typing import NamedTuple, Tuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import tqdm
from jaxtyping import PRNGKeyArray
from tensorflow_probability.substrates import jax as tfp

# Local
from .environment import Env, EnvParams, EnvState
from .run_optimization import RolloutWrapper

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================

#                                                                 Batch Manager
# =============================================================================


class BatchManager:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        action_size,
        state_space,
    ):
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda

        try:
            _ = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *self.action_size),
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
            "states": buffer["states"].at[buffer["_p"]].set(state),
            "actions": buffer["actions"].at[buffer["_p"]].set(action),
            "rewards": buffer["rewards"].at[buffer["_p"]].set(
                reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_p"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_p"]].set(value),
            "_p": (buffer["_p"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"],
            reward=buffer["rewards"],
            done=buffer["dones"],
        )
        batch = (
            buffer["states"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * \
                value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]


class PPOConfig(NamedTuple):
    num_train_steps: int = 150000
    num_train_envs: int = 8
    num_test_rollouts: int = 164
    n_steps: int = 10
    n_minibatch: int = 4
    epoch_ppo: int = 4
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_begin: float = 5e-04
    lr_end: float = 5e-04
    lr_warmup: float = 0.05
    max_grad_norm: float = 0.5
    evaluate_every_epochs: int = 1000


def train_ppo(env: Env, env_params: EnvParams, key: PRNGKeyArray,
              config: PPOConfig, model: eqx.Module,
              fix_initial_conditions: bool):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    schedule_fn = optax.linear_schedule(
        init_value=-float(config.lr_begin),
        end_value=-float(config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

    # # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutWrapper(env, env_params)

    batch_manager = BatchManager(
        discount=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps + 1,
        num_envs=config.num_train_envs,
        action_size=rollout_manager.action_size,
        state_space=rollout_manager.observation_space,
    )

    @eqx.filter_jit
    def get_transition(
        model: eqx.Module,
        obs: jax.Array,
        state: EnvState,
        batch: BatchManager,
        key: PRNGKeyArray,
        num_train_envs: int,
    ):
        action, log_pi, value = rollout_manager.select_action(
            model, obs, state, key
        )

        next_key, key_step = jr.split(key)

        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(
            jr.split(key_step, num_train_envs), state, action
        )
        batch = batch_manager.append(
            batch, obs, action, reward, done, log_pi, value
        )
        return model, next_obs, next_state, batch, next_key

    batch = batch_manager.reset()

    key, key_step, key_reset, key_eval, key_update = jr.split(key, 5)

    if fix_initial_conditions:
        key_eval = key_reset

    obs, state = rollout_manager.batch_reset(
        key=key_reset,
        num_envs=config.num_train_envs,
        fix_initial_conditions=fix_initial_conditions
    )

    total_steps = 0
    log_steps, log_return = [], []
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)
    for step in t:
        # key_step is updated each time step
        model, obs, state, batch, key_step = get_transition(
            model,
            obs,
            state,
            batch,
            key_step,
            config.num_train_envs,
        )
        total_steps += config.num_train_envs
        if step % (config.n_steps + 1) == 0:
            # key_update is updated each update
            metric_dict, model, opt_state, key_update = update(
                model=model,
                opt_state=opt_state,
                optimizer=tx,
                batch=batch_manager.get(batch),
                num_envs=config.num_train_envs,
                n_steps=config.n_steps,
                n_minibatch=config.n_minibatch,
                epoch_ppo=config.epoch_ppo,
                clip_eps=config.clip_eps,
                entropy_coeff=config.entropy_coeff,
                critic_coeff=config.critic_coeff,
                key=key_update,
            )
            batch = batch_manager.reset()

            # If we want to fix the environment stochastics, do not
            # change the key_reset
            if not fix_initial_conditions:
                key_reset, _ = jr.split(key_reset)

            obs, state = rollout_manager.batch_reset(
                key_reset, config.num_train_envs,
                fix_initial_conditions
            )

        if (step + 1) % config.evaluate_every_epochs == 0:
            key, key_eval = jr.split(key)
            (_, _, _, _, _, cum_return, _) = rollout_manager.batch_evaluate(
                key=key_eval,
                model=model,
                num_envs=config.num_test_rollouts,
                fix_initial_conditions=fix_initial_conditions,
            )
            rewards = jnp.mean(cum_return)
            log_steps.append(total_steps)
            log_return.append(rewards)
            t.set_description(f"R: {str(rewards)}")
            t.refresh()

    return (
        log_steps,
        log_return,
        model,
    )


@jax.jit
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def loss_actor_and_critic(
    model: eqx.Module,
    obs: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> jnp.ndarray:

    value_pred, logits = jax.vmap(model)(obs)
    pi = tfp.distributions.Categorical(logits=logits)

    value_pred = value_pred[:, 0]

    # TODO: Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi
    log_prob = pi.log_prob(action[..., -1])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.mean()
    gae = (gae - gae_mean) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor + critic_coeff * value_loss - entropy_coeff * entropy
    )

    return total_loss, (
        value_loss,
        loss_actor,
        entropy,
        value_pred.mean(),
        target.mean(),
        gae_mean,
    )


def update(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: Tuple,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    epoch_ppo: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    key: PRNGKeyArray,
):
    """Perform multiple epochs of updates with multiple updates."""
    obs, action, log_pi_old, value, target, gae = batch
    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch
    idxes = jnp.arange(num_envs * n_steps)
    avg_metrics_dict = defaultdict(int)
    key, key_perm = jr.split(key)

    for _ in range(epoch_ppo):
        key_perm, _ = jr.split(key_perm)
        idxes = jr.permutation(key_perm, idxes)
        idxes_list = [
            idxes[start: start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]

        model, opt_state, total_loss = update_epoch(
            model,
            opt_state,
            optimizer,
            idxes_list,
            flatten_dims(obs),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            jnp.array(flatten_dims(target)),
            jnp.array(flatten_dims(gae)),
            clip_eps,
            entropy_coeff,
            critic_coeff,
        )

        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            value_pred,
            target_val,
            gae_val,
        ) = total_loss

        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)

    for k, v in avg_metrics_dict.items():
        avg_metrics_dict[k] = v / (epoch_ppo)

    return avg_metrics_dict, model, opt_state, key


@eqx.filter_jit
def update_epoch(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    idxes: jnp.ndarray,
    obs,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
):
    for idx in idxes:
        grad_fn = eqx.filter_value_and_grad(
            loss_actor_and_critic, has_aux=True)
        # grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            model,
            obs=obs[idx],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        # train_state = train_state.apply_gradients(grads=grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        # updates, opt_state = optax.apply_updates(grads, opt_state)
        model = eqx.apply_updates(model, updates)

    return model, opt_state, total_loss
