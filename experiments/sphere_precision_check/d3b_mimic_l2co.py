"""D3b: Python-loop replica of l2co's optax_extra_args_fn step body.

Goal: isolate whether the l2co wiring or the surrounding ``jax.lax.scan``
is causing the linear-convergence regression vs bare optax (D3).

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3b_mimic_l2co.py
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.tree_util import Partial
from l2co_tasks import create_bbob_task


def make_task_loss(seed: int, ndim: int):
    task = create_bbob_task(
        fn_name="sphere", seed=seed, dimensionality=ndim, noise=0.0
    )
    return task, task.loss_fn, float(task.global_min)


def initial_point(task, key, std: float = 1.0):
    return task.model + std * jr.normal(key, task.model.shape)


def _run_with_step(p, opt_state, optimizer, step_fn, n_iterations, loss_fn):
    qvs = []
    for _ in range(n_iterations):
        p, opt_state, loss_val = step_fn(p, opt_state)
        qvs.append(float(loss_val))
    return np.asarray(qvs)


def run_l2co_style(ndim: int, seed: int, n_iterations: int):
    task, loss_fn, f_opt = make_task_loss(seed, ndim)
    key = jr.key(seed)
    p = initial_point(task, key)

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(p)

    @jax.jit
    def step(p, opt_state):
        # Mirror optax_extra_args_fn body: fn rebuilt via Partial.
        fn = Partial(loss_fn)
        loss, grads = optax.value_and_grad_from_state(fn)(p, state=opt_state)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, p, value=loss, grad=grads, value_fn=fn
        )
        new_p = eqx.apply_updates(p, updates)
        return new_p, new_opt_state, loss_fn(new_p)

    qvs = _run_with_step(p, opt_state, optimizer, step, n_iterations, loss_fn)
    return qvs - f_opt


def run_bare(ndim: int, seed: int, n_iterations: int):
    task, loss_fn, f_opt = make_task_loss(seed, ndim)
    key = jr.key(seed)
    p = initial_point(task, key)

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(p)
    v_and_g = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(p, opt_state):
        value, grad = v_and_g(p, state=opt_state)
        updates, new_opt_state = optimizer.update(
            grad, opt_state, p, value=value, grad=grad, value_fn=loss_fn
        )
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_state, loss_fn(new_p)

    qvs = _run_with_step(p, opt_state, optimizer, step, n_iterations, loss_fn)
    return qvs - f_opt


def run_l2co_scan(ndim: int, seed: int, n_iterations: int):
    """Same wiring as run_l2co_style but driven by jax.lax.scan."""
    task, loss_fn, f_opt = make_task_loss(seed, ndim)
    key = jr.key(seed)
    p = initial_point(task, key)

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(p)

    def step(carry, _):
        p, opt_state = carry
        fn = Partial(loss_fn)
        loss, grads = optax.value_and_grad_from_state(fn)(p, state=opt_state)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, p, value=loss, grad=grads, value_fn=fn
        )
        new_p = eqx.apply_updates(p, updates)
        return (new_p, new_opt_state), loss_fn(new_p)

    _, losses = jax.lax.scan(
        step, (p, opt_state), xs=None, length=n_iterations
    )
    return np.asarray(losses) - f_opt


def show(name, qvs):
    cum = np.minimum.accumulate(qvs)
    n = len(qvs)
    idx = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, n - 1]
    idx = [i for i in idx if 0 <= i < n]
    print(f"  {name:14s}", end="")
    for i in idx:
        print(f"  s{i}={qvs[i]:+.2e}", end="")
    print(f"  cum={cum[-1]:+.2e}")


if __name__ == "__main__":
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}\n")
    for ndim in (10,):
        for seed in (0, 1, 2):
            print(f"ndim={ndim} seed={seed}")
            show("bare", run_bare(ndim, seed, 50))
            show("l2co-py", run_l2co_style(ndim, seed, 50))
            show("l2co-scan", run_l2co_scan(ndim, seed, 50))
            print()
