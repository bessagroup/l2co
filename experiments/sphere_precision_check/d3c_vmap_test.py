"""D3c: does vmapping the L-BFGS rollout break convergence?

The actual l2co `batch_evaluate` is `eqx.filter_vmap(evaluate, ...)`.
Run the same scan-based L-BFGS rollout under vmap vs without, on the
same Sphere task, and see whether vmap alone reproduces the slow
linear convergence observed in the full l2co pipeline.

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3c_vmap_test.py
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from l2co_tasks import create_bbob_task


def run_one(key, loss_fn, optimizer, n_iterations: int, ndim: int):
    p = jr.normal(key, (ndim,))
    opt_state = optimizer.init(p)
    v_and_g = optax.value_and_grad_from_state(loss_fn)

    def step(carry, _):
        p, opt_state = carry
        value, grad = v_and_g(p, state=opt_state)
        updates, new_opt_state = optimizer.update(
            grad, opt_state, p, value=value, grad=grad, value_fn=loss_fn
        )
        new_p = optax.apply_updates(p, updates)
        return (new_p, new_opt_state), loss_fn(new_p)

    _, losses = jax.lax.scan(
        step, (p, opt_state), xs=None, length=n_iterations
    )
    return losses


def show(name: str, losses_arr: np.ndarray, f_opt: float):
    qvs = losses_arr - f_opt
    n = qvs.shape[-1]
    idx = [0, 1, 2, 5, 10, 20, 50, n - 1]
    idx = [i for i in idx if 0 <= i < n]
    if qvs.ndim == 1:
        line = "  ".join(f"s{i}={qvs[i]:+.2e}" for i in idx)
        cum = np.minimum.accumulate(qvs)[-1]
        print(f"  {name:16s} {line}  cum={cum:+.2e}")
    else:
        for r in range(qvs.shape[0]):
            cum = np.minimum.accumulate(qvs[r])[-1]
            line = "  ".join(f"s{i}={qvs[r, i]:+.2e}" for i in idx)
            print(f"  {name}[r={r}]: {line}  cum={cum:+.2e}")


def main():
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}\n")
    ndim, n_iter = 10, 50

    for seed in (0, 1):
        task = create_bbob_task(
            fn_name="sphere", seed=seed, dimensionality=ndim, noise=0.0
        )
        loss_fn = task.loss_fn
        f_opt = task.global_min
        optimizer = optax.lbfgs()
        key = jr.key(seed)

        # 1) Unvmapped, single key
        losses = run_one(key, loss_fn, optimizer, n_iter, ndim)
        print(f"seed={seed} (unvmapped, single key):")
        show("unvmapped", np.asarray(losses), f_opt)

        # 2) Vmapped, n_realizations=1
        keys = jr.split(key, 1)
        vrun = eqx.filter_vmap(
            lambda k: run_one(k, loss_fn, optimizer, n_iter, ndim)
        )
        losses_v1 = vrun(keys)
        print(f"seed={seed} (vmapped, n_realizations=1):")
        show("vmap_n=1", np.asarray(losses_v1), f_opt)

        # 3) Vmapped, n_realizations=4
        keys4 = jr.split(key, 4)
        losses_v4 = vrun(keys4)
        print(f"seed={seed} (vmapped, n_realizations=4):")
        show("vmap_n=4", np.asarray(losses_v4), f_opt)
        print()


if __name__ == "__main__":
    main()
