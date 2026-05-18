"""D3d: compare the FIRST step of the real RolloutWrapper L-BFGS
to bare optax.lbfgs called on the same initial point.

If the initial points differ, that's the bug. If they're the same but
step 1 results differ, the bug is in the l2co step body.

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3d_rollout_diff.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout.rollout_wrapper import RolloutWrapper
from l2co._src.sampler import normal_sampling
from l2co_tasks import create_bbob_task


def main():
    ndim, seed = 10, 0
    task = create_bbob_task(
        fn_name="sphere", seed=seed, dimensionality=ndim, noise=0.0
    )
    f_opt = task.global_min
    loss_fn = task.loss_fn

    key = jr.key(seed)
    opt = OptimizationStep("lbfgs", hyperparameters={})
    rw = RolloutWrapper.init(
        optimizer=opt, task=task, sampler=normal_sampling, key=key
    )

    # Run for ONE iteration and inspect the trajectory:
    run_state, batch_state, history = rw.batch_evaluate(
        key=key, n_iterations=2, n_realizations=1
    )
    output_min = np.asarray(history.output_min)  # (1, n_iter)
    print(f"l2co rollout, ndim={ndim} seed={seed} f_opt={f_opt:+.6f}")
    print(f"  output_min[0] = {output_min[0]}")
    print(f"  qv (cumin)    = {output_min[0] - f_opt}")
    init_params = np.asarray(run_state.params[0, 0])  # popsize=1
    print(
        f"  rollout p_init mean={init_params.mean():.3e} std={init_params.std():.3e}"
    )
    # rollout's params after run is final params after 2 iterations
    final = np.asarray(run_state.params[0, 0])
    print(f"  rollout final params: norm={np.linalg.norm(final):.3e}")

    # Re-init the rollout to access the *initial* parameters as the rollout sees them.
    rw2 = RolloutWrapper.init(
        optimizer=opt, task=task, sampler=normal_sampling, key=key
    )
    # NOTE: run_state.params has shape (popsize=1, ndim) -- the rollout's
    # initial point before any optimization step.
    p0 = jnp.asarray(rw2.run_state.params[0])
    print(
        f"  exact rollout p0 mean={float(p0.mean()):.3e} std={float(p0.std()):.3e}"
    )
    print(
        f"  loss(p0) = {float(loss_fn(p0))} (qv={float(loss_fn(p0)) - f_opt:+.3e})"
    )

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(p0)
    v_and_g = optax.value_and_grad_from_state(loss_fn)
    p = p0
    print(
        f"  bare loss at p0 = {float(loss_fn(p))} (qv={float(loss_fn(p)) - f_opt:+.3e})"
    )
    for step in range(3):
        value, grad = v_and_g(p, state=opt_state)
        updates, opt_state = optimizer.update(
            grad, opt_state, p, value=value, grad=grad, value_fn=loss_fn
        )
        p = optax.apply_updates(p, updates)
        print(
            f"  bare step {step}: loss = {float(loss_fn(p))} (qv={float(loss_fn(p)) - f_opt:+.3e})"
        )


if __name__ == "__main__":
    main()
