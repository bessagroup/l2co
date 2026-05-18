"""D3f: drive the l2co step_fn directly, no scan_loop.

Construct the same UpdateClass + step_fn as the real rollout, then run
the step_fn in a python for-loop with the exact same initial carry
as the rollout would have. If this converges fast (like bare optax),
the bug is in scan_loop / do_step / scan_step. If it converges
linearly (like the full rollout), the bug is in step_fn itself.

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3f_step_fn_direct.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout.rollout_wrapper import RolloutWrapper
from l2co._src.run_state import reset
from l2co._src.sampler import normal_sampling
from l2co_tasks import create_bbob_task


def main():
    ndim, seed = 10, 0
    task = create_bbob_task(
        fn_name="sphere", seed=seed, dimensionality=ndim, noise=0.0
    )
    loss_fn = task.loss_fn
    f_opt = task.global_min

    key = jr.key(seed)
    opt = OptimizationStep("lbfgs", hyperparameters={})
    rw = RolloutWrapper.init(
        optimizer=opt, task=task, sampler=normal_sampling, key=key
    )
    update_class = rw.run_state.update_class

    # Same reset as evaluate() does
    rs = reset(
        run_state=rw.run_state,
        batch_state=rw.batch_state,
        static=rw.static,
        dataset=rw.dataset,
        loss_fn=rw.loss_fn,
        pass_rng=rw.pass_rng,
        sampler=rw.sampler,
        key=key,
    )
    print(f"p_init = {np.asarray(rs.params[0])}")
    print(
        f"loss(p_init) = {float(loss_fn(rs.params[0]))} (qv={float(loss_fn(rs.params[0])) - f_opt:+.3e})\n"
    )

    # Drive the step_fn directly without scan_loop
    sample = {}  # BBOB has no dataset
    carry = (rs.params, rs.opt_state, key)
    print("Direct python-loop driving of step_fn (no scan, no scan_step):")
    for k in range(5):
        carry, history = update_class.step_fn(carry, sample=sample)
        # history.params has shape (1, popsize=1, ndim) -- the params BEFORE this step
        params_before = np.asarray(history.params[0, 0])
        loss_seen = float(history.loss[0])
        params_after = np.asarray(carry[0][0])
        loss_after = float(loss_fn(carry[0][0]))
        print(
            f"  step {k}: loss(before)={loss_seen:.4f} (qv={loss_seen - f_opt:+.3e})  "
            f"loss(after)={loss_after:.4f} (qv={loss_after - f_opt:+.3e})"
        )


if __name__ == "__main__":
    main()
