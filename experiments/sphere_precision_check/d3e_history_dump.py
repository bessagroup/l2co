"""D3e: extract the actual OptHistory from a real l2co L-BFGS rollout.

batch_evaluate returns HistoryState which only keeps summary stats
(output_min etc.). To see the actual params/grads at each step we
need to dive into the inner ``run_state.update_class.step`` path —
but that's not exposed publicly. Easiest workaround: monkey-patch
``optax_extra_args_fn`` to dump diagnostics per step.

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3e_history_dump.py
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.tree_util import Partial
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout.rollout_wrapper import RolloutWrapper
from l2co._src.sampler import normal_sampling
from l2co._src.optimizers import optax_implementations as opi
from l2co_tasks import create_bbob_task


# Patch optax_extra_args_fn to ALSO emit the input_params and updates as part
# of history.params/grads so we can inspect them.
ORIGINAL = opi.optax_extra_args_fn


def patched_optax_extra_args_fn(
    static, optimizer, loss_fn, bounded, opt_hash, pass_rng, **kwargs
):
    init_fn, step_fn = ORIGINAL(
        static, optimizer, loss_fn, bounded, opt_hash, pass_rng, **kwargs
    )
    return init_fn, step_fn


# Build task + rollout, then introspect step-by-step by manually unrolling.
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

    # Run for a few iterations and look at HistoryState.output_min
    # AND get the actual params at each step by reaching into the inner
    # step_fn -- the OptHistory contains per-step params.
    from l2co._src.run_state import evaluate

    # Call the un-vmapped evaluate directly with shape-(1, ndim) params:
    # we want to inspect history.params which is shape (n_iterations+1, popsize, ndim)
    update_class = rw.run_state.update_class

    # Replicate what `evaluate` does, but keep the OptHistory:
    # Reset to get initial sampled params
    from l2co._src.run_state import reset

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

    print(
        f"After reset: params shape={rs.params.shape} norm={float(jnp.linalg.norm(rs.params)):+.3e}"
    )
    print(f"  p_init = {np.asarray(rs.params[0])}")
    print(
        f"  loss(p_init) = {float(loss_fn(rs.params[0]))} (qv={float(loss_fn(rs.params[0])) - f_opt:+.3e})"
    )

    # Run inner step loop manually -- look at the carry between iterations
    # (UpdateClass.step internally builds a scan_loop with do_step/skip_step)
    n_iter = 5
    (
        params,
        best_params,
        best_loss,
        opt_state,
        batch_state,
        history_state,
    ) = update_class.step(
        opt_state=rs.opt_state,
        params=rs.params,
        batch_state=rw.batch_state,
        dataset=rw.dataset,
        key=key,
        n_iterations=n_iter,
        verbose=False,
    )

    print(f"\nAfter {n_iter} L-BFGS iterations via UpdateClass.step:")
    print(f"  output_min      = {np.asarray(history_state.output_min)}")
    print(
        f"  qv (vs f_opt)   = {np.asarray(history_state.output_min) - f_opt}"
    )
    print(f"  final params    = {np.asarray(params[0])}")
    print(
        f"  final loss      = {float(loss_fn(params[0]))} (qv={float(loss_fn(params[0])) - f_opt:+.3e})"
    )

    # Now run bare optax.lbfgs on the SAME initial point:
    p0 = jnp.asarray(rs.params[0])
    print(f"\nBare optax.lbfgs from same p0:")
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(p0)
    v_and_g = optax.value_and_grad_from_state(loss_fn)
    p = p0
    for step in range(n_iter):
        value, grad = v_and_g(p, state=opt_state)
        updates, opt_state = optimizer.update(
            grad, opt_state, p, value=value, grad=grad, value_fn=loss_fn
        )
        p = optax.apply_updates(p, updates)
        print(
            f"  step {step}: loss = {float(loss_fn(p))} (qv={float(loss_fn(p)) - f_opt:+.3e})"
        )


if __name__ == "__main__":
    main()
