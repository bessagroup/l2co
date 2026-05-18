"""D2: trace the l2co L-BFGS trajectory on a Sphere realization.

Runs RolloutWrapper with optimizer "lbfgs" on a randomized Sphere
(seed 0, ndim 10), 200 iterations, 1 realization. Reports the
per-step ``output_min - f_opt`` to see where (and when) the trajectory
plateaus relative to the ULP precision floor measured in D1.

Run from the l2co venv:

    uv run --no-sync python experiments/sphere_precision_check/d2_l2co_lbfgs_trace.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout.rollout_wrapper import RolloutWrapper
from l2co._src.sampler import normal_sampling
from l2co_tasks import create_bbob_task


def run(
    optimizer_name: str,
    ndim: int,
    seed: int,
    n_iterations: int,
    hyperparameters: dict | None = None,
) -> None:
    hyperparameters = hyperparameters or {}
    task = create_bbob_task(
        fn_name="sphere", seed=seed, dimensionality=ndim, noise=0.0
    )
    f_opt = task.global_min
    u = float(jnp.spacing(jnp.asarray(f_opt, dtype=jnp.float32)))
    hp_str = ",".join(f"{k}={v}" for k, v in hyperparameters.items()) or "{}"
    print(
        f"optimizer={optimizer_name}({hp_str})  ndim={ndim}  seed={seed}  "
        f"f_opt={f_opt:+.6f}  ULP(f_opt)={u:.3e}"
    )

    optimizer = OptimizationStep(
        optimizer_name, hyperparameters=hyperparameters
    )
    key = jr.key(seed)
    rw = RolloutWrapper.init(
        optimizer=optimizer, task=task, sampler=normal_sampling, key=key
    )

    _, _, history = rw.batch_evaluate(
        key=key, n_iterations=n_iterations, n_realizations=1
    )

    output_min = np.asarray(history.output_min[0])  # (n_iterations,)
    cummin = np.minimum.accumulate(output_min)
    qv = output_min - f_opt
    qv_cum = cummin - f_opt

    show_idx = list(range(0, min(10, n_iterations))) + [
        max(0, int(n_iterations * frac) - 1)
        for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    ]
    show_idx = sorted(set(i for i in show_idx if 0 <= i < n_iterations))
    print("  step | output_min        qv=output_min-f_opt | cummin-f_opt")
    for i in show_idx:
        print(
            f"  {i:5d} | {output_min[i]:+.8e}   {qv[i]:+.3e}     "
            f"{qv_cum[i]:+.3e}"
        )
    final = qv_cum[-1]
    # detect plateau: smallest decrease in last 20% of run
    tail_lo = int(0.8 * n_iterations)
    tail_diffs = np.diff(cummin[tail_lo:])
    tail_max_decrease = float(-tail_diffs.min()) if tail_diffs.size else 0.0
    hits_target = float(final) <= 1e-8
    print(
        f"  final cummin qv = {final:+.3e}  "
        f"(target 1e-8: {'REACHED' if hits_target else 'MISSED'})"
        f"  tail max decrease/step = {tail_max_decrease:.3e}\n"
    )


if __name__ == "__main__":
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}\n")
    configs = [
        ("lbfgs", {}),
        ("adam", {"learning_rate": 0.01}),
    ]
    # Match the production BBOB config: seed=0 only, dims {2,3,5,10,20,40}.
    # Budget = 1000 * d, per `based_on_dim.yaml`.
    for ndim in (2, 3, 5, 10, 20, 40):
        for opt, hp in configs:
            run(
                optimizer_name=opt,
                ndim=ndim,
                seed=0,
                n_iterations=1000 * ndim,
                hyperparameters=hp,
            )
