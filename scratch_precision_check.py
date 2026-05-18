"""Check precision of sepcmaes and pso on the BBOB sphere via l2co.

Quick sanity script — runs each evosax optimizer for a generous budget on a
2-D / 5-D / 10-D sphere and reports |best_loss - global_min|.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from l2co_tasks import create_bbob_task

from l2co import OptimizationStep
from l2co._src.rollout import RolloutWrapper
from l2co._src.sampler import normal_sampling


def run_one(
    optimizer_name: str,
    dim: int,
    n_iter: int,
    n_real: int,
    fn_name: str = "sphere",
):
    task = create_bbob_task(
        fn_name=fn_name, seed=0, dimensionality=dim, noise=0.0
    )
    optimizer = OptimizationStep(optimizer=optimizer_name)
    key = jr.key(0)
    rw = RolloutWrapper.init(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
        key=key,
        bounded=(None, None),
    )
    run_state, _, history = rw.batch_evaluate(
        key=key,
        n_iterations=n_iter,
        n_realizations=n_real,
    )
    best_per_real = jnp.asarray(run_state.best_loss).reshape(-1)
    gap = best_per_real - task.global_min
    fevals = int(jnp.asarray(history.fevals).sum() // n_real)
    print(
        f"{optimizer_name:>9s}  d={dim:>2d}  iters={n_iter:>5d}  "
        f"fevals/run~{fevals:>6d}  "
        f"global_min={task.global_min:+.4f}  "
        f"gap min={float(gap.min()):.3e}  "
        f"median={float(jnp.median(gap)):.3e}  "
        f"max={float(gap.max()):.3e}"
    )


if __name__ == "__main__":
    for dim, n_iter in [(2, 500), (5, 2000), (10, 5000)]:
        for name in ["sepcmaes", "pso"]:
            run_one(name, dim=dim, n_iter=n_iter, n_real=3)
        print()
