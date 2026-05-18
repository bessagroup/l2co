"""D1: verify the float32 precision floor on BBOB Sphere.

For a randomized Sphere with f_opt drawn from [-1000, 1000], compute
``loss(x_opt + delta * e_0)`` for a sweep of deltas and report the
quality value ``loss - f_opt``. The expectation is that once
``delta**2 < ULP(f_opt)`` the residual rounds away and the reported
quality value floors at 0 or +-ULP.

Run from the l2co venv:

    uv run --no-sync python experiments/sphere_precision_check/d1_precision_floor.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from bbob_jax import registry


def ulp(x: float) -> float:
    """Return the unit in the last place of ``x`` in its native dtype."""
    arr = jnp.asarray(x)
    return float(jnp.spacing(arr))


def run(ndim: int, seeds: tuple[int, ...]) -> None:
    deltas = np.logspace(-1, -8, 8)
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    print(f"ndim = {ndim}\n")

    for seed in seeds:
        key = jr.PRNGKey(seed)
        fn, f_opt = registry["sphere"](ndim=ndim, key=key)
        f_opt_val = float(f_opt)
        u = ulp(f_opt_val)
        print(
            f"seed={seed}  f_opt={f_opt_val:+.6f}  "
            f"ULP(f_opt)={u:.3e}  dtype={fn(jnp.zeros(ndim)).dtype}"
        )

        for delta in deltas:
            x = jnp.zeros(ndim)
            # Probe along the first axis from x_opt; we need x_opt though,
            # which is captured in the Partial. Re-derive by reading the
            # closure: fn is jax.tree_util.Partial(sphere, x_opt=..., ...).
            x_opt = fn.keywords["x_opt"]
            probe = x_opt.at[0].add(delta)
            loss = float(fn(probe))
            qv = loss - f_opt_val
            analytic_residual = delta**2
            print(
                f"  delta={delta:.1e}  delta^2={analytic_residual:.3e}  "
                f"loss-f_opt={qv:+.3e}"
            )
        print()


if __name__ == "__main__":
    run(ndim=10, seeds=(0, 1, 2, 7))
