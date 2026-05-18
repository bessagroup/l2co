"""D3: bare optax.lbfgs on the same scaled Sphere (no l2co wiring).

Reproduces the loss surface that l2co's `create_bbob_task` constructs:
``loss(p) = bbob_sphere(scale_input(p))`` where ``scale_input`` maps
``[0,1]^d`` to ``[-5, 5]^d``. Then runs optax.lbfgs with the canonical
value_and_grad_from_state recipe and reports per-step output and the
line-search internals.

Goal: see whether bare optax shows the same linear convergence ratio
~0.76/step as the l2co wiring (=> issue is in optax/the line-search config)
or whether bare optax converges quadratically (=> issue is the wiring).

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3_bare_optax_lbfgs.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from bbob_jax import registry


def make_loss(ndim: int, seed: int):
    bbob_fn, f_opt = registry["sphere"](ndim=ndim, key=jr.key(seed))

    # scale [0,1]^d -> [-5,5]^d (matches l2co_tasks.scale_input default)
    def loss(p):
        x_native = -5.0 + 10.0 * p
        return bbob_fn(x_native)

    return loss, float(f_opt)


def run(
    ndim: int, seed: int, n_iterations: int, initial_strategy: str
) -> None:
    loss, f_opt = make_loss(ndim, seed)
    key = jr.key(seed)
    # Match l2co's normal_sampling around model=zeros, std=1.
    p = jr.normal(key, (ndim,))
    opt = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20, initial_guess_strategy=initial_strategy
        )
    )
    opt_state = opt.init(p)

    v_and_g = optax.value_and_grad_from_state(loss)

    qvs = []
    step_sizes = []
    for k in range(n_iterations):
        value, grad = v_and_g(p, state=opt_state)
        updates, opt_state = opt.update(
            grad, opt_state, p, value=value, grad=grad, value_fn=loss
        )
        # extract the line-search step length if present in state
        ls_state = opt_state[-1] if isinstance(opt_state, tuple) else opt_state
        # optax 0.2.x exposes the chosen step length via tree_get('learning_rate')
        try:
            import optax.tree_utils as otu

            alpha = otu.tree_get(opt_state, "learning_rate")
        except Exception:
            alpha = None
        new_p = optax.apply_updates(p, updates)
        p = new_p
        # Re-evaluate at the new iterate to log loss (state cache may not be present yet)
        loss_val = float(loss(p))
        qvs.append(loss_val - f_opt)
        step_sizes.append(None if alpha is None else float(alpha))

    qvs = np.asarray(qvs)
    cum = np.minimum.accumulate(qvs)
    show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 49, 99, n_iterations - 1]
    show = [i for i in show if i < n_iterations]
    print(
        f"\nndim={ndim} seed={seed} f_opt={f_opt:+.6f} "
        f"initial={initial_strategy} steps={n_iterations}"
    )
    print("  step | qv (loss-f_opt)   cummin-qv      alpha")
    for i in show:
        alpha = step_sizes[i]
        a_str = "n/a" if alpha is None else f"{alpha:.3e}"
        print(f"  {i:4d} | {qvs[i]:+.3e}        {cum[i]:+.3e}     {a_str}")


if __name__ == "__main__":
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    for ndim in (10, 20, 40):
        for seed in (0, 1):
            run(ndim=ndim, seed=seed, n_iterations=50, initial_strategy="one")
