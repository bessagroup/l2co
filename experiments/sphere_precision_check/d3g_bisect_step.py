"""D3g: bisect the bug by incrementally adding l2co step-fn wrappings to bare optax.

Each variant starts from the same p_init and runs 5 steps. The first
variant that shows slow (linear ~0.76/step) convergence identifies
the offending wrapper.

Run:

    uv run --no-sync python experiments/sphere_precision_check/d3g_bisect_step.py
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.tree_util import Partial
from l2co_tasks import create_bbob_task


def make_setup(ndim: int, seed: int):
    task = create_bbob_task(
        fn_name="sphere", seed=seed, dimensionality=ndim, noise=0.0
    )
    loss_fn = task.loss_fn
    f_opt = task.global_min
    key = jr.key(seed)
    p0 = jnp.array(
        [
            1.0040143,
            -0.9063372,
            -0.7481722,
            -1.1713669,
            -0.8712328,
            0.5888381,
            0.72392994,
            -1.0255982,
            1.661628,
            -1.8910251,
        ]
    )
    return loss_fn, f_opt, p0, key


def trace(name, qvs):
    s = "  ".join(f"s{i}={qvs[i]:+.2e}" for i in range(len(qvs)))
    print(f"  {name:30s} {s}")


def variant_bare(loss_fn, f_opt, p0, key, n=5):
    optimizer = optax.lbfgs()
    p = p0
    opt_state = optimizer.init(p)
    v_and_g = optax.value_and_grad_from_state(loss_fn)
    qvs = []
    for _ in range(n):
        v, g = v_and_g(p, state=opt_state)
        updates, opt_state = optimizer.update(
            g, opt_state, p, value=v, grad=g, value_fn=loss_fn
        )
        p = optax.apply_updates(p, updates)
        qvs.append(float(loss_fn(p)) - f_opt)
    return np.asarray(qvs)


def variant_partial(loss_fn, f_opt, p0, key, n=5):
    """+ Partial(loss_fn) wrapping (matches l2co's Partial(loss_fn, **sample))."""
    optimizer = optax.lbfgs()
    p = p0
    opt_state = optimizer.init(p)
    qvs = []
    for _ in range(n):
        fn = Partial(loss_fn)
        v, g = optax.value_and_grad_from_state(fn)(p, state=opt_state)
        updates, opt_state = optimizer.update(
            g, opt_state, p, value=v, grad=g, value_fn=fn
        )
        p = optax.apply_updates(p, updates)
        qvs.append(float(loss_fn(p)) - f_opt)
    return np.asarray(qvs)


def variant_squeeze(loss_fn, f_opt, p0, key, n=5):
    """+ popsize squeeze/expand pattern + Partial."""
    optimizer = optax.lbfgs()
    params = jnp.expand_dims(p0, axis=0)  # (1, ndim) like l2co carry
    # init_fn does jax.tree.map(lambda x: x[0], params); init(squeezed)
    opt_state = optimizer.init(params[0])
    qvs = []
    for _ in range(n):
        # step_fn:
        input_params = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), params)
        model = eqx.combine(input_params, None)
        fn = Partial(loss_fn)
        v, g = optax.value_and_grad_from_state(fn)(model, state=opt_state)
        updates, opt_state = optimizer.update(
            g, opt_state, input_params, value=v, grad=g, value_fn=fn
        )
        new_params = eqx.apply_updates(input_params, updates)
        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)
        params = new_params
        qvs.append(float(loss_fn(params[0])) - f_opt)
    return np.asarray(qvs)


def variant_squeeze_jit(loss_fn, f_opt, p0, key, n=5):
    """variant_squeeze but jit the step (l2co does scan->jit)."""
    optimizer = optax.lbfgs()
    params = jnp.expand_dims(p0, axis=0)
    opt_state = optimizer.init(params[0])

    @jax.jit
    def step(carry):
        params, opt_state = carry
        input_params = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), params)
        model = eqx.combine(input_params, None)
        fn = Partial(loss_fn)
        v, g = optax.value_and_grad_from_state(fn)(model, state=opt_state)
        updates, new_opt_state = optimizer.update(
            g, opt_state, input_params, value=v, grad=g, value_fn=fn
        )
        new_params = eqx.apply_updates(input_params, updates)
        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)
        return (new_params, new_opt_state), v

    carry = (params, opt_state)
    qvs = []
    for _ in range(n):
        carry, v = step(carry)
        qvs.append(float(loss_fn(carry[0][0])) - f_opt)
    return np.asarray(qvs)


def variant_lr(loss_fn, f_opt, p0, key, n=5):
    """Reproduce l2co's accidental injection of learning_rate=1e-3 into lbfgs."""
    optimizer = optax.lbfgs(learning_rate=1e-3)
    p = p0
    opt_state = optimizer.init(p)
    v_and_g = optax.value_and_grad_from_state(loss_fn)
    qvs = []
    for _ in range(n):
        v, g = v_and_g(p, state=opt_state)
        updates, opt_state = optimizer.update(
            g, opt_state, p, value=v, grad=g, value_fn=loss_fn
        )
        p = optax.apply_updates(p, updates)
        qvs.append(float(loss_fn(p)) - f_opt)
    return np.asarray(qvs)


def main():
    ndim, seed = 10, 0
    loss_fn, f_opt, p0, key = make_setup(ndim, seed)
    print(
        f"loss(p0) = {float(loss_fn(p0))} qv={float(loss_fn(p0)) - f_opt:+.3e}"
    )
    print()

    print("Variant traces (qv = loss - f_opt at end of each step):")
    trace("bare lbfgs()", variant_bare(loss_fn, f_opt, p0, key))
    trace("bare + Partial", variant_partial(loss_fn, f_opt, p0, key))
    trace("bare + Partial + squeeze", variant_squeeze(loss_fn, f_opt, p0, key))
    trace(
        "bare + Partial + squeeze + jit",
        variant_squeeze_jit(loss_fn, f_opt, p0, key),
    )
    trace(
        "lbfgs(learning_rate=1e-3)  [BUG]", variant_lr(loss_fn, f_opt, p0, key)
    )


if __name__ == "__main__":
    main()
