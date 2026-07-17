"""
L-BFGS with a fresh PRNG key per linesearch evaluation.

``optax.lbfgs`` freezes the extra keyword arguments of ``value_fn``
(including a PRNG key) for the whole inner zoom linesearch: the key is
extracted once in ``scale_by_zoom_linesearch.update_fn`` and bound
unchanged into the ``lax.while_loop`` body, so every trial evaluation
of one outer step sees the *same* noise realization. On top of that,
``optax.value_and_grad_from_state`` re-serves the value/gradient cached
at the end of the previous step's linesearch, which was computed under
the *previous* key — each outer step therefore straddles exactly two
realizations (see ``FEEDBACK_optax.md``).

For stochastic objectives whose randomness is driven by an explicit
key (the BBOB-noisy suite, mini-batched losses), the honest semantics
for benchmarking is one independent noise realization per function
evaluation — the same contract the evosax adapters implement by
splitting one key per population member. This module provides that:

* :func:`scale_by_zoom_linesearch_per_eval_key` — a drop-in for
  ``optax.scale_by_zoom_linesearch`` that carries a key through the
  linesearch ``while_loop`` and splits a fresh subkey for every inner
  evaluation.
* :func:`lbfgs_per_eval_key` — ``optax.lbfgs`` chained with that
  linesearch.
* :func:`lbfgs_per_eval_key_update` — the registry factory
  (``optimizer="lbfgs_per_eval_key"``). On deterministic tasks
  (``task.pass_rng`` false) fresh keys are a mathematical no-op, so
  the factory degrades to the stock ``optax.lbfgs`` wiring of
  ``UpdateClass.optax_update_extra_kwargs``.

Per-evaluation keys make L-BFGS strictly *more* exposed to noise than
the stock per-step-realization wiring (the zoom bracketing invariants
then compare values from different realizations), which is the point:
the optimizer is billed the same noise contract as every other
optimizer in the registry. Fevals accounting is unchanged (1 per
iteration, matching the stock ``lbfgs`` entry — see
``docs/adr/0006``).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree
from l2co_tasks import Task
from optax import OptState as OptaxState

# The (init_fn, step_fn, cond_fn) triple is not re-exported by optax's
# public namespace (only the assembled ``scale_by_zoom_linesearch``
# is) — see FEEDBACK_optax.md.
from optax._src.linesearch import zoom_linesearch

# Local
from l2co._src.opt_history import OptHistory
from l2co._src.optimizers.update_class import UpdateClass
from l2co._src.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def scale_by_zoom_linesearch_per_eval_key(
    max_linesearch_steps: int = 20,
) -> optax.GradientTransformationExtraArgs:
    """Zoom linesearch drawing a fresh PRNG key per evaluation.

    Reuses optax's zoom machinery (``zoom_linesearch``'s
    ``init_fn``/``step_fn``/``cond_fn`` triple) but carries a key
    through the ``while_loop`` and splits one subkey per inner
    iteration, injected as the ``key`` keyword of ``value_fn``. Each
    trial evaluation of a stochastic objective therefore draws an
    independent noise realization, instead of the stock behavior
    where the whole linesearch shares the single realization bound at
    ``update`` time.

    The transformation state mirrors
    ``optax.ScaleByZoomLinesearchState`` (``learning_rate``,
    ``value``, ``grad``, ``info``), so
    ``optax.value_and_grad_from_state`` interoperates unchanged.

    Parameters
    ----------
    max_linesearch_steps : int, optional
        Maximum number of linesearch iterations (one evaluation
        each). Matches the default of the linesearch that
        ``optax.lbfgs`` ships with.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        Transformation whose ``update`` requires the extra keyword
        arguments ``value``, ``grad``, ``value_fn`` and ``key``;
        ``value_fn`` must accept a ``key`` keyword argument.
    """
    init_ls, step_ls, cond_step_ls = zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps
    )

    def init_fn(params: PyTree) -> optax.ScaleByZoomLinesearchState:
        """Initialise the linesearch state (mirrors the stock init)."""
        placeholder = jnp.empty((), jax.tree.leaves(params)[0].dtype)
        val_dtype = jnp.real(placeholder).dtype
        return optax.ScaleByZoomLinesearchState(
            learning_rate=jnp.asarray(1.0, dtype=val_dtype),
            value=jnp.asarray(jnp.inf, dtype=val_dtype),
            grad=optax.tree.zeros_like(params),
            info=optax.ZoomLinesearchInfo(
                num_linesearch_steps=jnp.asarray(0),
                decrease_error=jnp.asarray(jnp.inf),
                curvature_error=jnp.asarray(jnp.inf),
            ),
        )

    def update_fn(
        updates: PyTree,
        state: optax.ScaleByZoomLinesearchState,
        params: PyTree,
        *,
        value: jax.Array,
        grad: PyTree,
        value_fn: LossFunction,
        key: PRNGKeyArray,
        **extra_args,
    ) -> tuple[PyTree, optax.ScaleByZoomLinesearchState]:
        """Scale ``updates`` by a stepsize found under per-eval keys.

        ``value`` and ``grad`` seed the Wolfe reference
        (``value_init`` / ``slope_init``); when they come from
        ``optax.value_and_grad_from_state`` they belong to the
        realization of the *previous* step's key — a property of the
        caching utility, not of this linesearch.
        """
        del extra_args
        value_and_grad_fn = jax.value_and_grad(value_fn)

        init_state = init_ls(
            updates,
            params,
            value=value,
            grad=grad,
            prev_stepsize=state.learning_rate,
            initial_guess_strategy="one",
        )

        def cond_fn(carry) -> jax.Array:
            _, ls_state = carry
            return cond_step_ls(ls_state)

        def body_fn(carry):
            loop_key, ls_state = carry
            loop_key, eval_key = jr.split(loop_key)
            ls_state = step_ls(
                ls_state,
                value_and_grad_fn=value_and_grad_fn,
                fn_kwargs={"key": eval_key},
            )
            return (loop_key, ls_state)

        _, final = jax.lax.while_loop(cond_fn, body_fn, (key, init_state))

        learning_rate = final.stepsize
        scaled_updates = optax.tree.scale(learning_rate, updates)
        new_state = optax.ScaleByZoomLinesearchState(
            learning_rate=learning_rate,
            value=final.value,
            grad=final.grad,
            info=optax.ZoomLinesearchInfo(
                num_linesearch_steps=final.count,
                decrease_error=final.decrease_error,
                curvature_error=final.curvature_error,
            ),
        )
        return scaled_updates, optax.tree.cast_like(
            new_state, other_tree=state
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def lbfgs_per_eval_key(
    learning_rate: float | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    max_linesearch_steps: int = 20,
) -> optax.GradientTransformationExtraArgs:
    """``optax.lbfgs`` with the per-eval-key zoom linesearch.

    Parameters
    ----------
    learning_rate : float or None, optional
        Global scaling factor; ``None`` (default) leaves the stepsize
        to the linesearch, matching ``optax.lbfgs``.
    memory_size : int, optional
        Number of past parameter/gradient differences kept to
        approximate the inverse Hessian.
    scale_init_precond : bool, optional
        Whether to scale the identity preconditioner seed.
    max_linesearch_steps : int, optional
        Maximum number of linesearch iterations per outer step.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        The optimizer; its ``update`` requires ``value``, ``grad``,
        ``value_fn`` and ``key`` extra keyword arguments.
    """
    return optax.lbfgs(
        learning_rate=learning_rate,
        memory_size=memory_size,
        scale_init_precond=scale_init_precond,
        linesearch=scale_by_zoom_linesearch_per_eval_key(
            max_linesearch_steps=max_linesearch_steps
        ),
    )


def _stock_lbfgs(
    learning_rate: float | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    max_linesearch_steps: int = 20,
) -> optax.GradientTransformationExtraArgs:
    """``optax.lbfgs`` accepting the same signature as the per-eval
    variant, for the deterministic-task fallback."""
    return optax.lbfgs(
        learning_rate=learning_rate,
        memory_size=memory_size,
        scale_init_precond=scale_init_precond,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            initial_guess_strategy="one",
        ),
    )


def optax_per_eval_key_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    opt_hash: int,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions threading a PRNG key
    through the optimizer's ``update`` instead of freezing it into
    ``value_fn``.

    The counterpart of ``optax_extra_kwargs_fn`` for optimizers built
    around :func:`scale_by_zoom_linesearch_per_eval_key`: the loss
    closure is built *without* a key
    (``Partial(loss_fn, **sample)``) and the step's key is passed as
    the ``key`` extra argument of ``optimizer.update``, where the
    linesearch splits it per evaluation. The same key seeds the
    recompute branch of ``optax.value_and_grad_from_state`` (only
    taken at the first step or after a non-finite cached value).

    Only meaningful for stochastic losses; the caller must guarantee
    ``loss_fn`` accepts a ``key`` keyword argument (``task.pass_rng``
    true).

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optimizer whose ``update`` consumes ``value``, ``grad``,
        ``value_fn`` and ``key`` (e.g. :func:`lbfgs_per_eval_key`).
    loss_fn : LossFunction
        Loss function to optimize; must accept ``key``.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    opt_hash : int
        Hash for optimizer step tracking.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """

    def init_fn(
        params: InputParameters, key: PRNGKeyArray = None, **sample
    ) -> OptaxState:
        """Initialise the per-eval-key Optax optimizer state."""
        params = jax.tree.map(lambda x: x[0], params)
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        """Perform one L-BFGS step under per-evaluation noise keys."""
        input_params, opt_state, key = carry
        new_key, eval_key = jr.split(key)

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params
        )

        model = eqx.combine(input_params, static)
        fn = Partial(loss_fn, **sample)

        loss, grads = optax.value_and_grad_from_state(fn)(
            model, state=opt_state, key=eval_key
        )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            input_params,
            value=loss,
            grad=grads,
            value_fn=fn,
            key=eval_key,
        )

        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=jax.tree.map(lambda x: jnp.expand_dims(x, 0), loss),
            params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), input_params),
            grads=jax.tree.map(lambda x: jnp.expand_dims(x, 0), grads),
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(1, dtype=int),
            iterations=jnp.array(1, dtype=int),
        )

        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        # Expand dimensions
        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def lbfgs_per_eval_key_update(
    task: Task,
    opt_hash: int,
    popsize: int = 1,
    bounded: tuple[float, float] | None = None,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for per-eval-key L-BFGS.

    On stochastic tasks (``task.pass_rng`` true) the optimizer is
    :func:`lbfgs_per_eval_key`, driven by
    :func:`optax_per_eval_key_fn` so every linesearch evaluation
    draws an independent noise realization. On deterministic tasks
    fresh keys per evaluation are a mathematical no-op, so the
    factory delegates to ``UpdateClass.optax_update_extra_kwargs``
    with a stock ``optax.lbfgs`` of the same configuration — the
    resulting trajectories are identical to ``optimizer="lbfgs"``.

    Parameters
    ----------
    task : Task
        Task providing the model and loss function.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int, optional
        Population size; defaults to 1 (gradient-based optimizers
        operate on a single point).
    bounded : tuple[float, float] or None, optional
        Box bounds applied to parameters after each step.
    stop_fn : StopFunction or None, optional
        Stopping function.
    **hyperparameters
        Forwarded to :func:`lbfgs_per_eval_key` (``learning_rate``,
        ``memory_size``, ``scale_init_precond``,
        ``max_linesearch_steps``).

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    if bounded is None:
        bounded = (None, None)

    if not task.pass_rng:
        return UpdateClass.optax_update_extra_kwargs(
            task=task,
            optimizer=_stock_lbfgs,
            opt_hash=opt_hash,
            popsize=popsize,
            bounded=bounded,
            stop_fn=stop_fn,
            **hyperparameters,
        )

    _, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_per_eval_key_fn(
        static=static,
        optimizer=lbfgs_per_eval_key(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


lbfgs_per_eval_key_mapping = {
    "lbfgs_per_eval_key": Partial(lbfgs_per_eval_key_update)
}
