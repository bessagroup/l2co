"""
The ``UpdateClass`` adapter and its optimizer-specific constructors.

``UpdateClass`` is the static, JIT-friendly container that the rollout
machinery (``init_``, ``step``, ``run_``, ``batch_run_``) consumes; the
classmethods on this class are the factories that build one from a
``Task`` plus an optax / evosax optimizer specification. The closure
factories at module scope (``optax_fn``, ``optax_extra_kwargs_fn``,
``evosax_distribution_based_fn``, ``evosax_population_based_fn``) build
the ``(init_fn, step_fn)`` pairs that the classmethods then wrap.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from evosax.algorithms.base import Params
from evosax.algorithms.base import State as EvoSaxState
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree
from l2co_tasks import Task
from optax import OptState as OptaxState

# Local
from l2co._src.opt_history import OptHistory
from l2co._src.optimizers.menu_eval import menu_loss_and_grad
from l2co._src.optimizers.utils import resolve_popsize, variable_popsize
from l2co._src.sampler import random_sampling
from l2co._src.tree_utils import (
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
    vmapped_loss_with_rng,
)
from l2co._src.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    PopSize,
    StepFunction,
    StopFunction,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                       Optax closure factories
# =============================================================================


def optax_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for Optax optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """

    def init_fn(
        params: InputParameters, key: PRNGKeyArray | None = None, **sample
    ) -> OptaxState:
        """Initialise the Optax optimizer state."""
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        """Perform one gradient-based Optax step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss, grads = vmapped_loss_and_grad_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, 1),
            )

        else:
            loss, grads = vmapped_loss_and_grad(
                eqx.combine(input_params, static), loss_fn, sample
            )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            input_params,
            value=loss,
        )
        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(1, dtype=int),
            iterations=jnp.array(1, dtype=int),
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def step_fevals(opt_state: OptaxState, new_opt_state: OptaxState) -> jax.Array:
    """Function evaluations one optax step actually spent.

    Optimizers that search along a direction before committing to a
    step -- L-BFGS and anything else built on
    ``optax.scale_by_zoom_linesearch`` -- evaluate the objective once
    per linesearch trial, not once per iteration. Billing them a flat
    ``1`` under-reports their cost, and most severely *after* they
    converge: once the gradient bottoms out at a small but non-zero
    residual, the Wolfe curvature criterion (``|slope_step|`` against
    ``0.9 * |slope_init|``, with ``tol=0.0``) can no longer be
    satisfied exactly, so the search exhausts ``max_linesearch_steps``
    and fails out on most subsequent iterations. That tail is
    landscape-dependent -- an objective that converges to an *exactly*
    zero gradient, like a textbook quadratic, keeps costing one trial
    per step -- but on realistic problems it dominates the run. See
    ``docs/adr/0010``.

    The count comes from the linesearch's own bookkeeping
    (``ZoomLinesearchInfo.num_linesearch_steps``, reset per outer
    update), which is exactly the number of objective evaluations:
    optax's linesearch ``step_fn`` dispatches to *either* the interval
    search *or* the zoom, each of which evaluates once and increments
    the count once. Optimizers with no linesearch carry no such field
    and keep billing ``1``.

    The extra evaluation is the cache miss in
    ``optax.value_and_grad_from_state``, which recomputes only when
    the value cached by the previous step is non-finite -- in practice
    the first step alone. This mirrors optax's own branch condition.
    It is billed *logically*: under ``eqx.filter_vmap`` the underlying
    ``lax.cond`` degrades to a ``select`` and both branches execute, so
    a batched rollout really pays for it on every step, but billing the
    logical cost keeps single and batched runs reporting the same
    number.

    Parameters
    ----------
    opt_state : OptaxState
        Optimizer state *before* the update, holding the value cached
        by the previous step.
    new_opt_state : OptaxState
        Optimizer state returned by ``optimizer.update``, holding this
        step's linesearch info.

    Returns
    -------
    jax.Array
        Scalar integer count for ``OptHistory.fevals``.
    """
    n_linesearch = optax.tree.get(new_opt_state, "num_linesearch_steps")

    # Absent for every optimizer without a linesearch (adam, sgd, ...);
    # ``optax.tree.get`` returns a Python ``None`` at trace time, so
    # this branch is static.
    if n_linesearch is None:
        return jnp.array(1, dtype=int)

    cached_value = optax.tree.get(opt_state, "value")
    if cached_value is None:
        return jnp.asarray(n_linesearch, dtype=int)

    recompute = jnp.isinf(cached_value) | jnp.isnan(cached_value)
    return jnp.asarray(n_linesearch, dtype=int) + recompute.astype(int)


def optax_extra_kwargs_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for Optax optimizers with extra
    ``update`` keyword arguments (``value``, ``grad``, ``value_fn``).

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer instance supporting extra arguments.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
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
        """Initialise the extra-kwargs Optax optimizer state."""
        params = jax.tree.map(lambda x: x[0], params)
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        """Perform one extra-kwargs Optax step (e.g. L-BFGS)."""
        input_params, opt_state, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params
        )

        model = eqx.combine(input_params, static)
        if pass_rng:
            fn = Partial(loss_fn, key=key, **sample)
        else:
            fn = Partial(loss_fn, **sample)

        loss, grads = optax.value_and_grad_from_state(fn)(
            model, state=opt_state
        )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            input_params,
            value=loss,
            grad=grads,
            value_fn=fn,
        )

        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=jax.tree.map(lambda x: jnp.expand_dims(x, 0), loss),
            params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), input_params),
            grads=jax.tree.map(lambda x: jnp.expand_dims(x, 0), grads),
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=step_fevals(opt_state, new_opt_state),
            iterations=jnp.array(1, dtype=int),
        )

        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        # Expand dimensions
        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


#                                                      EvoSax closure factories
# =============================================================================


def evosax_distribution_based_fn(
    static: PyTree,
    optimizer: DistributionBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    popsize: int,
    es_params: Params,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for EvoSax distribution-based
    optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : DistributionBasedAlgorithm
        EvoSax distribution-based optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    popsize : int
        Population size.
    es_params : Params
        Evolution strategy parameters.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """

    def init_fn(
        params: InputParameters, key: PRNGKeyArray = None, **sample
    ) -> EvoSaxState:
        """Initialise the distribution-based ES state."""
        params = jax.tree.map(lambda x: x[0], params)
        return optimizer.init(key, params, es_params)

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        """Perform one distribution-based ES step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )

        else:
            loss = vmapped_loss(
                eqx.combine(input_params, static), loss_fn, sample
            )

        new_opt_state, _ = optimizer.tell(
            key=key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=key, state=new_opt_state, params=es_params
        )

        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(popsize, dtype=int),
            grads=grads,
            iterations=jnp.array(1, dtype=int),
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def evosax_population_based_fn(
    static: PyTree,
    optimizer: PopulationBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float, float],
    popsize: int,
    es_params: Params,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for EvoSax population-based
    optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : PopulationBasedAlgorithm
        EvoSax population-based optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float, float]
        Parameter bounds for clipping.
    popsize : int
        Population size.
    es_params : Params
        Evolution strategy parameters.
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
        params: InputParameters, key: PRNGKeyArray, **sample
    ) -> EvoSaxState:
        """Initialise the population-based ES state."""
        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )
        else:
            loss = vmapped_loss(eqx.combine(params, static), loss_fn, sample)

        return optimizer.init(
            key=key, population=params, fitness=loss, params=es_params
        )

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        """Perform one population-based ES step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )
        else:
            loss = vmapped_loss(
                eqx.combine(input_params, static), loss_fn, sample
            )

        new_opt_state, _ = optimizer.tell(
            key=key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=key, state=new_opt_state, params=es_params
        )

        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(popsize, dtype=int),
            grads=grads,
            iterations=jnp.array(1, dtype=int),
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


#                                                                   UpdateClass
# =============================================================================


def l2co_population_based_fn(
    strategy,
    static: PyTree,
    loss_fn: LossFunction,
    bounded: tuple[float | None, float | None],
    pass_rng: bool,
) -> tuple[InitFunction, StepFunction]:
    """Build ``(init_fn, step_fn)`` driving the L2CO strategy wrapper.

    The analogue of :func:`evosax_population_based_fn` for
    :func:`l2co._src.strategy_wrapper.l2co_population_strategy`
    (mirroring rl2co's ``rl2co_population_based_fn``). Three
    differences from the stock evosax factory:

    * **Ask-eval-tell rhythm.** The wrapper proposes via ``ask``
      before it can ``tell`` (the stage-1 fire and the handshake
      happen inside ``ask``), so each ``step_fn`` asks, evaluates the
      proposed population, then tells. The stock factory instead
      evaluates the carried population, tells, then asks.
    * **Evaluation is dispatched to the active optimizer's width.**
      The menu mixes gradient-based and population-based
      sub-optimizers, and the wrapper's ``tell`` consumes ``grads=``
      but reads only ``grads[0]``. Rather than evaluating the full
      generation width with gradients every generation, the evaluation
      goes through :func:`~l2co._src.optimizers.menu_eval.
      menu_loss_and_grad`, which evaluates ``popsize`` rows value-only
      for a population generation and the single iterate with
      gradients for a gradient one. That helper also owns the NaN
      contract (padding rows and diverged candidates alike), so the
      sentinel is robust regardless of how ``loss_fn`` treats NaN
      inputs. See ``docs/adr/0011-per-optimizer-evaluation-width-on-
      the-strategy-bridge.md``.
    * **The ``update_step`` hash is dynamic.** Before the stage-1
      fire every entry carries the stage-0 optimizer's hash, after a
      switching fire the stage-1 optimizer's — read from the wrapper
      state after each ``ask`` — so the stored history records which
      optimizer produced each entry (one combined history instead of
      the old path's two concatenated per-stage histories).

    ``init_fn`` does **not** evaluate the initial population: it
    forwards an all-``NaN`` fitness to ``strategy.init``, matching
    the stock machinery (where the initial samples are only evaluated
    by ``reset``'s hidden bookkeeping pass, outside any history) and
    the old eager path's init point (``samples[0]``).

    Parameters
    ----------
    strategy : l2co._src.strategy_wrapper._L2COPopulationStrategy
        The wrapper returned by ``l2co_population_strategy``. Closed
        over; its stage-1 weights become compile-time constants under
        JIT. (Typed loosely to keep this module importable without
        the strategy wrapper, which is imported instead by
        :meth:`l2co._src.meta_optimizer.L2COUpdateClass.
        l2co_population_update`, the sole caller.)
    static : PyTree
        Non-inexact-array partition of ``task.model``; recombined
        with each population via :func:`equinox.combine` before
        evaluation.
    loss_fn : LossFunction
        Task loss function (``task.loss_fn``).
    bounded : tuple[float | None, float | None]
        Box bounds clipped onto every asked population before
        evaluation; ``(None, None)`` disables clipping.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword
        (``task.pass_rng``).

    Returns
    -------
    tuple[InitFunction, StepFunction]
        ``(init_fn, step_fn)`` ready to slot into an
        :class:`UpdateClass`.
    """
    popsize = strategy.population_size
    default_params = strategy.default_params
    name_hashes = strategy.name_hashes_arr
    popsizes = jnp.array(
        [sub.popsize for sub in strategy.sub_optimizers], dtype=int
    )
    evaluate = menu_loss_and_grad(
        sub_optimizers=strategy.sub_optimizers,
        population_size=popsize,
        static=static,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
    )

    def init_fn(
        params: InputParameters,
        key: PRNGKeyArray,
        **sample,
    ) -> PyTree:
        """Build the wrapper state from the initial population.

        No evaluation happens here (see the factory docstring): the
        fitness forwarded to ``strategy.init`` is all-``NaN``, so
        best-tracking seeds to ``+inf`` and the sub-optimizers
        initialise around ``params[0]``.

        Parameters
        ----------
        params : InputParameters
            Initial population, leading axis
            ``strategy.population_size``.
        key : PRNGKeyArray
            Key for sub-optimizer initialisation.
        **sample
            Current data batch forwarded by
            :func:`l2co._src.update_class.init_`; unused.

        Returns
        -------
        PyTree
            The wrapper state, used as the opt_state in the scan
            carry.
        """
        fitness = jnp.full((popsize,), jnp.nan, dtype=float)
        return strategy.init(key, params, fitness, default_params)

    def step_fn(
        carry: tuple[InputParameters, PyTree, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, PyTree, PRNGKeyArray], OptHistory]:
        """Run one ask-eval-tell generation of the L2CO wrapper.

        Parameters
        ----------
        carry : tuple[InputParameters, PyTree, PRNGKeyArray]
            ``(params, wrapper_state, key)``. ``params`` is the
            scan's carried population; the asked population is
            returned in its slot so ``history.params`` stays
            meaningful.
        sample : dict
            Current data batch.

        Returns
        -------
        tuple[tuple[InputParameters, PyTree, PRNGKeyArray], OptHistory]
            ``((population, new_state, new_key), history)`` with
            ``history.update_step`` carrying the dynamic hash of the
            sub-optimizer active this generation.
        """
        _params, state, key = carry
        new_key, ask_key, eval_key, tell_key = jr.split(key, 4)

        # Ask: fires stage-1 at the switch generation and emits either
        # the best_one handshake (on a switch) or the active
        # sub-optimizer's own population.
        population, state = strategy.ask(ask_key, state, default_params)
        population = jax.tree.map(lambda p: jnp.clip(p, *bounded), population)

        # The sub-optimizer active *this* generation; drives both the
        # dynamic history hash and the feval count.
        active_idx = state.active_opt_idx

        # Eval at the active sub-optimizer's evaluation width: a
        # population generation gets ``popsize`` rows value-only, a
        # gradient one gets the single iterate with gradients. Padding
        # rows and diverged candidates come back NaN.
        eval_keys = jr.split(eval_key, popsize)
        loss, grads = evaluate(active_idx, population, sample, eval_keys)

        def value_fn(candidate_params: PyTree) -> jax.Array:
            """Task loss of one candidate params partition.

            Lets line-search grad sub-optimizers (``optax.lbfgs``)
            probe trial points; reuses candidate 0's key for
            stochastic losses so the line search is consistent with
            the ``value``/``grad`` passed alongside it.
            """
            candidate_model = eqx.combine(candidate_params, static)
            if pass_rng:
                return loss_fn(candidate_model, key=eval_keys[0], **sample)
            return loss_fn(candidate_model, **sample)

        state, _ = strategy.tell(
            tell_key,
            population,
            loss,
            state,
            default_params,
            grads=grads,
            value_fn=value_fn,
        )

        history = OptHistory(
            loss=loss,
            params=population,
            grads=grads,
            update_step=name_hashes[active_idx].astype(int),
            fevals=popsizes[active_idx],
            iterations=jnp.array(1, dtype=int),
        )
        return (population, state, new_key), history

    return init_fn, step_fn


# =============================================================================


def _default_stop_fn(*_, **__) -> bool:
    return False


class UpdateClass(eqx.Module):
    """Static container wrapping one optimizer step.

    Encapsulates the initialization function, step function, population
    size, optimizer hash, and stopping criterion for a single optimizer.
    Constructed via one of the ``optax_update`` / ``evosax_*_update``
    classmethods (or a subclass like ``RandomSearchUpdateClass``); the
    instance is then handed to the rollout helpers in
    ``l2co._src.update_class``.

    Attributes
    ----------
    init_fn : InitFunction
        Function to initialize optimizer state.
    step_fn : StepFunction
        Function to perform a single optimization step.
    popsize : int
        Population size for the optimizer (1 for gradient-based).
    hash : int
        Stable hash identifier for the optimizer configuration.
    stop_fn : StopFunction
        Function to check stopping criteria. Defaults to a function
        that never stops.
    max_fevals_per_step : int or None
        Most function evaluations this optimizer can bill in one
        iteration, or ``None`` (default) when that is just ``popsize``.
        Only linesearch optimizers set it: L-BFGS bills one evaluation
        per linesearch trial, so it can spend far more than its
        population of one (see :func:`step_fevals` and
        ``docs/adr/0010``). Read it through :attr:`fevals_bound`, which
        resolves the ``None``. Consumers that size a budget or
        normalize a per-evaluation cost must use that bound rather than
        ``popsize`` -- notably rl2co's ``AUCEnv``, whose reward
        normalization assumes no optimizer bills more than
        ``max_iterations * max(popsize)``.
    sequential_realizations : bool
        Class-level marker (not a field): whether a multi-realization
        run must execute one realization at a time rather than with all
        realizations vectorized together. ``False`` for every plain
        optimizer, which is the fast choice. A subclass sets it to
        ``True`` when its ``step_fn`` branches on a value that differs
        per realization -- the meta-optimizers, whose step picks a
        sub-optimizer from a menu. Read by
        :func:`l2co._src.run_state.batch_run`, which is where the
        consequences are spelled out.
    """

    #: See the class docstring. A ``ClassVar`` annotation keeps this out
    #: of the dataclass fields, so subclasses override it with a plain
    #: assignment and no constructor signature changes.
    sequential_realizations: ClassVar[bool] = False

    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)
    stop_fn: StopFunction = eqx.field(static=True, default=_default_stop_fn)
    max_fevals_per_step: int | None = eqx.field(static=True, default=None)

    @property
    def fevals_bound(self) -> int:
        """Upper bound on ``OptHistory.fevals`` for one iteration.

        Returns
        -------
        int
            :attr:`max_fevals_per_step` when set, else :attr:`popsize`.
        """
        if self.max_fevals_per_step is None:
            return self.popsize
        return self.max_fevals_per_step

    @classmethod
    def optax_update(
        cls,
        task: Task,
        optimizer: Callable[..., optax.GradientTransformation],
        opt_hash: int,
        popsize: int = 1,
        bounded: tuple[float, float] | None = None,
        stop_fn: StopFunction | None = None,
        **hyperparameters,
    ) -> UpdateClass:
        """Construct an ``UpdateClass`` for a standard optax optimizer.

        Parameters
        ----------
        task : Task
            Task providing the model and loss function.
        optimizer : Callable[..., optax.GradientTransformation]
            Optax optimizer constructor (e.g. ``optax.adam``).
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
            Forwarded to the optimizer constructor.

        Returns
        -------
        UpdateClass
            Configured optimizer wrapper.
        """
        _, static = eqx.partition(task.model, eqx.is_inexact_array)

        init_fn, step_fn = optax_fn(
            static=static,
            optimizer=optimizer(**hyperparameters),
            loss_fn=task.loss_fn,
            bounded=bounded,
            opt_hash=opt_hash,
            pass_rng=task.pass_rng,
        )

        return cls(
            init_fn=init_fn,
            step_fn=step_fn,
            popsize=popsize,
            hash=opt_hash,
            stop_fn=stop_fn,
        )

    @classmethod
    def optax_update_extra_kwargs(
        cls,
        task: Task,
        optimizer: Callable[..., optax.GradientTransformationExtraArgs],
        opt_hash: int,
        popsize: int = 1,
        bounded: tuple[float, float] | None = None,
        stop_fn: StopFunction | None = None,
        max_fevals_per_step: int | None = None,
        **hyperparameters,
    ) -> UpdateClass:
        """Construct an ``UpdateClass`` for optax optimizers with extra
        ``update`` kwargs (``value``, ``grad``, ``value_fn`` -- e.g.
        L-BFGS).

        Parameters
        ----------
        task : Task
            Task providing the model and loss function.
        optimizer : Callable[..., optax.GradientTransformationExtraArgs]
            Optax optimizer constructor supporting extra update kwargs.
        opt_hash : int
            Stable hash stamped into ``OptHistory.update_step``.
        popsize : int, optional
            Population size; defaults to 1.
        bounded : tuple[float, float] or None, optional
            Box bounds applied to parameters after each step.
        stop_fn : StopFunction or None, optional
            Stopping function.
        max_fevals_per_step : int or None, optional
            Forwarded to :attr:`UpdateClass.max_fevals_per_step`. Set
            it for linesearch optimizers, whose per-iteration cost is
            not ``popsize``; ``None`` (default) leaves the bound at
            ``popsize``.
        **hyperparameters
            Forwarded to the optimizer constructor.

        Returns
        -------
        UpdateClass
            Configured optimizer wrapper.
        """
        _, static = eqx.partition(task.model, eqx.is_inexact_array)

        init_fn, step_fn = optax_extra_kwargs_fn(
            static=static,
            optimizer=optimizer(**hyperparameters),
            loss_fn=task.loss_fn,
            bounded=bounded,
            opt_hash=opt_hash,
            pass_rng=task.pass_rng,
        )

        return cls(
            init_fn=init_fn,
            step_fn=step_fn,
            popsize=popsize,
            hash=opt_hash,
            stop_fn=stop_fn,
            max_fevals_per_step=max_fevals_per_step,
        )

    @classmethod
    def evosax_distribution_update(
        cls,
        task: Task,
        optimizer: type[DistributionBasedAlgorithm],
        opt_hash: int,
        popsize: PopSize = variable_popsize,
        bounded: tuple[float, float] | None = (None, None),
        stop_fn: StopFunction | None = None,
        **hyperparameters,
    ) -> UpdateClass:
        """Construct an ``UpdateClass`` for an EvoSax distribution-based
        algorithm (CMA-ES, OpenAI-ES, ...).

        Parameters
        ----------
        task : Task
            Task providing the model and loss function.
        optimizer : type[DistributionBasedAlgorithm]
            EvoSax algorithm class.
        opt_hash : int
            Stable hash stamped into ``OptHistory.update_step``.
        popsize : int or Callable[[Task], int], optional
            Population size; either an integer or a callable taking the
            ``Task`` and returning an integer. Defaults to
            :func:`variable_popsize` (``int(4 + 3 * log(d))``).
        bounded : tuple[float, float] or None, optional
            Box bounds applied to parameters after each step.
        stop_fn : StopFunction or None, optional
            Stopping function.
        **hyperparameters
            Override fields on ``optimizer.default_params``.

        Returns
        -------
        UpdateClass
            Configured optimizer wrapper.
        """
        popsize = resolve_popsize(popsize, task)
        params, static = eqx.partition(task.model, eqx.is_inexact_array)

        optimizer = optimizer(population_size=popsize, solution=params)

        init_fn, step_fn = evosax_distribution_based_fn(
            static=static,
            optimizer=optimizer,
            loss_fn=task.loss_fn,
            bounded=bounded,
            popsize=popsize,
            es_params=optimizer.default_params.replace(**hyperparameters),
            opt_hash=opt_hash,
            pass_rng=task.pass_rng,
        )

        return cls(
            init_fn=init_fn,
            step_fn=step_fn,
            popsize=popsize,
            hash=opt_hash,
            stop_fn=stop_fn,
        )

    @classmethod
    def evosax_random_search_update(
        cls,
        task: Task,
        optimizer: type[DistributionBasedAlgorithm],
        opt_hash: int,
        popsize: PopSize = variable_popsize,
        bounded: tuple[float, float] | None = (None, None),
        stop_fn: StopFunction | None = None,
        **hyperparameters,
    ) -> UpdateClass:
        """Construct an ``UpdateClass`` for EvoSax's ``RandomSearch``
        algorithm.

        Distinct from :meth:`evosax_distribution_update` because
        ``RandomSearch`` requires an explicit ``sampling_fn`` argument
        on construction.

        Parameters
        ----------
        task : Task
            Task providing the model and loss function.
        optimizer : type[DistributionBasedAlgorithm]
            EvoSax ``RandomSearch`` algorithm class.
        opt_hash : int
            Stable hash stamped into ``OptHistory.update_step``.
        popsize : int or Callable[[Task], int], optional
            Population size; either an integer or a callable taking the
            ``Task`` and returning an integer. Defaults to
            :func:`variable_popsize` (``int(4 + 3 * log(d))``).
        bounded : tuple[float, float] or None, optional
            Box bounds applied to parameters after each step.
        stop_fn : StopFunction or None, optional
            Stopping function.
        **hyperparameters
            Override fields on ``optimizer.default_params``.

        Returns
        -------
        UpdateClass
            Configured optimizer wrapper.
        """
        popsize = resolve_popsize(popsize, task)
        params, static = eqx.partition(task.model, eqx.is_inexact_array)

        sampling_fn = partial(
            random_sampling,
            params=params,
            n_samples=1,
        )

        optimizer = optimizer(
            population_size=popsize,
            solution=params,
            sampling_fn=sampling_fn,
        )

        init_fn, step_fn = evosax_distribution_based_fn(
            static=static,
            optimizer=optimizer,
            loss_fn=task.loss_fn,
            bounded=bounded,
            popsize=popsize,
            es_params=optimizer.default_params.replace(**hyperparameters),
            opt_hash=opt_hash,
            pass_rng=task.pass_rng,
        )

        return cls(
            init_fn=init_fn,
            step_fn=step_fn,
            popsize=popsize,
            hash=opt_hash,
            stop_fn=stop_fn,
        )

    @classmethod
    def evosax_population_update(
        cls,
        task: Task,
        optimizer: type[PopulationBasedAlgorithm],
        opt_hash: int,
        popsize: PopSize = variable_popsize,
        bounded: tuple[float, float] | None = (None, None),
        stop_fn: StopFunction | None = None,
        **hyperparameters,
    ) -> UpdateClass:
        """Construct an ``UpdateClass`` for an EvoSax population-based
        algorithm (DE, PSO, ...).

        Parameters
        ----------
        task : Task
            Task providing the model and loss function.
        optimizer : type[PopulationBasedAlgorithm]
            EvoSax algorithm class.
        opt_hash : int
            Stable hash stamped into ``OptHistory.update_step``.
        popsize : int or Callable[[Task], int], optional
            Population size; either an integer or a callable taking the
            ``Task`` and returning an integer. Defaults to
            :func:`variable_popsize` (``int(4 + 3 * log(d))``).
        bounded : tuple[float, float] or None, optional
            Box bounds applied to parameters after each step.
        stop_fn : StopFunction or None, optional
            Stopping function.
        **hyperparameters
            Override fields on ``optimizer.default_params``.

        Returns
        -------
        UpdateClass
            Configured optimizer wrapper.
        """
        popsize = resolve_popsize(popsize, task)
        params, static = eqx.partition(task.model, eqx.is_inexact_array)

        optimizer = optimizer(population_size=popsize, solution=params)

        init_fn, step_fn = evosax_population_based_fn(
            static=static,
            optimizer=optimizer,
            loss_fn=task.loss_fn,
            bounded=bounded,
            popsize=popsize,
            es_params=optimizer.default_params.replace(**hyperparameters),
            opt_hash=opt_hash,
            pass_rng=task.pass_rng,
        )

        return cls(
            init_fn=init_fn,
            step_fn=step_fn,
            popsize=popsize,
            hash=opt_hash,
            stop_fn=stop_fn,
        )
