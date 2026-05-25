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
            fevals=jnp.array(1, dtype=int),
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
    """

    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)
    stop_fn: StopFunction = eqx.field(static=True, default=_default_stop_fn)

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
