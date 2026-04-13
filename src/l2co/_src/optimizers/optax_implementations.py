"""
Module for optax based optimizers.
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

# Local
from l2co._src.opt_history import OptHistory
from l2co._src.optimizers.utils import normalize_key
from l2co._src.tree_utils import (
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
)
from l2co._src.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
)
from l2co._src.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#                                                                         Optax
# =============================================================================


def optax_fn(
    static: PyTree,
    optimizer: optax.GradientTransformation,
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
    optimizer : optax.GradientTransformation
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
            grads, opt_state, input_params
        )
        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=opt_hash,
            fevals=1,
            iterations=1,
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def optax_extra_args_fn(
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
    arguments.

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
        """Initialise the extra-args Optax optimizer state."""
        params = jax.tree.map(lambda x: x[0], params)
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
        *args,
        **kwargs,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        """Perform one extra-args Optax step (e.g. L-BFGS)."""
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
            update_step=opt_hash,
            fevals=1,
            iterations=1,
        )

        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        # Expand dimensions
        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def optax_update(
    task: Task,
    optimizer: optax.GradientTransformation,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for Optax optimizers.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : optax.GradientTransformation
        Optax optimizer constructor.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
    stop_fn : StopFunction or None, optional
        Stopping function.
    **hyperparameters
        Additional optimizer hyperparameters.

    Returns
    -------
    UpdateClass
        UpdateClass instance for optimization.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


def optax_update_extra_args(
    task: Task,
    optimizer: optax.GradientTransformationExtraArgs,
    opt_hash: int,
    popsize: int,
    bounded: tuple[float, float] | None = None,
    stop_fn: StopFunction | None = None,
    **hyperparameters,
) -> UpdateClass:
    """
    Creates an UpdateClass for Optax optimizers with extra arguments.

    Parameters
    ----------
    task : Task
        Task object containing model and loss function.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer constructor supporting extra arguments.
    opt_hash : int
        Hash for optimizer step tracking.
    popsize : int
        Population size.
    bounded : tuple[float, float] or None, optional
        Parameter bounds for clipping.
    stop_fn : StopFunction or None, optional
        Stopping function.
    **hyperparameters
        Additional optimizer hyperparameters.

    Returns
    -------
    UpdateClass
        UpdateClass instance for optimization.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_extra_args_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
    )


# =============================================================================


def learning_rate_optimizer(
    task: Task,
    optax_optimizer: optax.GradientTransformation,
    learning_rate: float | None = None,
    **kwargs,
) -> UpdateClass:
    """Create an Optax UpdateClass with a configurable learning rate.

    Automatically selects ``optax_update`` or
    ``optax_update_extra_args`` depending on the optimizer type.

    Parameters
    ----------
    task : Task
        Task containing the model and loss function.
    optax_optimizer : optax.GradientTransformation
        Optax optimizer constructor.
    learning_rate : float or None, optional
        Learning rate. Defaults to ``1e-3`` when ``None``.
    **kwargs
        Forwarded to the underlying update function.

    Returns
    -------
    UpdateClass
        Configured optimizer update class.
    """
    if optax_optimizer in normalized_optax_normal.values():
        update_fn = optax_update
    elif optax_optimizer in normalized_optax_from_state.values():
        update_fn = optax_update_extra_args
    else:
        raise ValueError(
            f"Optimizer not recognized as a valid Optax optimizer \
                {optax_optimizer}."
        )

    learning_rate = learning_rate or 1e-3

    optimizer = Partial(update_fn, optimizer=optax_optimizer, popsize=1)

    return optimizer(**kwargs, task=task, learning_rate=learning_rate)


# =============================================================================

opt_names_normal = [
    "adabelief",
    "adadelta",
    "adan",
    "adafactor",
    "adagrad",
    "adam",
    "adamw",
    "adamax",
    "adamaxw",
    "amsgrad",
    "fromage",
    "lamb",
    "lars",
    "lion",
    "nadam",
    "nadamw",
    "noisy_sgd",
    "novograd",
    "optimistic_gradient_descent",
    "optimistic_adam",
    # "polyak_sgd", # function call signature is different, ignored!
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
    "sign_sgd",
    # "signum", # not available in this version of optax!
    "sm3",
    "yogi",
]

opt_names_from_state = [
    "lbfgs",
]

normalized_optax_normal = {
    normalize_key(name): getattr(optax, name) for name in opt_names_normal
}

normalized_optax_from_state = {
    normalize_key(name): getattr(optax, name) for name in opt_names_from_state
}

optax_mapping = {
    name: Partial(
        learning_rate_optimizer,
        optax_optimizer=cls,
    )
    for name, cls in (
        normalized_optax_normal | normalized_optax_from_state
    ).items()
}
