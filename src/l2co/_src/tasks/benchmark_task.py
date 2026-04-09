from __future__ import annotations

import functools

# Standard
import bbob_jax
import jax

# Third-party
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from f3dasm import Block, ExperimentData

# Local
from ..task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================


# =============================================================================


class CEC2019Sampler(Block):
    """
    Sampler for CEC 2019 benchmark functions.

    Methods
    -------
    call(data)
        Generate experiment data for CEC 2019 benchmark functions.
    """

    def __init__(self):
        self.dim_mapping = {
            "F12019": 9,
            "F22019": 16,
            "F32019": 18,
            "F42019": 10,
            "F52019": 10,
            "F62019": 10,
            "F72019": 10,
            "F82019": 10,
            "F92019": 10,
            "F102019": 10,
        }

    def call(self, data: ExperimentData) -> ExperimentData:
        """
        Generate experiment data for CEC 2019 benchmark functions.

        Parameters
        ----------
        data : ExperimentData
            Input experiment data.

        Returns
        -------
        ExperimentData
            Experiment data with CEC 2019 benchmark functions.
        """
        fns = data.domain.input_space["fn_name"].categories

        df = pd.DataFrame(
            {
                "dimensionality": [self.dim_mapping[f] for f in fns],
                "fn_name": fns,
            }
        )

        return ExperimentData(domain=data.domain, input_data=df)


def scale_input(domain_bounds: jax.Array, function_bounds: jax.Array):
    """
    Create a decorator to scale input parameters between different bounds.

    Parameters
    ----------
    domain_bounds : jax.Array
        Domain bounds with shape (n_dims, 2), where each row contains
        [lower, upper] for normalization.
    function_bounds : jax.Array
        Target function bounds with shape (n_dims, 2), where each row
        contains [lower, upper] for scaling.

    Returns
    -------
    Callable
        Decorator that scales function inputs from domain_bounds to
        function_bounds.

    Notes
    -----
    Performs two-step transformation: first normalizes input to [0, 1]
    using domain_bounds, then scales to function_bounds.
    """

    def decorator(func):
        def wrapper(x, *args, **kwargs):
            # Extract bounds
            dmn_lower = domain_bounds[:, 0]
            dmn_upper = domain_bounds[:, 1]
            fn_lower = function_bounds[:, 0]
            fn_upper = function_bounds[:, 1]

            # Scale x to [0, 1] based on standard bounds
            normalized_x = (x - dmn_lower) / (dmn_upper - dmn_lower)

            # Scale normalized_x to target bounds
            scaled_x = fn_lower + normalized_x * (fn_upper - fn_lower)

            # Pass transformed input to the original function
            return func(scaled_x, *args, **kwargs)

        return wrapper

    return decorator


def add_noise(noise_level: float):
    """
    Create a decorator to add multiplicative Gaussian noise to function
    output.

    Parameters
    ----------
    noise_level : float
        Standard deviation of the Gaussian noise relative to the
        function value.

    Returns
    -------
    Callable
        Decorator that adds noise to function output using provided
        random key.

    Notes
    -----
    The noise is multiplicative: $y_{noisy} = y + y \\cdot \\epsilon$,
    where $\\epsilon \\sim \\mathcal{N}(0, \\text{noise_level})$.
    Requires a 'key' argument in the wrapped function signature.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(x, key, *args, **kwargs):
            noise = jr.normal(key, shape=()) * noise_level
            y = func(x, *args, **kwargs)
            return y + y * noise

        return wrapper

    return decorator


def create_bbob_task(
    fn_name: str, seed: int, dimensionality: int, noise: float = 0.0
) -> Task:
    """
    Create an optimization task from a BBOB benchmark function.

    Parameters
    ----------
    fn_name : str
        Name of the BBOB benchmark function.
    seed : int
        Random seed for function instance generation.
    dimensionality : int
        Dimensionality of the optimization problem.
    noise : float, optional
        Multiplicative noise level (standard deviation), by default 0.0.

    Returns
    -------
    Task
        Configured optimization task with scaled and optionally noisy
        objective function.

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to BBOB's standard
    [-5, 5]^d range. If noise > 0, the function requires a random
    key and applies multiplicative Gaussian noise.
    """
    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 2)),
        function_bounds=jnp.tile(jnp.array([-5.0, 5.0]), (dimensionality, 2)),
    )

    if noise > 0.0:
        fn_noise = add_noise(noise_level=noise)
        pass_rng = True
    else:
        fn_noise = lambda f: f
        pass_rng = False

    bbob_fn, global_min = bbob_jax.registry[fn_name](
        ndim=dimensionality, key=jr.key(seed)
    )

    loss_fn = fn_noise(fn_scale_input(bbob_fn))
    model = jnp.zeros(dimensionality)
    tag = bbob_jax.function_characteristics[fn_name].copy()
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=model,
        global_min=float(global_min),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )


def create_cec2005_task(
    fn_name: str, seed: int, dimensionality: int, noise: float = 0.0
) -> Task:
    """
    Create an optimization task from a CEC2005 benchmark function.

    Parameters
    ----------
    fn_name : str
        Name of the CEC2005 benchmark function.
    seed : int
        Random seed for function instance generation.
    dimensionality : int
        Dimensionality of the optimization problem.
    noise : float, optional
        Multiplicative noise level (standard deviation), by default 0.0.

    Returns
    -------
    Task
        Configured optimization task with scaled and optionally noisy
        objective function.

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to CEC2005's bounds.
    If noise > 0, the function requires a random
    key and applies multiplicative Gaussian noise.
    """

    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 2)),
        function_bounds=jnp.tile(
            jnp.array(bbob_jax.bounds.cec2005_bounds[fn_name]),
            (dimensionality, 2),
        ),
    )

    if noise > 0.0:
        fn_noise = add_noise(noise_level=noise)
        pass_rng = True
    else:
        fn_noise = lambda f: f
        pass_rng = False

    # These functions have built-in stochasticity.
    if fn_name in ["f4", "f17", "f24", "f25"]:
        pass_rng = True

    bbob_fn, global_min = bbob_jax.cec2005_registry[fn_name](
        ndim=dimensionality, key=jr.key(seed)
    )

    loss_fn = fn_noise(fn_scale_input(bbob_fn))
    model = jnp.zeros(dimensionality)
    tag = bbob_jax.cec2005_function_characteristics[fn_name].copy()
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=model,
        global_min=float(global_min),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )
