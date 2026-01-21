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
        # global_min=float(global_min),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )
