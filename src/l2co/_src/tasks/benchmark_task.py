from __future__ import annotations

# Standard
from functools import partial

import bbob_jax
import jax

# Third-party
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opfunu
import pandas as pd
from f3dasm import Block, ExperimentData
from opfunu import create_benchmark_loss_function

# Local
from ..task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Planning"
# =============================================================================

opfunu.np = np

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


def create_benchmark_task(
    fn_name: str,
    seed: int,
    noise: float,
    dimensionality: int,
    offset: bool = True,
    scale_output: bool = True,
    grad_noise: float = 0.0,
) -> Task:
    """
    Create a benchmark task for optimization.

    Parameters
    ----------
    fn_name : str
        Name of the benchmark function.
    seed : int
        Random seed for reproducibility.
    noise : float
        Noise level for the benchmark function.
    dimensionality : int
        Dimensionality of the input space.
    offset : bool, optional
        Whether to apply an offset to the function, by default True.
    scale_output : bool, optional
        Whether to scale the output, by default True.
    grad_noise : float, optional
        Gradient noise level, by default 0.0.

    Returns
    -------
    Task
        The created benchmark task.
    """
    pass_rng = True if noise > 0.0 else False

    model, loss_fn, tag = create_benchmark_loss_function(
        fn_name=str(fn_name),
        seed=int(seed),
        scale_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 1)),
        dimensionality=int(dimensionality),
        noise=float(noise),
        offset=offset,
        scale_output=scale_output,
        grad_noise=float(noise),
    )

    return Task(model=model, pass_rng=pass_rng, loss_fn=loss_fn, tag=tag)


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


def create_bbob_task(fn_name: str, seed: int, dimensionality: int) -> Task:
    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 2)),
        function_bounds=jnp.tile(jnp.array([-5.0, 5.0]), (dimensionality, 2)),
    )
    loss_fn = fn_scale_input(
        partial(bbob_jax.registry[fn_name], key=jr.key(seed))
    )
    model = jnp.zeros(dimensionality)
    tag = bbob_jax.function_characteristics[fn_name].copy()
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed

    return Task(model=model, pass_rng=False, loss_fn=loss_fn, tag=tag)
