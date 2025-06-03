from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import jax.tree as jt
import jax.tree_util as jtu
from f3dasm import Block, ExperimentData
from jaxtyping import PyTree

# =============================================================================

SAMPLER_HASH = '0000000000000000'

# =============================================================================


class RandomWeightSampler(Block):
    def __init__(self,
                 lower_bound: float = 0.0,
                 upper_bound: float = 1.0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def call(self, data: ExperimentData, model: PyTree, num_samples: int,
             seed: int,
             **kwargs) -> ExperimentData:

        params, static = eqx.partition(model, eqx.is_inexact_array)
        sample_list_input, sample_list_output = [], []
        key = jrd.key(seed)

        for _ in range(num_samples):
            _, key = jrd.split(key)
            sample = jt.map(lambda x: jrd.uniform(
                key=key, shape=x.shape, minval=self.lower_bound,
                maxval=self.upper_bound), params)

            sample_list_input.append({'x': eqx.combine(sample, static)})
            sample_list_output.append({'opt_step': SAMPLER_HASH})

        experiment_data = ExperimentData(
            domain=data.domain, input_data=sample_list_input,
            output_data=sample_list_output,
            project_dir=data.project_dir)

        return experiment_data.mark_all('open')


class NormalWeightSampler(Block):
    def call(self, data: ExperimentData, model: PyTree,
             num_samples: int, seed: int, **kwargs) -> ExperimentData:
        params, static = eqx.partition(model, eqx.is_inexact_array)
        sample_list = []
        key = jrd.key(seed)
        for _ in range(num_samples):
            _, key = jrd.split(key)
            sample = jt.map(lambda x: jrd.normal(
                key, shape=x.shape), params)

            sample_list.append({'x': eqx.combine(sample, static)})

        return ExperimentData(domain=data.domain, input_data=sample_list,
                              project_dir=data.project_dir)


class GridWeightSampler(Block):
    def __init__(self, stepsize: float,
                 lower_bound: float, upper_bound: float):
        self.stepsize = stepsize
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.num_points = max(
            2, int(round((self.upper_bound - self.lower_bound) / self.stepsize)
                   ) + 1)

    def generate_grid(self, params):
        """Generate a grid of values for each parameter ensuring lower
        & upper bounds are included."""
        grid_params = []
        for param in jtu.tree_leaves(params):
            if param.ndim == 0:  # Scalar parameter
                grid = jnp.linspace(
                    self.lower_bound, self.upper_bound,
                    num=self.num_points)
            else:  # Tensor parameter
                grid = jnp.meshgrid(
                    *[jnp.linspace(
                        self.lower_bound, self.upper_bound,
                        num=self.num_points)] * param.size)
                # Reshape into parameter shape
                grid = jnp.stack(grid, axis=-1).reshape((-1,) + param.shape)
            grid_params.append(grid)
        return jtu.tree_unflatten(jtu.tree_structure(params), grid_params)

    def call(self, data: ExperimentData, model: PyTree,
             **kwargs) -> ExperimentData:
        params, static = eqx.partition(model, eqx.is_inexact_array)

        # Generate the grid of sampled parameter values
        grid_samples = self.generate_grid(params)

        sample_list = [{'x': eqx.combine(sample, static)}
                       for sample in grid_samples]

        return ExperimentData(domain=data.domain, input_data=sample_list,
                              project_dir=data.project_dir)


class XavierNormalWeightSampler:
    def call(self, data: ExperimentData, model: PyTree,
             num_samples: int, seed: int, **kwargs) -> ExperimentData:
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params, static = eqx.partition(model, eqx.is_inexact_array)

        sample_list = []

        key = jrd.key(seed)

        for _ in range(num_samples):
            _, key = jrd.split(key)
            wkey, bkey = jrd.split(key)
            params = init_part(
                params=params,
                filter_fn=lambda m: [x.weight for x in jt.leaves(
                    m, is_leaf=is_linear)
                    if is_linear(x)],
                init_fn=jnn.initializers.xavier_normal(),
                key=wkey)
            params = init_part(
                params=params,
                filter_fn=lambda m: [x.bias for x in jt.leaves(
                    m, is_leaf=is_linear)
                    if is_linear(x)],
                init_fn=jnn.initializers.zeros,
                key=bkey)

            # Append the sample to the sample list
            sample_list.append({'x': eqx.combine(params, static)})

        # Return the ExperimentData with the updated domain and sampled data
        return ExperimentData(domain=data.domain, input_data=sample_list,
                              project_dir=data.project_dir)


def init_part(
    params: PyTree, filter_fn: Callable, init_fn: jnn.initializers.Initializer,
    *, key=jrd.PRNGKey
) -> PyTree:
    w = filter_fn(params)
    new_w = [
        init_fn(subkey, weight.shape, weight.dtype)
        for weight, subkey in zip(w, jrd.split(key, len(w)))
    ]
    return eqx.tree_at(filter_fn, params, new_w)
