import gc

import equinox as eqx
import jax
import pytest

from l2co import OptimizationStep
from l2co._src.tasks.benchmark_task import create_benchmark_task

# # Define the fixture to clear JAX cache after each test
# @pytest.fixture(autouse=True)
# def clear_jax_cache():
#     # Clear JAX's XLA cache before and after each test
#     yield
#     eqx.clear_caches()
#     jax.clear_caches()
#     gc.collect()
