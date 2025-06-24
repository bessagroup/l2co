import gc

import equinox as eqx
import jax
import pytest

from l2co import OptimizationSchedule, OptimizationStep
from l2co._src.tasks.benchmark_task import create_benchmark_task


# Define the fixture to clear JAX cache after each test
@pytest.fixture(autouse=True)
def clear_jax_cache():
    # Clear JAX's XLA cache before and after each test
    yield
    eqx.clear_caches()
    jax.clear_caches()
    gc.collect()


# Define the OPTIMIZATION_SCHEDULES globally for use in tests
@pytest.fixture
def optimization_schedules():
    return [
        OptimizationSchedule(steps=[OptimizationStep(
            optimizer='cmaes', hyperparameters={'popsize': 30})], budgets=1.0),
        OptimizationSchedule(steps=[OptimizationStep(
            optimizer='adam', hyperparameters={'learning_rate': 1e-3})], budgets=1.0),
        OptimizationSchedule(
            steps=[OptimizationStep(optimizer='lbfgsb')], budgets=1.0)
    ]


@pytest.fixture(params=[
    {'offset': True, 'scale_output': True, 'noise': 0.1,
        'dimensionality': 5, 'grad_noise': 0.1},
    {'offset': False, 'scale_output': False, 'noise': 0.2,
        'dimensionality': 10, 'grad_noise': 0.05},
    {'offset': True, 'scale_output': False, 'noise': 0.05,
        'dimensionality': 3, 'grad_noise': 0.2}
], ids=lambda p: f"offset_{p['offset']}_scale_{p['scale_output']}_noise_{p['noise']}_dim_{p['dimensionality']}_grad_{p['grad_noise']}")
def benchmark_task(request):
    params = request.param
    return create_benchmark_task(
        fn_name='ackley',
        seed=42,
        noise=params['noise'],
        dimensionality=params['dimensionality'],
        offset=params['offset'],
        scale_output=params['scale_output'],
        grad_noise=params['grad_noise']
    )
