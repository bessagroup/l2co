from functools import partial

import jax.random as jr
import pytest
from bbob_jax import registry

from l2co._src.budget import based_on_dimensionality
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.run_optimization import (
    RunOptimization,
    initialize_optimizer,
    run_optimization_many_realizations,
)
from l2co._src.tasks.benchmark_task import create_bbob_task
from l2co.sampling import normal_sampling

from . import OPTIMIZATION_STEPS

# =============================================================================


@pytest.mark.parametrize("step", OPTIMIZATION_STEPS)
@pytest.mark.parametrize("fn_name", ["sphere", "rastrigin"])
@pytest.mark.parametrize("dim", [2, 5])
def test_optimization_schedules(
    step: OptimizationStep, fn_name: str, dim: int
):
    task = create_bbob_task(
        fn_name=fn_name,
        seed=0,
        dimensionality=dim,
    )
    key = jr.key(0)
    bounded = (None, None)
    update_state, update_class = initialize_optimizer(
        optimizer=step, task=task, bounded=bounded, key=key
    )

    history_state = run_optimization_many_realizations(
        update_state=update_state,
        optimizer=update_class,
        sampler=normal_sampling,
        task=task,
        n_iterations=5,
        n_realizations=2,
        key=key,
    )

    data_xarr = history_state.to_xarray_realizations()
