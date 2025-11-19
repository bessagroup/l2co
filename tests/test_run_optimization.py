from functools import partial

import jax.random as jr
import pytest
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import Callable
from bbob_jax import registry
from pathlib import Path

from l2co._src.budget import based_on_dimensionality
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.run_optimization import (
    RunOptimization,
    initialize_optimizer,
    run_optimization_many_realizations,
)
from l2co._src.tasks.benchmark_task import create_bbob_task
from l2co._src.tasks.pde import create_pde_task
from l2co.sampling import normal_sampling
from l2co import Task

from . import OPTIMIZATION_STEPS

# =============================================================================


def run(
    optimizer: OptimizationStep,
    task: Task,
    sampler: Callable,
    n_iterations: int = 5,
    n_realizations: int = 2,
    key: PRNGKeyArray | None = None,
    bounded: tuple[float | None, float | None] = (None, None),
    include_input_mean: bool = True,
):
    if key is None:
        key = jr.key(0)
    update_state, update_class = initialize_optimizer(
        optimizer=optimizer, task=task, bounded=bounded, key=key
    )

    history_state = run_optimization_many_realizations(
        update_state=update_state,
        optimizer=update_class,
        sampler=sampler,
        task=task,
        n_iterations=5,
        n_realizations=2,
        key=key,
    )

    data_xarr = history_state.to_xarray_realizations(
        include_input_mean=include_input_mean
    )


# =============================================================================


@pytest.mark.parametrize("optimizer", OPTIMIZATION_STEPS)
def test_pde(optimizer: OptimizationStep, tmp_path: Path):
    task = create_pde_task(
        pde_task_name="convection",
        seed=0,
        dataset_path=str(tmp_path),
        x_range=(0.0, 2.0 * jnp.pi),
        t_range=(0.0, 1.0),
        beta=40,
        xgrid_resolution=255,
        tgrid_resolution=101,
        num_ic_points=257,
        num_bc_points=101,
        num_res_points=10000,
        hidden_size=32,
    )

    run(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
        include_input_mean=False,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZATION_STEPS)
@pytest.mark.parametrize("fn_name", ["sphere", "rastrigin"])
@pytest.mark.parametrize("dim", [2, 5])
def test_bbob(optimizer: OptimizationStep, fn_name: str, dim: int):
    task = create_bbob_task(
        fn_name=fn_name,
        seed=0,
        dimensionality=dim,
    )
    run(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
    )
