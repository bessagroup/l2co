from functools import partial

import jax.random as jr
import pytest
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import Callable
from bbob_jax import registry
from pathlib import Path

from l2co._src.budget import based_on_dimensionality
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout import (
    RunOptimization,
    RolloutWrapper,
)
from l2co_tasks import create_bbob_task
from l2co_tasks import create_pde_task
from l2co.sampling import normal_sampling
from l2co_tasks import Task

from tests import ALL_OPTIMIZERS, OPTIMIZATION_STEPS

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

    rollout_wrapper = RolloutWrapper.init(
        optimizer=optimizer,
        task=task,
        sampler=sampler,
        key=key,
        bounded=bounded,
    )

    _, _, history_state = rollout_wrapper.batch_evaluate(
        key=key,
        n_iterations=n_iterations,
        n_realizations=n_realizations,
    )

    data_xarr = history_state.to_xarray_realizations()


# =============================================================================


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize("optimizer", OPTIMIZATION_STEPS)
@pytest.mark.parametrize("fn_name", ["sphere", "rastrigin"])
@pytest.mark.parametrize("dim", [2, 5])
@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_bbob(
    optimizer: OptimizationStep, fn_name: str, dim: int, noise: float
):
    task = create_bbob_task(
        fn_name=fn_name,
        seed=0,
        dimensionality=dim,
        noise=noise,
    )
    run(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
        include_input_mean=False,
    )


def test_optimization_step_name_rl2co_with_extension():
    step = OptimizationStep(
        "rl2co",
        hyperparameters={
            "model": "/some/long/path/my_model.eqx",
            "n_steps": 10,
            "sequence_length": 50,
        },
    )
    assert step.name == "rl2co_my_model"


def test_optimization_step_name_rl2co_no_extension():
    step = OptimizationStep(
        "rl2co", hyperparameters={"model": "/path/to/model"}
    )
    assert step.name == "rl2co_model"


def test_optimization_step_name_rl2co_no_model():
    step = OptimizationStep("rl2co", hyperparameters={})
    assert step.name == "rl2co"


def test_optimization_step_name_non_rl2co_unchanged():
    step = OptimizationStep("adam", hyperparameters={"learning_rate": 0.01})
    assert step.name == "adam_learning_rate=0.01"


@pytest.mark.slow
@pytest.mark.parametrize("optimizer", ALL_OPTIMIZERS)
def test_all_optimizers(optimizer: str):
    task = create_bbob_task(
        fn_name="rastrigin",
        seed=0,
        dimensionality=2,
    )
    run(
        optimizer=OptimizationStep(optimizer=optimizer),
        task=task,
        sampler=normal_sampling,
        include_input_mean=False,
    )


def test_random_search_one_shot_shapes_and_best():
    """Random search must emit one row per iteration and sum fevals correctly.

    Random search bypasses the scan loop and evaluates all
    ``n_iterations * popsize`` candidates in a single batched call, so the
    standard ``HistoryState`` contract has to be reconstructed by reshape.
    This test pins the externally observable shape/semantics:

    * ``output_min`` has one entry per logical iteration;
    * ``fevals`` sums to ``n_iterations * popsize``;
    * the reported ``best_loss`` is the global best across all candidates.
    """
    n_iterations = 7
    popsize = 4
    task = create_bbob_task(
        fn_name="sphere",
        seed=0,
        dimensionality=3,
    )
    optimizer = OptimizationStep(
        optimizer="randomsearch",
        hyperparameters={"sampler": "normal", "popsize": popsize},
    )

    key = jr.key(0)
    rollout_wrapper = RolloutWrapper.init(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
        key=key,
        bounded=(None, None),
    )
    run_state, _, history_state = rollout_wrapper.batch_evaluate(
        key=key,
        n_iterations=n_iterations,
        n_realizations=1,
    )

    assert history_state.output_min.shape == (1, n_iterations)
    assert int(history_state.fevals.sum()) == n_iterations * popsize
    # Global best must be at least as good as any per-iteration min.
    assert (
        float(run_state.best_loss[0])
        <= float(history_state.output_min.min()) + 1e-6
    )
