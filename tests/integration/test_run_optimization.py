from functools import partial

import jax.random as jr
import pytest
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import Callable
from bbob_jax import registry
from pathlib import Path

import equinox as eqx
import jax

from l2co._src.budget import based_on_dimensionality
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.rollout import (
    RunOptimization,
    RolloutWrapper,
)
from l2co._src.run_state import batch_reset
from l2co._src.update_class import (
    batch_run_randomsearch,
    run_randomsearch,
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
@pytest.mark.parametrize(
    "optimizer", ALL_OPTIMIZERS, ids=lambda step: step.optimizer
)
def test_all_optimizers(optimizer: OptimizationStep):
    task = create_bbob_task(
        fn_name="rastrigin",
        seed=0,
        dimensionality=2,
    )
    run(
        optimizer=optimizer,
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


def test_random_search_lax_map_matches_vmap():
    """``batch_run_randomsearch`` (sequential ``lax.map``) must match vmap.

    The runner was changed from ``eqx.filter_vmap`` (which materialises
    every realization's candidate set at once) to a sequential
    ``jax.lax.map`` to bound peak memory to a single realization. Because
    each realization consumes its own pre-split ``key``, the two execution
    strategies must produce the same per-realization outputs (up to
    float32 reduction-ordering noise, which XLA does not pin across the
    two lowerings). This test pins that equivalence by running the same
    inputs through the shipped ``lax.map`` implementation and an inline
    vmap reference, and asserting every returned array leaf matches.
    """
    n_iterations = 6
    popsize = 4
    n_realizations = 3
    key = jr.key(0)

    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=5)
    optimizer = OptimizationStep(
        optimizer="randomsearch",
        hyperparameters={"sampler": "normal", "popsize": popsize},
    )
    rw = RolloutWrapper.init(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
        key=key,
        bounded=(None, None),
    )

    # Per-realization keys + realization-axis run state, exactly as the
    # public ``batch_evaluate`` path builds them.
    keys = jr.split(key, n_realizations)
    rs = batch_reset(
        rw.run_state,
        rw.batch_state,
        rw.static,
        rw.dataset,
        rw.loss_fn,
        rw.pass_rng,
        rw.sampler,
        keys,
    )

    # Shipped implementation: sequential lax.map over realizations.
    out_map = batch_run_randomsearch(
        update_class=rs.update_class,
        opt_state=rs.opt_state,
        params=rs.params,
        batch_state=rw.batch_state,
        dataset=rw.dataset,
        key=keys,
        n_iterations=n_iterations,
        verbose=False,
    )

    # Reference: the previous eqx.filter_vmap strategy, inlined. Mirrors
    # the batch_state broadcast the shipped runner performs internally.
    bs = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_realizations, *x.shape)),
        rw.batch_state,
    )

    def _single(opt_state, params, batch_state, key):
        return run_randomsearch(
            update_class=rs.update_class,
            opt_state=opt_state,
            params=params,
            batch_state=batch_state,
            dataset=rw.dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=False,
        )

    out_vmap = eqx.filter_vmap(_single, in_axes=(0, 0, 0, 0))(
        rs.opt_state, rs.params, bs, keys
    )

    map_leaves = jax.tree.leaves(eqx.filter(out_map, eqx.is_array))
    vmap_leaves = jax.tree.leaves(eqx.filter(out_vmap, eqx.is_array))
    assert len(map_leaves) == len(vmap_leaves)
    for got, ref in zip(map_leaves, vmap_leaves):
        assert got.shape == ref.shape
        # Every output must carry the leading realization axis.
        assert got.shape[0] == n_realizations
        assert jnp.allclose(got, ref, equal_nan=True)


def test_random_search_chunked_semantics(monkeypatch):
    """Chunked random search stays self-consistent across chunk splits.

    ``run_randomsearch`` processes the candidate set in chunks of
    ``_rs_chunking`` iterations to bound peak memory. Forcing a tiny chunk
    target splits a small budget into several padded chunks, exercising the
    scan, the padding mask, and the cross-chunk best/final reductions.
    ``n_iterations=7`` is deliberately not a multiple of the chunk size, so
    the last chunk carries a padded iteration that must be masked out.
    """
    import l2co._src.update_class as uc_mod

    # 8 candidates/chunk with popsize 3 -> chunk_iters=2, n_chunks=4 for
    # n_iterations=7 (padded to 8; last iteration is padding).
    monkeypatch.setattr(uc_mod, "_RS_CHUNK_TARGET_CANDIDATES", 8)
    assert uc_mod._rs_chunking(7, 3) == (4, 2)

    n_iterations = 7
    popsize = 3
    n_realizations = 2
    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=4)
    optimizer = OptimizationStep(
        optimizer="randomsearch",
        hyperparameters={"sampler": "normal", "popsize": popsize},
    )
    key = jr.key(0)
    rw = RolloutWrapper.init(
        optimizer=optimizer,
        task=task,
        sampler=normal_sampling,
        key=key,
        bounded=(None, None),
    )
    run_state, _, history_state = rw.batch_evaluate(
        key=key, n_iterations=n_iterations, n_realizations=n_realizations
    )

    # One row per logical iteration (padding dropped), fevals unaffected.
    assert history_state.output_min.shape == (n_realizations, n_iterations)
    assert int(history_state.fevals.sum()) == (
        n_realizations * n_iterations * popsize
    )
    # No padded (+inf) candidate leaked into the reported best, and the
    # global best equals the smallest per-iteration minimum.
    assert bool(jnp.all(jnp.isfinite(run_state.best_loss)))
    assert bool(jnp.all(jnp.isfinite(history_state.output_min)))
    for r in range(n_realizations):
        assert float(run_state.best_loss[r]) == pytest.approx(
            float(history_state.output_min[r].min()), abs=1e-5
        )
