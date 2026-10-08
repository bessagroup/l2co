"""The Task-level bridge (``docs/adr/0018``) on real ``l2co_tasks.Task``s.

``l2co-optimizers`` takes ``model`` / ``loss_fn`` / ``pass_rng``, never a
task, so the question "does a real ``Task`` reach every bare factory"
is answered here, where the bridge lives. Moved from l2co-optimizers'
``test_l2co_tasks_integration.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import l2co_optimizers as lo
import l2co_tasks
import pytest

import l2co

# Hyperparameters a factory cannot default (optax constructors take a
# required learning rate); everything else builds from its defaults.
_REQUIRED = {"learning_rate": 1e-2}
_NEEDS_BOUNDS = {"shade", "turbo", "rbftrustregion"}


@pytest.fixture(scope="module", params=[False, True], ids=["det", "noisy"])
def task(request):
    if request.param:
        return l2co_tasks.create_bbob_noisy_task(
            fn_name="bbob_noisy_f101", seed=0, dimensionality=4
        )
    return l2co_tasks.create_bbob_task(
        fn_name="sphere", seed=0, dimensionality=4
    )


def _run_steps(update_class, model, n_steps: int):
    """A bare ``init_fn`` then ``step_fn`` loop; one history per step."""
    key = jr.key(0)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: jnp.repeat(x[None], update_class.popsize, axis=0), params
    )
    carry = (params, update_class.init_fn(params, key), key)
    histories = []
    for _ in range(n_steps):
        carry, history = update_class.step_fn(carry, sample={})
        histories.append(history)
    return histories


def _run_whole(update_class, task, n_iterations: int):
    """The run path ``RolloutWrapper`` drives: ``RunState`` then ``run``."""
    run_state = lo.RunState.init(
        update_class,
        model=task.model,
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=jr.key(0),
    )
    batch_state = lo.BatchState.init(
        dataset=task.loaded_dataset, batch_size=task.batch_size, key=jr.key(0)
    )
    run_state, _, _ = lo.run(
        run_state=run_state,
        batch_state=batch_state,
        dataset=task.loaded_dataset,
        n_iterations=n_iterations,
        key=jr.key(1),
        verbose=False,
    )
    return run_state


def test_count_parameters_matches_task_dimensionality(task):
    assert lo.count_parameters(task.model) == task.dimensionality


@pytest.mark.parametrize("name", sorted(lo.optimizers))
def test_every_bare_factory_builds_and_steps_through_the_bridge(name, task):
    factory = l2co.optimizer_mapping(name)
    kwargs = dict(task=task, opt_hash=1)
    if name in _NEEDS_BOUNDS:
        kwargs.update(bounded=(-5.0, 5.0), x_min=-5.0, x_max=5.0)
    try:
        update_class = factory(**kwargs)
    except TypeError:
        update_class = factory(**kwargs, **_REQUIRED)
    assert isinstance(update_class, lo.UpdateClass)
    if hasattr(update_class, "sampling_fn"):
        return
    try:
        histories = _run_steps(update_class, task.model, n_steps=2)
    except NotImplementedError:
        # A whole-run entry -- the scipy minimisers and IPOPT
        # (l2co-optimizers ADRs 0002, 0003) -- has no single step: it runs
        # its whole budget inside one call, one realization at a time.
        # Drive that instead.
        assert update_class.sequential_realizations
        run_state = _run_whole(update_class, task, n_iterations=5)
        assert jnp.isfinite(run_state.best_loss).all()
        return
    assert jnp.isfinite(histories[-1].loss).any()


def test_init_run_state_batch_evaluates_a_real_task(task):
    """The path ``RolloutWrapper`` drives: resolve, init, evaluate."""
    step = lo.OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 0.05}
    )
    run_state = l2co.init_run_state(optimizer=step, task=task, key=jr.key(0))
    batch_state = lo.BatchState.init(
        dataset=task.loaded_dataset, batch_size=task.batch_size, key=jr.key(0)
    )
    run_state, _, history = lo.batch_evaluate(
        run_state=run_state,
        batch_state=batch_state,
        static=eqx.filter(task.model, eqx.is_inexact_array, inverse=True),
        dataset=task.loaded_dataset,
        loss_fn=task.loss_fn,
        sampler=lo.normal_sampling,
        n_iterations=3,
        pass_rng=task.pass_rng,
        key=jr.split(jr.key(1), 2),
        verbose=False,
    )
    assert run_state.best_loss.shape == (2,)
    assert jnp.isfinite(run_state.best_loss).all()
    assert history.output_min.shape == (2, 3)


def test_init_run_state_matches_the_bare_path(task):
    """The bridge only unpacks the task: same ``RunState`` as by hand."""
    step = lo.OptimizationStep(optimizer="cmaes")
    via_bridge = l2co.init_run_state(optimizer=step, task=task, key=jr.key(0))
    update_class = lo.optimizer_mapping("cmaes")(
        model=task.model,
        loss_fn=task.loss_fn,
        pass_rng=task.pass_rng,
        opt_hash=step.hash,
        bounded=(None, None),
        stop_fn=step.stopping_fn,
    )
    by_hand = lo.RunState.init(
        update_class,
        model=task.model,
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=jr.key(0),
    )
    assert via_bridge.update_class.popsize == by_hand.update_class.popsize
    assert jnp.array_equal(via_bridge.params, by_hand.params)
    for a, b in zip(
        jax.tree.leaves(via_bridge.opt_state),
        jax.tree.leaves(by_hand.opt_state),
        strict=True,
    ):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b), equal_nan=True)
