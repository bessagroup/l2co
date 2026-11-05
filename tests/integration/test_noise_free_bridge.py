"""A noisy task's noise-free loss reaches the run (``docs/adr/0024``).

A task with a stochastic loss carries its noise-free loss in
``Task.loss_fn_true``. The optimizer must keep seeing the noisy value,
but the run records, per step, the lowest noise-free loss of the
evaluated population (``HistoryState.output_min_true``), and noisy
tasks are scored on that. l2co is the bridge between the task and the
optimizer, so it attaches the noise-free loss to the ``UpdateClass``
(:func:`l2co.optimization.noise_free_loss`). It does so where dataset
generation builds the optimizer, ``RolloutWrapper.init``; a direct
``init_run_state`` call opts in with ``record_noise_free=True``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import l2co_optimizers as lo
import numpy as np
import pytest
from l2co_tasks import Task

import l2co
from l2co.naming import OUTPUT_MIN, OUTPUT_MIN_TRUE

#: The toy "noise" is a constant offset (the key is ignored), so the
#: noise-free minimum of any evaluated population is exactly the observed
#: one minus the offset. That makes the recorded series checkable.
OFFSET = 1.0

N_ITER = 5
N_REAL = 2


def _true_loss(model):
    return jnp.sum((model - 2.0) ** 2)


def _offset_loss(model, key):
    del key
    return _true_loss(model) + OFFSET


def _uniform_noise_loss(model, key):
    return _true_loss(model) + jr.uniform(key)


def _noisy_task(loss_fn=_offset_loss) -> Task:
    return Task(
        model=jnp.full(3, 0.5),
        loss_fn=loss_fn,
        pass_rng=True,
        loss_fn_true=_true_loss,
        global_min=0.0,
        tag={"task_name": "pull_to_two", "noise": True},
    )


def _clean_task() -> Task:
    return Task(
        model=jnp.full(3, 0.5),
        loss_fn=_true_loss,
        global_min=0.0,
        tag={"task_name": "pull_to_two", "noise": False},
    )


_STEPS = [
    lo.OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 0.05}
    ),
    lo.OptimizationStep(optimizer="cmaes"),
    lo.OptimizationStep(optimizer="randomsearch"),
]
_IDS = [s.optimizer for s in _STEPS]


def _rollout(step, task, key=0):
    wrapper = l2co.RolloutWrapper.init(
        optimizer=step,
        task=task,
        sampler=lo.normal_sampling,
        key=jr.key(key),
    )
    _, _, history = wrapper.batch_evaluate(
        key=jr.key(key + 1), n_iterations=N_ITER, n_realizations=N_REAL
    )
    return wrapper, history


def _by_hand(step, task, key=0):
    """The pre-bridge ``RolloutWrapper.init`` body: no noise-free loss.

    Wrapped in a ``RolloutWrapper`` so it runs through the same jitted
    ``batch_evaluate``: compiled and eager runs may differ in the last
    float32 digit, which is not what these tests are about.
    """
    update_class = l2co.optimizer_mapping(step.optimizer)(
        **step.hyperparameters,
        task=task,
        opt_hash=step.hash,
        bounded=l2co.task_bounds(task),
        stop_fn=step.stopping_fn,
    )
    run_state = lo.RunState.init(
        update_class,
        model=task.model,
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=jr.key(key),
    )
    batch_state = lo.BatchState.init(
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=jr.key(key),
    )
    _, static = eqx.partition(task.model, eqx.is_inexact_array)
    wrapper = l2co.RolloutWrapper(
        run_state=run_state,
        batch_state=batch_state,
        static=static,
        dataset=task.loaded_dataset,
        loss_fn=task.loss_fn,
        sampler=lo.normal_sampling,
        pass_rng=task.pass_rng,
        verbose=False,
    )
    _, _, history = wrapper.batch_evaluate(
        key=jr.key(key + 1), n_iterations=N_ITER, n_realizations=N_REAL
    )
    return history


# ------------------------------------------------------------ RolloutWrapper


@pytest.mark.parametrize("step", _STEPS, ids=_IDS)
def test_rollout_records_the_noise_free_minimum_of_a_noisy_task(step):
    wrapper, history = _rollout(step, _noisy_task())

    assert wrapper.run_state.update_class.true_loss_fn is not None
    assert history.output_min_true is not None
    assert history.output_min_true.shape == (N_REAL, N_ITER)
    assert jnp.isfinite(history.output_min_true).all()
    np.testing.assert_allclose(
        history.output_min_true,
        history.output_min - OFFSET,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("step", _STEPS, ids=_IDS)
def test_noise_free_series_lies_within_the_noise_of_the_observed_one(step):
    """With real noise in ``[0, 1)``: ``0 <= observed - true < 1``.

    The observed minimum over a population is at least the noise-free
    minimum and at most the noisy value of the noise-free best member.
    """
    _, history = _rollout(step, _noisy_task(_uniform_noise_loss))

    gap = np.asarray(history.output_min - history.output_min_true)
    assert (gap >= -1e-6).all()
    assert (gap < 1.0 + 1e-6).all()
    assert (gap > 1e-6).any()  # the noise is really there


@pytest.mark.parametrize("step", _STEPS, ids=_IDS)
def test_attaching_does_not_change_what_the_optimizer_sees(step):
    """The observed series of a noisy task is the same with or without
    the noise-free loss attached: the optimizer never reads it."""
    _, history = _rollout(step, _noisy_task(_uniform_noise_loss))
    by_hand = _by_hand(step, _noisy_task(_uniform_noise_loss))

    assert by_hand.output_min_true is None
    np.testing.assert_array_equal(history.output_min, by_hand.output_min)


@pytest.mark.parametrize("step", _STEPS, ids=_IDS)
def test_rollout_of_a_noise_free_task_is_unchanged(step):
    wrapper, history = _rollout(step, _clean_task())
    by_hand = _by_hand(step, _clean_task())

    assert wrapper.run_state.update_class.true_loss_fn is None
    assert history.output_min_true is None
    assert jax.tree.structure(history) == jax.tree.structure(by_hand)
    for a, b in zip(
        jax.tree.leaves(history), jax.tree.leaves(by_hand), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_rollout_export_carries_the_noise_free_series():
    _, history = _rollout(_STEPS[1], _noisy_task())
    ds = l2co.history_to_xarray_realizations(history)

    assert OUTPUT_MIN_TRUE in ds
    assert ds[OUTPUT_MIN_TRUE].dims == ds[OUTPUT_MIN].dims
    np.testing.assert_allclose(
        ds[OUTPUT_MIN_TRUE].values,
        ds[OUTPUT_MIN].values - OFFSET,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("name", ["slsqp", "trustconstr"])
def test_whole_run_scipy_entry_refuses_a_noisy_task(name):
    """The whole-run scipy entries cannot record the series and say so."""
    wrapper = l2co.RolloutWrapper.init(
        optimizer=lo.OptimizationStep(optimizer=name),
        task=_noisy_task(),
        sampler=lo.normal_sampling,
        key=jr.key(0),
    )
    with pytest.raises(NotImplementedError, match="noise-free"):
        wrapper.batch_evaluate(key=jr.key(1), n_iterations=3)


# ------------------------------------------------------------ init_run_state


def test_init_run_state_does_not_attach_by_default():
    """rl2co's environment builds run states here and appends their
    histories; a noise-free series would change its history's structure,
    so attaching is opt-in."""
    run_state = l2co.init_run_state(
        optimizer=_STEPS[0], task=_noisy_task(), key=jr.key(0)
    )
    assert run_state.update_class.true_loss_fn is None


def test_init_run_state_attaches_on_request():
    task = _noisy_task()
    run_state = l2co.init_run_state(
        optimizer=_STEPS[1], task=task, key=jr.key(0), record_noise_free=True
    )
    true_loss_fn = run_state.update_class.true_loss_fn
    assert true_loss_fn is not None
    member = jax.tree.map(lambda p: p[0], run_state.params)
    assert float(true_loss_fn(member)) == pytest.approx(
        float(_true_loss(task.model))
    )


def test_init_run_state_on_request_leaves_a_noise_free_task_alone():
    run_state = l2co.init_run_state(
        optimizer=_STEPS[1],
        task=_clean_task(),
        key=jr.key(0),
        record_noise_free=True,
    )
    assert run_state.update_class.true_loss_fn is None


# ------------------------------------------------------------ noise_free_loss


def test_noise_free_loss_is_none_for_a_noise_free_task():
    assert l2co.noise_free_loss(_clean_task()) is None


def test_noise_free_loss_rebuilds_the_model_from_one_members_params():
    """A model with a static (non-array) part: the closure combines the
    member's parameters with it before calling ``loss_fn_true``."""

    def true_loss(model):
        return model["scale"] * jnp.sum(model["w"] ** 2)

    def noisy_loss(model, key):
        return true_loss(model) + jr.normal(key)

    model = {"w": jnp.array([1.0, 2.0]), "scale": 3}
    task = Task(
        model=model,
        loss_fn=noisy_loss,
        pass_rng=True,
        loss_fn_true=true_loss,
        tag={"task_name": "scaled"},
    )
    params = eqx.filter(model, eqx.is_inexact_array)

    fn = l2co.noise_free_loss(task)
    assert float(fn(params)) == pytest.approx(15.0)


def _noisy_task_with_data() -> Task:
    return Task(
        model=jnp.zeros(2),
        loss_fn=_offset_loss,
        pass_rng=True,
        loss_fn_true=_true_loss,
        dataset={"dataset_path": "data.npz", "batch_size": 4, "seed": 0},
        tag={"task_name": "with_data"},
    )


def test_noise_free_loss_refuses_a_task_with_a_dataset():
    """``true_loss_fn`` gets no data batch, so a noise-free loss that
    needs one cannot be recorded yet; raise rather than call it wrong."""
    with pytest.raises(NotImplementedError, match="dataset"):
        l2co.noise_free_loss(_noisy_task_with_data())


def test_rollout_wrapper_refuses_a_noisy_task_with_a_dataset():
    """Dataset generation builds through ``RolloutWrapper.init``, which
    raises before anything runs (or the dataset is loaded)."""
    with pytest.raises(NotImplementedError, match="dataset"):
        l2co.RolloutWrapper.init(
            optimizer=_STEPS[0],
            task=_noisy_task_with_data(),
            sampler=lo.normal_sampling,
            key=jr.key(0),
        )


def test_noise_free_loss_is_nan_for_a_member_with_nan_params():
    """The meta-optimizer's step pads its unused population rows with
    NaN parameters. They must not count even under a ``loss_fn_true``
    that maps NaN to a finite value: the rule the observed loss gets
    from ``row_has_nan`` in the menu evaluator."""

    def masking_true_loss(model):
        return jnp.sum(jnp.nan_to_num(model, nan=0.0) ** 2)

    task = Task(
        model=jnp.ones(2),
        loss_fn=_offset_loss,
        pass_rng=True,
        loss_fn_true=masking_true_loss,
        tag={"task_name": "masking"},
    )
    fn = l2co.noise_free_loss(task)

    assert jnp.isnan(fn(jnp.array([jnp.nan, 1.0])))
    assert jnp.isnan(jax.jit(fn)(jnp.array([1.0, jnp.nan])))
    assert float(fn(jnp.array([1.0, 2.0]))) == pytest.approx(5.0)
