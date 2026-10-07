"""Bounds come from the task (``docs/adr/0020``).

``task_bounds`` is the only way bounds leave a task; the two entry
points that build optimizers (``init_run_state``, ``RolloutWrapper.init``)
take them from it and, during the transition, refuse any bounds a caller
still passes. A path with no way to enforce constraints refuses a
constrained task outright.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import l2co_optimizers as lo
import pytest
from l2co_tasks import Box, Equality, Inequality, Task

import l2co


def _pull_to_two(model):
    """Minimum at 2.0 on every parameter, outside the unit box."""
    return jnp.sum((model - 2.0) ** 2)


def _task(constraints=()):
    return Task(
        model=jnp.full(3, 0.5),
        loss_fn=_pull_to_two,
        global_min=0.0,
        tag={"task_name": "pull_to_two"},
        constraints=constraints,
    )


# ---------------------------------------------------------------- task_bounds


def test_unconstrained_task_has_no_bounds():
    assert l2co.task_bounds(_task()) == (None, None)


def test_uniform_box_becomes_the_pair():
    assert l2co.task_bounds(_task([Box(0.0, 1.0)])) == (0.0, 1.0)


def test_open_side_becomes_none():
    assert l2co.task_bounds(_task([Box(-jnp.inf, 1.0)])) == (None, 1.0)


def test_non_uniform_box_is_refused():
    box = Box(jnp.array([0.0, 0.0, -1.0]), 1.0)
    with pytest.raises(ValueError, match="differs between parameters"):
        l2co.task_bounds(_task([box]))


@pytest.mark.parametrize(
    "constraint",
    [
        Inequality(lambda m: m[0] - 1.0, name="g"),
        Equality(lambda m: jnp.sum(m) - 1.5, name="h"),
    ],
    ids=["inequality", "equality"],
)
def test_inequality_and_equality_are_refused(constraint):
    with pytest.raises(ValueError, match="only a Box"):
        l2co.task_bounds(_task([constraint, Box(0.0, 1.0)]))


# ------------------------------------------------------- refuse_constraints


def test_refuse_constraints_passes_an_unconstrained_task():
    l2co.refuse_constraints(_task(), "here")


def test_refuse_constraints_refuses_even_a_box():
    with pytest.raises(ValueError, match="here has no way"):
        l2co.refuse_constraints(_task([Box(0.0, 1.0)]), "here")


def test_public_strategy_refuses_a_constrained_task():
    # Refused before the model is touched, so no trained model is needed.
    with pytest.raises(ValueError, match="l2co_population_strategy"):
        l2co.l2co_population_strategy(
            model=None,
            task=_task([Box(0.0, 1.0)]),
            sampler=lo.random_sampling,
            t=0.5,
            max_iterations=10,
        )


# ------------------------------------------------ entry points, transition

_ADAM = lo.OptimizationStep(
    optimizer="adam", hyperparameters={"learning_rate": 0.1}
)


@pytest.mark.parametrize("bounded", [None, (None, None), [None, None]])
def test_init_run_state_accepts_no_bounds(bounded):
    l2co.init_run_state(_ADAM, _task(), bounded, jr.key(0))


def test_init_run_state_refuses_caller_bounds():
    with pytest.raises(ValueError, match="bounds now come from the task"):
        l2co.init_run_state(_ADAM, _task(), (0.0, 1.0), jr.key(0))


def test_rollout_wrapper_refuses_caller_bounds():
    with pytest.raises(ValueError, match="RolloutWrapper.init"):
        l2co.RolloutWrapper.init(
            optimizer=_ADAM,
            task=_task(),
            sampler=lo.random_sampling,
            key=jr.key(0),
            bounded=(0.0, 1.0),
        )


def _final_params(task):
    run_state = l2co.init_run_state(_ADAM, task, None, jr.key(0))
    batch_state = lo.BatchState.init(
        dataset=task.loaded_dataset, batch_size=task.batch_size, key=jr.key(0)
    )
    run_state, _, _ = lo.batch_evaluate(
        run_state=run_state,
        batch_state=batch_state,
        static=eqx.filter(task.model, eqx.is_inexact_array, inverse=True),
        dataset=task.loaded_dataset,
        loss_fn=task.loss_fn,
        sampler=lo.random_sampling,
        n_iterations=40,
        pass_rng=task.pass_rng,
        key=jr.split(jr.key(1), 2),
        verbose=False,
    )
    return run_state.params


def test_the_task_box_reaches_the_optimizer():
    free = _final_params(_task())
    boxed = _final_params(_task([Box(0.0, 1.0)]))
    assert float(jnp.max(free)) > 1.0  # the loss pulls outside the box
    assert float(jnp.max(boxed)) <= 1.0  # the task's box holds it in
    assert float(jnp.min(boxed)) >= 0.0
