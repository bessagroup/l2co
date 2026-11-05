"""The budget functions of ``l2co.budget`` (ADR 0026).

``evaluations_based_on_dimensionality`` gives every run the same number
of function evaluations, B = min(evaluations_per_dimension * d,
max_evaluations), and returns it as iterations, floor(B / e), where e is
what one iteration of the given optimizer costs. The optimizers are
built by the real factories, so the tests also pin which attributes
decide e: ``popsize`` for a fixed cost per step, and a set
``max_fevals_per_step`` for the two whose cost per step varies.
"""

from functools import partial

import pytest
from l2co_optimizers import OptimizationStep
from l2co_tasks import create_bbob_task, create_embedded_bbob_task

import l2co
from l2co.budget import (
    based_on_dimensionality,
    evaluation_budget,
    evaluations_based_on_dimensionality,
    fixed_budget,
)

# =============================================================================


def _built(name: str, task, **hyperparameters):
    """The ``UpdateClass`` dataset generation would build for ``name``."""
    return l2co.optimizer_mapping(name)(
        task=task,
        opt_hash=0,
        bounded=(None, None),
        stop_fn=None,
        **hyperparameters,
    )


@pytest.fixture(scope="module")
def task_10d():
    return create_bbob_task(fn_name="sphere", seed=0, dimensionality=10)


@pytest.fixture(scope="module")
def task_1024d():
    return create_bbob_task(fn_name="sphere", seed=0, dimensionality=1024)


#                                          the evaluation budget, B
# =============================================================================


def test_evaluation_budget_is_1000_per_dimension_by_default(task_10d):
    assert evaluation_budget(task_10d) == 10_000


def test_evaluation_budget_is_capped_at_100_000(task_1024d):
    # 1000 * 1024 > 10^5.
    assert evaluation_budget(task_1024d) == 100_000


def test_evaluation_budget_settings_reach_the_formula(task_10d):
    assert (
        evaluation_budget(
            task_10d, evaluations_per_dimension=7, max_evaluations=55
        )
        == 55
    )
    assert (
        evaluation_budget(
            task_10d, evaluations_per_dimension=7, max_evaluations=1_000
        )
        == 70
    )


def test_evaluation_budget_uses_the_ambient_dimension():
    task = create_embedded_bbob_task(
        fn_name="sphere", seed=0, intrinsic_dim=2, ambient_dim=64
    )
    assert evaluation_budget(task) == 64_000


def test_evaluation_budget_needs_no_optimizer(task_10d):
    # B is the same for every optimizer; only the iterations differ.
    assert type(evaluation_budget(task_10d)) is int


def test_iterations_are_the_evaluation_budget_over_e(task_10d):
    # The two functions agree on B for every setting.
    cmaes = _built("cmaes", task_10d)
    for per_dimension, cap in [(1000, 100_000), (7, 55), (3, 1_000)]:
        b = evaluation_budget(
            task_10d,
            evaluations_per_dimension=per_dimension,
            max_evaluations=cap,
        )
        assert (
            evaluations_based_on_dimensionality(
                task=task_10d,
                optimizer=cmaes,
                evaluations_per_dimension=per_dimension,
                max_evaluations=cap,
            )
            == b // cmaes.popsize
        )


#                                          evaluations per iteration, e
# =============================================================================


def test_population_method_gets_b_over_lambda_iterations(task_10d):
    cmaes = _built("cmaes", task_10d)
    assert cmaes.popsize == 10
    assert cmaes.max_fevals_per_step is None

    n_iterations = evaluations_based_on_dimensionality(
        task=task_10d, optimizer=cmaes
    )

    # B = 1000 * 10 = 10^4 evaluations, at lambda = 10 per iteration.
    assert n_iterations == 1_000
    assert n_iterations * cmaes.popsize <= 10_000


def test_single_point_gradient_method_gets_b_iterations(task_10d):
    adam = _built("adam", task_10d, learning_rate=1e-2)
    assert adam.popsize == 1

    assert (
        evaluations_based_on_dimensionality(task=task_10d, optimizer=adam)
        == 10_000
    )


@pytest.mark.parametrize("name", ["lbfgs", "neldermead"])
def test_variable_cost_methods_get_b_iterations(name, task_10d):
    # L-BFGS spends 1 to 21 evaluations per step, Nelder-Mead 2 to n + 3.
    # Both are budgeted at one evaluation per iteration; the evaluations
    # past B are cut in post-processing, not here.
    update_class = _built(name, task_10d)
    assert update_class.max_fevals_per_step is not None

    assert (
        evaluations_based_on_dimensionality(
            task=task_10d, optimizer=update_class
        )
        == 10_000
    )


def test_nelder_mead_is_not_budgeted_on_its_simplex(task_10d):
    # Its popsize is the simplex, n + 1 points, but a step after the
    # first evaluates only a few of them: e is 1, not popsize.
    nelder_mead = _built("neldermead", task_10d)
    assert nelder_mead.popsize == 11

    assert (
        evaluations_based_on_dimensionality(
            task=task_10d, optimizer=nelder_mead
        )
        != 10_000 // nelder_mead.popsize
    )


#                                                       B and its cap
# =============================================================================


def test_the_cap_binds_at_1024d(task_1024d):
    adam = _built("adam", task_1024d, learning_rate=1e-2)
    cmaes = _built("cmaes", task_1024d)
    assert cmaes.popsize == 24

    # 1000 * 1024 > 10^5, so B = 10^5.
    assert (
        evaluations_based_on_dimensionality(task=task_1024d, optimizer=adam)
        == 100_000
    )
    assert (
        evaluations_based_on_dimensionality(task=task_1024d, optimizer=cmaes)
        == 100_000 // 24
    )


def test_embedded_task_is_budgeted_on_its_ambient_dimension():
    task = create_embedded_bbob_task(
        fn_name="sphere", seed=0, intrinsic_dim=2, ambient_dim=64
    )
    assert task.dimensionality == 64
    adam = _built("adam", task, learning_rate=1e-2)

    # 1000 * 64, not 1000 * 2.
    assert (
        evaluations_based_on_dimensionality(task=task, optimizer=adam)
        == 64_000
    )


def test_settings_reach_the_formula(task_10d):
    cmaes = _built("cmaes", task_10d)
    budget_fn = partial(
        evaluations_based_on_dimensionality,
        evaluations_per_dimension=7,
        max_evaluations=55,
    )

    # B = min(7 * 10, 55) = 55; floor(55 / 10) = 5.
    assert budget_fn(task=task_10d, optimizer=cmaes) == 5


def test_defaults_are_the_decided_rule(task_10d):
    adam = _built("adam", task_10d, learning_rate=1e-2)
    configured = partial(
        evaluations_based_on_dimensionality,
        evaluations_per_dimension=1000,
        max_evaluations=100_000,
    )

    assert evaluations_based_on_dimensionality(
        task=task_10d, optimizer=adam
    ) == configured(task=task_10d, optimizer=adam)


#                                                   refusing a guess
# =============================================================================


def test_missing_optimizer_raises(task_10d):
    with pytest.raises(TypeError, match="needs the optimizer"):
        evaluations_based_on_dimensionality(task=task_10d)
    with pytest.raises(TypeError, match="needs the optimizer"):
        evaluations_based_on_dimensionality(task=task_10d, optimizer=None)


def test_a_budget_below_one_iteration_raises(task_10d):
    # B = min(1 * 10, 5) = 5 evaluations, but one CMA-ES iteration costs
    # lambda = 10: the run would get 0 iterations, which is never meant.
    cmaes = _built("cmaes", task_10d)
    with pytest.raises(ValueError, match="less than one iteration"):
        evaluations_based_on_dimensionality(
            task=task_10d,
            optimizer=cmaes,
            evaluations_per_dimension=1,
            max_evaluations=5,
        )
    # Exactly one iteration's worth is still a run.
    assert (
        evaluations_based_on_dimensionality(
            task=task_10d,
            optimizer=cmaes,
            evaluations_per_dimension=1,
            max_evaluations=10,
        )
        == 1
    )


def test_an_optimization_step_is_not_an_optimizer(task_10d):
    # The schedule is what ``RunOptimization`` calls ``optimizer``; the
    # budget needs the ``UpdateClass`` built from it.
    step = OptimizationStep("adam", hyperparameters={"learning_rate": 1e-2})
    with pytest.raises(TypeError, match="UpdateClass; got OptimizationStep"):
        evaluations_based_on_dimensionality(task=task_10d, optimizer=step)


#                                       the iteration budgets keep meaning
# =============================================================================


def test_iteration_budgets_ignore_the_optimizer(task_10d):
    cmaes = _built("cmaes", task_10d)

    assert (
        based_on_dimensionality(
            task=task_10d, budget_per_dimension=1000, optimizer=cmaes
        )
        == based_on_dimensionality(task=task_10d, budget_per_dimension=1000)
        == 10_000
    )
    assert (
        based_on_dimensionality(
            task=task_10d,
            budget_per_dimension=1000,
            max_iterations=500,
            optimizer=cmaes,
        )
        == 500
    )
    assert fixed_budget(budget=7, task=task_10d, optimizer=cmaes) == 7


def test_facade_exports_the_new_functions():
    from l2co._src import budget as budget_src

    assert (
        l2co.budget.evaluations_based_on_dimensionality
        is budget_src.evaluations_based_on_dimensionality
    )
    assert "evaluations_based_on_dimensionality" in l2co.budget.__all__
    assert l2co.budget.evaluation_budget is budget_src.evaluation_budget
    assert "evaluation_budget" in l2co.budget.__all__
