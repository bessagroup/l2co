"""Dataset generation hands the built optimizer to the budget function.

``RunOptimization.run_optimization`` calls ``budget_fn`` after
``RolloutWrapper.init``, with the ``UpdateClass`` of the run state it is
about to roll out, so an evaluation budget can turn into the right
number of iterations (ADR 0026).
"""

from functools import partial

import pytest
from l2co_optimizers import OptimizationStep
from l2co_tasks import create_bbob_task

from l2co._src.rollout import dataset_generation
from l2co._src.rollout.dataset_generation import RunOptimization
from l2co._src.rollout.rollout_wrapper import RolloutWrapper
from l2co.budget import evaluations_based_on_dimensionality
from l2co.naming import BUDGET, FEVALS
from l2co.sampling import normal_sampling

# =============================================================================


class _FakeSample:
    """Duck-typed ``ExperimentSample``: keeps what is stored in memory."""

    def __init__(self):
        self.stored = {}

    def store(self, name, object, to_disk=False):
        self.stored[name] = object


class _StopBeforeRollout(Exception):
    """Raised by the fake budget to skip the (compiling) rollout."""


def test_budget_fn_receives_the_run_states_update_class(monkeypatch):
    built = []
    real_init = RolloutWrapper.init

    class _RecordingRolloutWrapper:
        @staticmethod
        def init(**kwargs):
            wrapper = real_init(**kwargs)
            built.append(wrapper)
            return wrapper

    monkeypatch.setattr(
        dataset_generation, "RolloutWrapper", _RecordingRolloutWrapper
    )
    calls = []

    def budget_fn(**kwargs):
        calls.append(kwargs)
        raise _StopBeforeRollout

    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=3)
    runner = RunOptimization(
        sampler=normal_sampling, budget_fn=budget_fn, seed=0
    )

    with pytest.raises(_StopBeforeRollout):
        runner.run_optimization(
            experiment_sample=_FakeSample(),
            task=task,
            optimizer=OptimizationStep("cmaes"),
        )

    assert len(built) == 1, "budget_fn ran before the optimizer was built"
    (call,) = calls
    assert set(call) == {"task", "optimizer"}
    assert call["task"] is task
    assert call["optimizer"] is built[0].run_state.update_class


@pytest.mark.slow
def test_a_population_run_spends_at_most_b_evaluations():
    task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)
    # B = 20 * 2 = 40 evaluations; CMA-ES at 2-D has lambda = 6.
    runner = RunOptimization(
        sampler=normal_sampling,
        budget_fn=partial(
            evaluations_based_on_dimensionality, evaluations_per_dimension=20
        ),
        seed=0,
        n_realizations=2,
        clear_jax_cache=False,
    )
    sample = _FakeSample()

    runner.run_optimization(
        experiment_sample=sample,
        task=task,
        optimizer=OptimizationStep("cmaes"),
    )

    history = sample.stored["loss history"]
    assert history.sizes[BUDGET] == 40 // 6
    assert int(history[FEVALS].isel({BUDGET: -1}).max()) == (40 // 6) * 6
