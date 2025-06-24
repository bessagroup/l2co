from functools import partial

import pytest
import xarray as xr

from l2co._src.budget import based_on_dimensionality
from l2co._src.run_optimization import \
    run_optimization_schedule_multiple_realizations

from . import OPTIMIZATION_SCHEDULES

# =============================================================================


@pytest.mark.parametrize("bounded", [(0.0, 1.0), None], ids=["bounded_0_1", "bounded_none"])
@pytest.mark.parametrize("optimization_schedule", OPTIMIZATION_SCHEDULES, ids=lambda s: f"optimizer_{s.steps[0].optimizer}")
def test_optimization_schedules(benchmark_task, bounded, optimization_schedule):
    res = run_optimization_schedule_multiple_realizations(
        task=benchmark_task, optimization_schedule=optimization_schedule,
        budget_fn=partial(based_on_dimensionality, budget_per_dimension=5),
        seed=42,
        partition_factor=1.0,
        n_realizations=2,
        include_input=False,
        bounded=bounded,
        sampler=None)

    assert res is not None
    assert isinstance(res, xr.Dataset)
