"""Unit tests for ``DataLoader.from_xarray`` label parsing and the
``plot_update_step`` sentinel reclassification."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

import l2co
from l2co._src.naming import (
    BUDGET,
    FEATURE,
    OUTPUT_DIM,
    OUTPUT_MIN,
    REALIZATION,
    SCHEDULE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)

#                                                                      Fixtures
# =============================================================================

FEATS = [
    "separable",
    "unimodal",
    "dimensionality",
    "fn_name",
    "seed",
    "noise",
    "global_min",
]
VALS = ["False", "False", "20", "gallagher_21_peaks", "0", "0.0", "-116.56"]


def _labelled_dataset() -> xr.Dataset:
    """Minimal dataset shaped like ``HistoryState.to_dataloader`` output."""
    labels = (
        xr.DataArray(
            VALS, coords={FEATURE: FEATS}, dims=[FEATURE], name=TASK_LABELS
        )
        .astype(str)
        .expand_dims({REALIZATION: [0, 1, 2]})
        .expand_dims({TASK_ID: ["abc"]})
    )
    data = xr.DataArray(
        np.zeros((1, 1, 3, 5, 1)),
        dims=[TASK_ID, SCHEDULE, REALIZATION, BUDGET, OUTPUT_DIM],
        coords={
            TASK_ID: ["abc"],
            SCHEDULE: ["rl2co"],
            REALIZATION: [0, 1, 2],
        },
        name=OUTPUT_MIN,
    )
    return xr.merge(
        [data.to_dataset(), labels.to_dataset()], compat="override"
    )


#                                                       from_xarray label parse
# =============================================================================


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_from_xarray_preserves_label_types(lazy: bool):
    """Labels keep their parsed types; a leading boolean feature must not
    collapse the whole array to ``bool`` (regression: dask-backed zarr
    input turned ``global_min`` into ``True`` -> 1.0)."""
    ds = _labelled_dataset()
    if lazy:
        ds = ds.chunk({})

    d = l2co.DataLoader.from_xarray(ds)
    row = d.task_labels.isel(problem=0)

    assert d.task_labels.dtype == object
    assert row.sel({FEATURE: "separable"}).item() is False
    assert row.sel({FEATURE: "fn_name"}).item() == "gallagher_21_peaks"
    assert row.sel({FEATURE: "dimensionality"}).item() == 20
    assert row.sel({FEATURE: "noise"}).item() == 0.0
    assert float(
        d.task_labels.sel({FEATURE: "global_min"}).astype(float).values[0]
    ) == pytest.approx(-116.56)


#                                            plot_update_step reclassification
# =============================================================================


def _plot_dataset(
    update_step: np.ndarray, output_min: np.ndarray, global_min: float
) -> l2co.DataLoader:
    """DataLoader with explicit ``update_step`` / ``output_min`` rows."""
    n_problems, n_budget = update_step.shape
    feats = ["dimensionality", "global_min"]
    vals = ["2", str(global_min)]
    labels = (
        xr.DataArray(
            vals, coords={FEATURE: feats}, dims=[FEATURE], name=TASK_LABELS
        )
        .astype(str)
        .expand_dims({REALIZATION: list(range(n_problems))})
        .expand_dims({TASK_ID: ["abc"]})
    )

    def _wrap(arr: np.ndarray, name: str) -> xr.DataArray:
        return xr.DataArray(
            arr[None, None, :, :, None].astype(float),
            dims=[TASK_ID, SCHEDULE, REALIZATION, BUDGET, OUTPUT_DIM],
            coords={
                TASK_ID: ["abc"],
                SCHEDULE: ["rl2co"],
                REALIZATION: list(range(n_problems)),
            },
            name=name,
        )

    ds = xr.merge(
        [
            _wrap(update_step, UPDATE_STEP).to_dataset(),
            _wrap(output_min, OUTPUT_MIN).to_dataset(),
            labels.to_dataset(),
        ],
        compat="override",
    )
    return l2co.DataLoader.from_xarray(ds)


def test_plot_update_step_sentinels():
    """Solved cells go black, padding stays white even when the running
    minimum carries through it, unsolved rows keep optimizer colors."""
    opt = l2co.OptimizationStep(optimizer="lbfgs")
    h = opt.hash
    nan = np.nan
    # row 0: solved at column 2, padding (unwritten, hash 0) from col 4
    # row 1: never solved, padding from col 4
    # row 2: solved at column 3, fully written
    update_step = np.array(
        [
            [h, h, h, h, 0, 0],
            [h, h, h, h, 0, 0],
            [h, h, h, h, h, h],
        ]
    )
    output_min = np.array(
        [
            [5.0, 2.0, 0.0, 0.0, nan, nan],
            [5.0, 4.0, 3.0, 2.0, nan, nan],
            [5.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    d = _plot_dataset(update_step, output_min, global_min=0.0)

    fig, ax = l2co.plotting.plot_update_step(
        d=d,
        schedule="rl2co",
        optimizers=[opt],
        show_legend=True,
        hide_when_done=True,
    )
    mesh = ax.collections[0]
    z = mesh.get_array().reshape(3, 6)
    done_index = 1  # one optimizer -> black appended at index 1

    # row 0: colors until solved, black after, padding masked (white)
    assert list(z[0, :2]) == [0, 0]
    assert list(z[0, 2:4]) == [done_index, done_index]
    assert bool(np.all(z.mask[0, 4:]))
    # row 1: never solved -> colors, then white padding
    assert list(z[1, :4]) == [0, 0, 0, 0]
    assert bool(np.all(z.mask[1, 4:]))
    # row 2: fully written, black from the solve column on
    assert list(z[2, :3]) == [0, 0, 0]
    assert list(z[2, 3:]) == [done_index] * 3

    # legend carries both sentinel entries
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Global min. reached" in labels
    assert "No data" in labels

    matplotlib.pyplot.close(fig)
