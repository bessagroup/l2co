"""
Module containing functions that can be represent baseslines in the training
set.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path

# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def output_cummin(ds: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        np.minimum.accumulate(ds.output.values, axis=ds.output.get_axis_num(
            'iterations')), coords=ds.output.coords, dims=ds.output.dims)


def rs_baseline(rs: xr.Dataset, size: int, shuffles: int = 99) -> np.ndarray:

    rs_data = np.random.choice(rs['output'].values.ravel(), size=size)
    shuffled_arrays = np.minimum.accumulate(rs_data).reshape(1, -1)

    for _ in range(shuffles):
        np.random.shuffle(rs_data)
        shuffled_arrays = np.concatenate(
            [shuffled_arrays, np.minimum.accumulate(rs_data).reshape(1, -1)])

    # take the median over shuffled_arrays first dimension;

    return np.median(shuffled_arrays, axis=0)


def calculate_metric(
        ds: xr.Dataset, baseline: np.ndarray) -> xr.DataArray:
    ds['output_cummin'] = output_cummin(ds)
    y_min = ds.output.min()

    a = ds.output_cummin.sel(
        output_dim='y').median('realization') - y_min + 1.0
    b = baseline - y_min.values + 1.0

    return np.log(a/b)


def _validate_path(path: str | float, project_dir: Path) -> bool:
    if isinstance(path, str):
        return Path(project_dir / path).exists()

    return False


def get_metric(
        data: ExperimentData, index: int, baseline: str = 'randomsearch'):

    experiment_sample = data.get_experiment_sample(index)

    ds = xr.open_mfdataset(
        [data.project_dir / path for key, path in
         experiment_sample.output_data_with_references.items()
         if _validate_path(
             path, data.project_dir) and key != 'random_samples'])

    if baseline == 'randomsearch':
        rs = xr.open_dataset(
            data.project_dir / experiment_sample.
            output_data_with_references['random_samples'])

        baseline = rs_baseline(rs, size=len(ds.iterations))

    return calculate_metric(ds, baseline)
