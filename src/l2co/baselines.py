"""
Module containing functions that can be represent baseslines in the training
set.
"""

#                                                                       Modules
# =============================================================================

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


def rs_baseline(rs: xr.Dataset, budget: int, shuffles: int = 99) -> np.ndarray:
    rs_data = rs['output'].values.ravel()
    shuffled_arrays = np.minimum.accumulate(rs_data).reshape(1, -1)

    for _ in range(shuffles):
        np.random.shuffle(rs_data)
        shuffled_arrays = np.concatenate(
            [shuffled_arrays, np.minimum.accumulate(rs_data).reshape(1, -1)])

    # take the median over shuffled_arrays first dimension;

    return np.median(shuffled_arrays, axis=0)[:budget]


def calculate_metric(
        ds: xr.Dataset, baseline: np.ndarray, budget: int) -> xr.DataArray:
    ds['output_cummin'] = output_cummin(ds)
    y_min = ds.output.min()

    a = ds.output_cummin.sel(
        output_dim='y',
        iterations=range(0, budget)).median('realization') - y_min + 1.0
    b = baseline - y_min.values + 1.0

    return np.log(a/b)


def get_metric(
        data: ExperimentData, index: int,
        budget: int, baseline: str = 'randomsearch'):

    experiment_sample = data.get_experiment_sample(index)

    ds = xr.open_mfdataset(
        [data.project_dir / path for key, path in
         experiment_sample.output_data_with_references.items()
         if isinstance(path, str) and key != 'random_samples'])

    if baseline == 'randomsearch':
        rs = xr.open_dataset(
            data.project_dir / experiment_sample.
            output_data_with_references['random_samples'])

        baseline = rs_baseline(rs, budget)

    return calculate_metric(ds, baseline, budget)
