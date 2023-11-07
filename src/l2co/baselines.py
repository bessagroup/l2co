import numpy as np
import xarray as xr


def output_cummin(ds: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        np.minimum.accumulate(ds.output.values, axis=ds.output.get_axis_num(
            'iterations')), coords=ds.output.coords, dims=ds.output.dims)


def rs_baseline(rs: xr.Dataset, shuffles: int = 99) -> np.ndarray:
    rs_data = rs['output'].values.ravel()
    shuffled_arrays = np.minimum.accumulate(rs_data).reshape(1, -1)

    for _ in range(shuffles):
        np.random.shuffle(rs_data)
        shuffled_arrays = np.concatenate(
            [shuffled_arrays, np.minimum.accumulate(rs_data).reshape(1, -1)])

    # take the median over shuffled_arrays first dimension;
    return np.median(shuffled_arrays, axis=0)
