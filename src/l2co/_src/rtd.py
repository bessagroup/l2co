from typing import List, Optional

import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

# from .data_loader import DataLoader
from .performance import output_cummin


def scaling(data: xr.Dataset) -> xr.DataArray:
    """
    Scale the data to be used as input for the RTD algorithm.

    Parameters
    ----------
    data : xr.Dataset
        The data to be scaled.

    Returns
    -------
    xr.DataArray
        The scaled data.
    """
    # 1. Calculate cumulative minimum (quality indictor value)
    data['output_cummin'] = output_cummin(data)

    # 2. Make sure the loss is non-negative
    y_min = data.output_cummin.min(dim=['optimizer', 'iterations']).where(
        data.output_cummin.min(dim=['optimizer', 'iterations']) < 0.0, 0.0)
    ds_positive = data['output_cummin'] - y_min.min('realization')

    # 3. Make sure the loss is non-zero
    ds_add_to_become_non_zero = (
        ds_positive.min(dim=['optimizer', 'iterations']
                        ).min('realization') == 0.0).where(
        (~(ds_positive.min(dim=['optimizer', 'iterations']
                           ).min('realization') == 0.0)
         ), other=ds_positive.where(
             ds_positive > 0.0).min(
                 dim=['optimizer', 'iterations']).min('realization')
    )

    ds_positive += ds_add_to_become_non_zero

    # 4. Take the base 10 logarithm of the data
    ds_log10_positive = np.log10(ds_positive)

    # 5. Scale the data
    scaler = MinMaxScaler()

    def scale(arr, minimum_over_realizations):
        scaler.fit(minimum_over_realizations.reshape(-1, 1))
        scaler.partial_fit(arr.flatten().reshape(-1, 1))
        return np.reshape(
            scaler.transform(arr.flatten().reshape(-1, 1)), arr.shape)

    return xr.apply_ufunc(
        scale, ds_log10_positive,
        ds_log10_positive.min(
            dim=['optimizer', 'iterations', 'output_dim', 'realization']),
        input_core_dims=[
            ['optimizer', 'iterations', 'output_dim'],
            []],
        output_core_dims=[['optimizer', 'iterations', 'output_dim']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
    )