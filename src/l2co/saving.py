import json
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from f3dasm.run_optimization import (OptimizationResult,
                                     create_optimizationresult_from_json)

from l2co.create_recommender_dataset import \
    create_recommender_data_features_functions


def save_as_cdf(name: str, results: List[OptimizationResult]):
    """
    Save the optimization results as as cdf object

    Note
    ----
    I checked, the data is the same as:

    xd = xr.load_dataset(path)  # Load with xarray
    opt_res = load_results(path)  # Load with built-in f3dasm json decoder

    """

    df_list = [[d.data for d in results[i].data] for i, _ in enumerate(results)]
    for i, _ in enumerate(df_list):
        for j, _ in enumerate(df_list[i]):
            df_list[i][j].columns = df_list[i][j].columns.droplevel()

    data_arrays = [[df.to_xarray() for jj, df in enumerate(df_list[ii])] for ii, _ in enumerate(df_list)]

    optimizer_names = [res.optimizer.get_name() for res in results]  # Get optimizer names

    f = [xr.concat(data_arrays[index], dim='realization').assign_coords(optimizer=optimizer_names[index])
         for index, _ in enumerate(data_arrays)]  # Add optimizer name as a coordinate
    xarr = xr.concat(f, dim='optimizer')

    xarr.attrs['number_of_samples']: int = results[0].number_of_samples
    xarr.attrs['realization_seeds']: List[int] = list(results[0].seeds)

    # Benchmark functions
    xarr.attrs['function_seed']: int = results[0].function.seed
    xarr.attrs['function_name']: str = results[0].function.get_name()
    xarr.attrs['function_noise']: str = results[0].function.noise
    xarr.attrs['function_dimensionality']: int = results[0].function.dimensionality
    xarr.attrs['function_features']: str = create_recommender_data_features_functions(results)

    # Global minimum function
    _, g = results[0].function.get_global_minimum(d=results[0].function.dimensionality)
    xarr.attrs['function_global_minimum']: float = float(np.array(g if not isinstance(g, list) else g[0])[0, 0])

    # Add optimizer features
    optimizer_features = xr.DataArray([",".join(
        result.optimizer.get_info()) for result in results],
        coords={'optimizer': [result.optimizer.get_name() for result in results]})

    xarr = xarr.assign_coords(optimizer_features=optimizer_features)

    # Save to disc
    xarr.to_netcdf(f'{name}.cd')


def from_xarray_to_dataframe(xarr: xr.Dataset) -> List[List[pd.DataFrame]]:
    df_list = []
    for optimizer in xarr.optimizer:
        df_optimizer = []
        for realization in xarr.realization:
            df_realization = xarr.sel(optimizer=optimizer, realization=realization).to_dataframe()
            df_realization.columns = pd.MultiIndex.from_tuples(
                [tuple(col.split('_')) for col in df_realization.columns])
            df_optimizer.append(df_realization)
        df_list.append(df_optimizer)
    return df_list


def load_results(filepath: str) -> List[OptimizationResult]:
    # Load results
    with open(filepath) as f:
        json_string = json.load(f)

    json_results = json.loads(json_string)
    results = [create_optimizationresult_from_json(r) for r in json_results]
    return results
