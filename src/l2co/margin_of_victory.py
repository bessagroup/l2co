#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Union

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm.run_optimization import OptimizationResult, calculate_mean_std
from sklearn import preprocessing

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def calculate_mean_std_xarray(xd: xr.Dataset) -> xr.Dataset:
    # WORKING: get the cummin of the y values for each optimizer and realization

    xd = xd.assign(y_cummin=(['optimizer', 'realization', 'index'], np.nan*np.ones(xd['y'].shape)))

    # loop over the optimizer and realization dimensions
    for opt in xd.optimizer.values:
        for r in xd.realization.values:

            y_cummin = xd['y_cummin'].sel(optimizer=opt, realization=r)

            y: np.ndarray = xd['y'].sel(optimizer=opt, realization=r).values
            y_accumulate = np.minimum.accumulate(y)  # 1D numpy array

            y_cummin[:] = y_accumulate

    # WORKING: retrieve the mean of the dataset over the realizations
    xd_mean = xd.mean(dim="realization")

    df_xd = xd_mean.to_dataframe()
    df = df_xd.pivot_table(values='y_cummin', index='index', columns='optimizer')

    return df


def calculate_mean_std_listoptimizationresult(results: List[OptimizationResult]) -> pd.DataFrame:
    # Create df with all results
    df = pd.concat([calculate_mean_std(results[i])[0] for i, _ in enumerate(results)], axis=1)

    # Change columnnames
    df.columns = [results[i].optimizer.get_name() for i, _ in enumerate(results)]

    return df


def margin_of_victory(results: Union[List[OptimizationResult], xr.Dataset]) -> pd.DataFrame:
    """Calculate the margin of victory of optimizationresults

    Parameters
    ----------
    results
        List of OptimizationResults objects or a xr.Dataset

    Returns
    -------
        DataFrame containing the algorithms and the margin of victory
    """
    if isinstance(results, xr.Dataset):
        df = calculate_mean_std_xarray(results)
        optimizer_names = list(results.optimizer.values)

    else:
        df = calculate_mean_std_listoptimizationresult(results)
        optimizer_names = [results[i].optimizer.get_name() for i, _ in enumerate(results)]

    # Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    # Reshape to 1D array
    df_numpy = df.values  # returns a numpy array
    df_numpy_reshaped = df_numpy.reshape(-1, 1)

    x_scaled = min_max_scaler.fit_transform(df_numpy_reshaped)

    # Transform back
    x_scaled = x_scaled.reshape(df_numpy.shape)
    df = pd.DataFrame(x_scaled)
    df.columns = optimizer_names

    # Calculate margin of victory
    mov = []
    for name in optimizer_names:
        df_dropped = df.drop(name, axis=1)
        mov.append(df_dropped.min(axis=1) - df[name])

    # Create df with all MoV
    df_margin_of_victory = pd.concat(mov, axis=1)

    # Change columnnames
    df_margin_of_victory.columns = optimizer_names

    return df_margin_of_victory
