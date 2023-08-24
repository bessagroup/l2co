#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Protocol, Tuple, Union

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


class Performance(Protocol):
    def __call__(self, data: xr.Dataset) -> pd.DataFrame:
        """Protocol function that calculates the performance of a dataset

        Parameters
        ----------
        data
            raw optimization data

        Returns
        -------
            performance of the dataset in dataframe format
        """
        ...


def _calculate_mean_std(data: Union[xr.Dataset, List[OptimizationResult]]) -> Tuple[pd.DataFrame, List[str]]:
    if isinstance(data, xr.Dataset):
        df = _calculate_mean_std_xarray(data)
        optimizer_names = list(data.optimizer.values)

    elif isinstance(data, List):
        df = _calculate_mean_std_listoptimizationresult(data)
        optimizer_names = [data[i].optimizer.get_name() for i, _ in enumerate(data)]

    else:
        raise TypeError(f"results should be of type List[OptimizationResult] or xr.Dataset, not {type(data)}")

    return df, optimizer_names


def _calculate_mean_std_xarray(xd: xr.Dataset) -> pd.DataFrame:
    xd = xd.assign(output_cummin=(['optimizer', 'realization', 'iterations',
                   'output_dim'], np.nan*np.ones(xd['output'].shape)))

    # loop over the optimizer and realization dimensions
    for opt in xd.optimizer.values:
        for r in xd.realization.values:

            y_cummin = xd['output_cummin'].sel(optimizer=opt, realization=r)

            y: np.ndarray = xd['output'].sel(optimizer=opt, realization=r).values
            y_accumulate = np.minimum.accumulate(y)  # 1D numpy array

            y_cummin[:] = y_accumulate

    # WORKING: retrieve the mean of the dataset over the realizations
    xd_mean = xd.mean(dim="realization")

    df_xd = xd_mean.to_dataframe()
    df = df_xd.pivot_table(values='output_cummin', index='iterations', columns='optimizer')
    return df


def _calculate_mean_std_xarray_old(xd: xr.Dataset) -> pd.DataFrame:
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


def _calculate_mean_std_listoptimizationresult(results: List[OptimizationResult]) -> pd.DataFrame:
    # Create df with all results
    df = pd.concat([calculate_mean_std(results[i])[0] for i, _ in enumerate(results)], axis=1)

    # Change columnnames
    df.columns = [results[i].optimizer.get_name() for i, _ in enumerate(results)]

    return df


def _normalize(df: pd.DataFrame, optimizer_names: List[str]) -> pd.DataFrame:
    # Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    # Reshape to 1D array
    df_numpy = df.values  # returns a numpy array
    df_numpy_reshaped = df_numpy.reshape(-1, 1)

    x_scaled = min_max_scaler.fit_transform(df_numpy_reshaped)

    # Transform back
    x_scaled = x_scaled.reshape(df_numpy.shape)
    df = pd.DataFrame(x_scaled, columns=optimizer_names)
    # df.columns = optimizer_names
    return df


def margin_of_victory(results: Union[List[OptimizationResult], xr.Dataset], budget: int = -1) -> pd.DataFrame:
    """Calculate the margin of victory of optimizationresults

    Parameters
    ----------
    results
        List of OptimizationResults objects or a xr.Dataset

    Returns
    -------
        DataFrame containing the algorithms and the margin of victory
    """
    df, optimizer_names = _calculate_mean_std(results)

    df = _normalize(df, optimizer_names)

    # Calculate margin of victory
    mov = []
    for name in optimizer_names:
        df_dropped = df.drop(name, axis=1)
        mov.append(df_dropped.min(axis=1) - df[name])

    # Create df with all MoV
    df_margin_of_victory = pd.concat(mov, axis=1)

    # Change columnnames
    df_margin_of_victory.columns = optimizer_names

    if budget == -1:
        return df_margin_of_victory

    original_len = len(df_margin_of_victory)
    # use only the first 100 rows
    df_margin_of_victory = df_margin_of_victory[:budget]

    # Create a new DataFrame with original_len rows
    new_df = pd.DataFrame(index=range(original_len))

    # Interpolate between rows
    for column in df_margin_of_victory.columns:
        new_df[column] = new_df.index.map(
            lambda idx: df_margin_of_victory[column][idx * len(df_margin_of_victory) // original_len]
        )

    return new_df


def area_under_curve(results: Union[List[OptimizationResult], xr.Dataset]) -> pd.DataFrame:
    """Calculate the area under the curve of optimizationresults

    Parameters
    ----------
    results
        List of OptimizationResults objects or a xr.Dataset

    Returns
    -------
        DataFrame containing the algorithms and the area under the curve
    """
    df, optimizer_names = _calculate_mean_std(results)

    df = _normalize(df, optimizer_names)

    # Calculate area under curve

    # calculate the cumulative sum along the columns (area under the curve)
    df_cumsum = df.cumsum(axis=0)

    # divide df_cumsum by the number of rows
    df_cumsum = df_cumsum.div(df_cumsum.index + 1, axis=0)

    # reverse the sign in order to make good algorithms have a higher value
    df_cumsum = -df_cumsum

    return df_cumsum
