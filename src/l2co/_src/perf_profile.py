"""
Module containing functions that compute the performance of optimizers

Functions
---------

get_performance
    Generate a dataset that captures the performance of an optimizer
    on a benchmark function as function of the number of function evaluations.
query_t
    Query the performance dataset for the performance of the optimizers
    after a given number of function evaluations.
is_sparse
    Check if the performance DataArray is sparse.
collaborative_filtering
    Fill in missing values in the performance DataFrame using
    collaborative filtering.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from itertools import cycle
from typing import Iterable, List, Optional

# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from f3dasm import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def _output_cummin(ds):
    """
    Calculate the cumulative minimum of the given dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate the cumulative minimum for.

    Returns
    -------
    xr.Dataset
        The dataset with the cumulative minimum calculated.

    Notes
    -----
    The dataset calculate the cumulative minimum for the coordinate named
    'iterations'.
    """

    def min_accumulate_func(arr):
        return np.minimum.accumulate(arr)

    return xr.apply_ufunc(
        min_accumulate_func,
        ds,
        input_core_dims=[['iterations']],
        output_core_dims=[['iterations']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[ds.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )


def load_all_data(experiment_data: ExperimentData) -> xr.Dataset:
    """
    Load all the optimization dynamics data for the given experiment data.

    Parameters
    ----------
    experiment_data : ExperimentData
        The experiment data to load the optimization dynamics data for.

    Returns
    -------
    ds_raw : xr.Dataset
        The optimization dynamics data for the given experiment data.

    Notes
    -----

    The ExperimentData object contains output columns representing the
    different optimizers. Each of these columns contains a list of files
    containing the optimization dynamics data for the different problems.

    We combine all these files into a single dataset, where the itemID (rows
    in ExperimentData) and optimizer (columnname in output data) are used
    as coordinates.
    """
    _, df_output = experiment_data.to_pandas()
    # for every element in df_output, preceed it with project_dir
    df_output = df_output.loc[:, df_output.columns != 'random_samples']
    df_output = df_output.apply(lambda x: experiment_data.project_dir / x)
    files = df_output.values.tolist()
    # files

    # Open all files in ds_raw
    return xr.open_mfdataset(
        files, concat_dim=['itemID', 'optimizer'], combine='nested')


def calculate_performance_ratio(
        ds_budget: xr.DataArray,
        best_optimizer_each_problem: xr.DataArray,
        max_f_value: float = 10.0) -> xr.DataArray:
    """
    Calculate the performance ratio for the given dataset.

    Parameters
    ----------

    ds_budget : xr.DataArray
        The dataset to calculate the performance ratio for.

    best_optimizer_each_problem : xr.DataArray
        The best optimizer for each problem.

    max_f_value : float, optional
        The maximum value of f, by default 10.0

    Returns
    -------
    performance_profile : xr.DataArray
        The performance profile for the given dataset.
    """
    # Calculate the performance ratio rij
    rij = ds_budget / best_optimizer_each_problem

    # Create a logarithmic scale for f
    # combine rij and f in a dataset
    ds_f = xr.Dataset({'rij': rij, 'f': xr.DataArray(
        np.logspace(np.log10(1.0), np.log10(max_f_value), 1000), dims='f')})

    performance_profile = np.less_equal(ds_f, ds_f.f+1e-4).where(
        ~ds_f.isnull(), np.nan).mean(dim='problem').rij
    return performance_profile


def create_performance_profile(
        ds_raw: ExperimentData | xr.Dataset,
        ds_samples: xr.Dataset,
        iterations: int,
        t_switch: int,
        strategies: Optional[List[xr.DataArray]] = None) -> xr.Dataset:
    """
    Create a performance profile for the given experiment data and iterations.

    Parameters
    ----------
    experiment_data : ExperimentData | xr.Dataset
        The experiment data to create the performance profile for.

    iterations : int
        The total budget for each of the optimization problems

    strategies : List[xr.DataArray], optional
        The strategies to include in the performance profile, by default None

    Returns
    -------
    performance_profile : xr.Dataset
        The performance profile for the given experiment data and iterations.
        The variables are the different static optimizers, and the optional
        added strategies.

    Notes
    -----
    If the experiment_data is an ExperimentData object, the optimization
    dynamics data is loaded from the files in the ExperimentData object.

    Strategies are optional, and can be added to the performance profile
    by passing a list of xr.DataArray objects. The xr.DataArray objects
    should have a 'problem' coordinate which is identical to the data provided
    in the experiment_data parameter.
    """

    # Load all the optimization dynamics data
    if isinstance(ds_raw, ExperimentData):
        ds_raw = load_all_data(ds_raw)

    # Select only the output variable
    ds_output = ds_raw.sel(output_dim='y', drop=True).output
    data_samples = ds_samples.sel(output_dim='y', drop=True).output

    opts = [entry for entry in ds_output.optimizer.values if '+' not in entry]

    a = []
    for opt in opts:
        a.append(
            xr.concat([data_samples.sel(
                iterations=np.arange(t_switch)), ds_output.sel(
                    optimizer=opt, iterations=np.arange(iterations-t_switch)
            ).assign_coords(iterations=np.arange(t_switch, iterations))],
                dim='iterations').assign_coords(
                    optimizer=f'random_samples+{opt}'))

    random_combinations = xr.concat(a, dim='optimizer')

    ds_output = xr.concat([ds_output, random_combinations], dim='optimizer')

    # combine the itemID and realization coordinate
    # ds_output = ds_output.stack(problem=('itemID', 'realization'))

    # Calculat the best found value (cummin)
    ds_output = _output_cummin(ds_output)

    # Scale everything to be 1.0 or larger
    # ds_positive = ds_output - \
    #     ds_output.min(dim=['optimizer', 'iterations']) + 1.0

    ds_positive = ds_output - \
        ds_output.min(dim=['optimizer', 'iterations']).where(
            ds_output.min(dim=['optimizer', 'iterations']
                          ) < 0, 0.0) + 1e-17

    # Slice at the given total number of iterations
    ds_budget = ds_positive.sel(iterations=iterations-1)
    ds_budget = ds_budget.sortby('problem')

    # Calculate the performance profile
    performance_profile = calculate_performance_ratio(
        ds_budget, ds_budget.min(dim='optimizer'))

    pp_to_dataset = xr.Dataset({opt: performance_profile.sel(
        optimizer=opt, drop=True) for opt
        in performance_profile.optimizer.values})

    if strategies is not None:
        for strategy in strategies:
            _strategy = strategy.sortby('problem')
            pp_to_dataset[strategy.name] = calculate_performance_ratio(
                ds_budget.sel(optimizer=_strategy), ds_budget.min(
                    dim='optimizer')
            )

    return pp_to_dataset


def plot_performance_profile(
    performance_profile: xr.Dataset,
        strategy_names: Optional[List[str]] = None,
        axes: Optional[Iterable[plt.Axes, plt.Axes]] = None,
        show_combinations: bool = True):
    """
    Plot the performance profile for the given dataset.

    Parameters
    ----------
    performance_profile : xr.Dataset
        The performance profile to plot
    strategy_names : List[str], optional
        The strategies to highlight in the plot, by default None
    axes : List[plt.Axes], optional
        The axes to plot the performance profile on, by default None
    show_combinations : bool, optional
        If combinations of optimizers should be shown, by default True

    Returns
    -------
    axes : List[plt.Axes]
        The axes the performance profile was plotted on.
    """

    if strategy_names is None:
        strategy_names = []

    legend_patches = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle_iter = cycle(color_cycle)

    if axes is None:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
            4, 1]}, figsize=(10, 7))
    # axes[0].set_xscale('log')
    # ax.set_yscale('log')

    x = performance_profile.f * 100 - 100
    for optimizer in performance_profile.data_vars:
        linewidth = 1
        linestyle = '-'

        if optimizer in strategy_names:
            # make the line thicker
            linewidth = 3

        # check if optimizer has a + in the name
        if '+' in optimizer:
            if show_combinations is False:
                continue
            else:
                linestyle = '--'

        legend_patches.append(mpatches.Patch(
            color=next(color_cycle_iter),
            label=optimizer, linestyle=linestyle))

        axes[0].plot(x,
                     performance_profile[optimizer],
                     label=optimizer, linewidth=linewidth,
                     linestyle=linestyle)

    axes[1].legend(handles=legend_patches, loc='center',
                   ncol=len(legend_patches) // 3, fontsize='small')

    axes[1].axis('off')

    axes[0].set_ylabel('Relative error [%]')
    axes[0].set_xlabel('Performance ratio')

    return axes
