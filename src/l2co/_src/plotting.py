"""
Plotting module

Functions
---------
set_size
    Set figure dimensions
plot_rtd_profile
    Plot the RTD profile for the given DataLoader.
plot_perf_profile
    Plot the performance profile for the given DataLoader.
"""
#                                                                       Modules
# =============================================================================

from typing import Iterable, List, Optional, Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# Local
from l2co._src.data_loader import DataLoader, OptimizationDynamics
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


#                                                                 Aspect ratios
# =============================================================================


OPTIMIZER_COLORS = {
    'Adam': plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
    'CMAES': plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
    'LBFGSB': plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
    'PSO': plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
}


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'paper':
        width_pt = 345.0
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

# =============================================================================


def plot_rtd_profile(d: DataLoader, ax: Optional[plt.Axes] = None,
                     strategies: Optional[List[xr.DataArray]] = None,
                     include_best: bool = False,
                     optimizers_to_plot: Optional[Iterable[str]] = None
                     ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the run-time distribution for the given DataLoader.

    Parameters
    ----------
    d: DataLoader
        DataLoader instance
    ax: plt.Axes, optional
        Matplotlib axes
    strategies: list of xr.DataArray, optional
        List of strategies to plot, by default None
    include_best: bool, optional
        Include the best strategy in the plot, by default False
    optimizers_to_plot: list of str, optional
        List of optimizers to plot, by default None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes

    Notes
    -----
    If no axes are provided, a new figure is created.
    """
    _rtd = d.rtd_profile(strategies=strategies,
                         include_best=include_best).to_array(
        'optimizer').mean('problem').isel(output_dim=0)

    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=set_size(width='paper', subplots=(1, 1)))
    else:
        fig = ax.get_figure()

    if optimizers_to_plot is None:
        for optimizer in _rtd.optimizer:
            ax.plot(_rtd.sel(optimizer=optimizer), label=optimizer.values)

    else:
        for optimizer in optimizers_to_plot:
            ax.plot(_rtd.sel(optimizer=optimizer), label=optimizer)

    ax.legend()
    ax.set_xlabel('Function evaluations')
    ax.set_ylabel('Fraction of function, target pairs')
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def plot_perf_profile(
    d: DataLoader, ax: Optional[plt.Axes] = None,
    strategies: Optional[List[xr.DataArray]] = None,
    include_best: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the performance profile for the given DataLoader.

    Parameters
    ----------
    d: DataLoader
        DataLoader instance
    ax: plt.Axes, optional
        Matplotlib axes
    strategies: list of xr.DataArray, optional
        List of strategies to plot, by default None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes

    Notes
    -----
    If no axes are provided, a new figure is created.
    """
    _perf = d.performance_profile(
        strategies, include_best=include_best).to_array('optimizer')
    rel_error = _perf.f * 100 - 100

    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=set_size(width='paper', subplots=(1, 1)))
    else:
        fig = ax.get_figure()

    for optimizer in _perf.optimizer:
        ax.plot(rel_error, _perf.sel(optimizer=optimizer),
                label=optimizer.values)

    ax.legend()
    ax.set_xlabel('Relative error [%]')
    ax.set_ylabel('Fraction of solved problems')
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def plot_trajectories(
        optimization_dynamics: OptimizationDynamics, optimizer: str,
        n_rows: int = 10, n_columns: int = 10,
        seed: Optional[int] = None,
        idx: Iterable[int] = None) -> Tuple[plt.Figure, np.ndarray]:

    X_traj = optimization_dynamics._X_trajectory.isel(
        optimizer=0).T
    Y = optimization_dynamics._Y.isel(output_dim=0).T

    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(2*n_rows, 2*n_columns),
        sharex=True, sharey=True)
    axes = axes.flatten()
    Y_substract_mean = Y - Y.mean('optimizer')

    if idx is None:
        # create numpy generator
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_traj), size=len(axes))

    idx = idx[:len(axes)]

    # randomly select 100 trajectories
    for ax, i in zip(axes, idx):

        # Bar colors
        bar_colors = [
            'white' if opt == optimizer else OPTIMIZER_COLORS[opt]
            for opt in Y_substract_mean.optimizer.values
        ]

        ax.plot(X_traj.isel(problem=i), color=OPTIMIZER_COLORS[optimizer])
        ax.set_title(f'#{i}')
        ax.set_xticks([])

        # [left, bottom, width, height]
        bar_ax = ax.inset_axes([0, -0.3, 1, 0.3])
        bar_ax.bar(
            range(Y_substract_mean.optimizer.size), Y_substract_mean.isel(
                problem=i),
            color=bar_colors,
            edgecolor=[OPTIMIZER_COLORS[opt]
                       for opt in Y_substract_mean.optimizer.values],
            linewidth=1.5)

        bar_ax.set_xticks([])
        bar_ax.set_yticks([])

    fig.tight_layout()

    return fig, axes


def plot_quadric_curve(y: np.ndarray, coeffs, color: str,
                       ax: Optional[plt.Axes] = None, res: int = 500):
    """
    Plots the original points and the fitted quadratic curve.

    Args:
    - points: numpy array of shape (n, 2), the data points.
    - coeffs: numpy array of shape (6,), the coefficients of the quadric curve.
    """
    if ax is None:
        _, ax = plt.subplots()

    A, B, C, D, E, F = coeffs
    x = np.linspace(0., 1., len(y))
    y = y

    # Plot the original points
    scatter = ax.scatter(x, y, color=color, label='Data Points', s=3, zorder=2)

    # Create a grid for x and y values
    x_range = np.linspace(x.min(), x.max(), res)
    y_range = np.linspace(y.min(), y.max(), res)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate Z values for the quadratic curve equation
    Z = A * X**2 + B * X * Y + C * Y**2 + D * X + E * Y + F

    # Custom colormap: white for negative, black near 0, white for positive
    colors = [(1, 1, 1), (0, 0, 0), (1, 1, 1)]  # RGB: white -> black -> white
    custom_cmap = LinearSegmentedColormap.from_list(
        'white_black_white', colors)
    norm = SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-0.3, vmax=0.3)

    # Plot filled contour with custom colormap
    _ = ax.contourf(X, Y, Z, levels=100,
                    cmap=custom_cmap, norm=norm, zorder=1, alpha=0.8)

    # Add the fitted curve as a contour line
    ax.contour(X, Y, Z, levels=[0], colors='k', linewidths=2, zorder=3)

    # Add a proxy artist for the contour to the legend
    contour_proxy = plt.Line2D(
        [0], [0], color='k', linewidth=2, label='Quadric surface')

    # Add text for coefficients
    coeffs_text = (f"Fitted Coefficients:\nA = {A:.3f}\nB = {B:.3f}\nC = "
                   f"{C:.3f}\nD = {D:.3f}\nE = {E:.3f}\nF = {F:.3f}")
    ax.text(
        0.05, 0.95, coeffs_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_xlabel('normalized function evaluations (x)')
    ax.set_ylabel('performance value (y)')
    ax.legend(handles=[contour_proxy, scatter], loc='best')
    ax.grid()
