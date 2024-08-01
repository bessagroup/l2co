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

from typing import List, Optional, Tuple

# Third-party
import matplotlib.pyplot as plt
import xarray as xr

# Local
from l2co._src.data_loader import DataLoader

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


#                                                                 Aspect ratios
# =============================================================================


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
                     strategies: Optional[List[xr.DataArray]] = None
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

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes

    Notes
    -----
    If no axes are provided, a new figure is created.
    """
    _rtd = d.rtd_profile(strategies).to_array('optimizer').mean('problem')

    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=set_size(width='paper', subplots=(1, 1)))
    else:
        fig = ax.get_figure()

    for optimizer in _rtd.optimizer:
        ax.plot((1-_rtd).sel(optimizer=optimizer), label=optimizer.values)

    ax.legend()
    ax.set_xlabel('Function evaluations')
    ax.set_ylabel('Fraction of function, target pairs')
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def plot_perf_profile(
    d: DataLoader, ax: Optional[plt.Axes] = None,
    strategies: Optional[List[xr.DataArray]] = None
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
    _perf = d.performance_profile(strategies).to_array('optimizer')
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
