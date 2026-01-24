"""
Plotting module
"""
#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Iterable
from typing import Literal

# Third-party
import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import (
    LogNorm,
)
from scipy.interpolate import griddata

from l2co._src.batching import BatchState

# Local
from l2co._src.data_loader import DataLoader
from l2co._src.naming import (
    PROBLEM,
    SCHEDULE,
)
from l2co._src.sampler import grid_sampling
from l2co._src.task import Task
from l2co._src.update_class import evaluate
from l2co._src.visualize.colors import get_schedule_styles

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
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
    if width == "thesis":
        width_pt = 426.79135
    elif width == "paper":
        width_pt = 345.0
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


#                                                           Helper Functions
# =============================================================================


def _get_or_create_axes(
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Get or create a matplotlib axes object.

    Parameters
    ----------
    ax : plt.Axes, optional
        Existing matplotlib axes. If None, creates new figure and axes.
    figsize : tuple[float, float], optional
        Figure size for new figure creation. If None and ax is None,
        uses default size.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=figsize or set_size(width="paper", subplots=(1, 1))
        )
    else:
        fig = ax.get_figure()
    return fig, ax


def _prepare_loss_landscape(
    task: Task,
    stepsize: float = 0.033,
    seed: int = 0,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare loss landscape by grid sampling and evaluation.

    Parameters
    ----------
    task : Task
        The optimization task containing the model and evaluation function.
    stepsize : float, optional
        Step size for grid sampling, by default 0.033.
    seed : int, optional
        Random seed for evaluation, by default 0.
    lower_bound : float, optional
        Lower bound for sampling, by default 0.0.
    upper_bound : float, optional
        Upper bound for sampling, by default 1.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        samples, losses arrays from evaluated samples.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)
    samples = grid_sampling(
        params=params,
        n_points_per_dim=int(1 // stepsize),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    key = jr.key(seed)
    n_samples = jax.tree_util.tree_leaves(samples)[0].shape[0]

    dataset = task.loaded_dataset
    batch_state = BatchState.init(
        dataset=dataset, batch_size=task.batch_size, key=key
    )

    losses = evaluate(
        samples,
        batch_state,
        dataset,
        task.loss_fn,
        task.pass_rng,
        key,
        jr.split(key, n_samples),
    )

    return samples, losses


def _interpolate_2d_grid_from_scattered(
    x: np.ndarray,
    y: np.ndarray,
    losses: np.ndarray,
    plot_resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a 2D grid from scattered data points.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of sample points.
    y : np.ndarray
        Y coordinates of sample points.
    losses : np.ndarray
        Loss values at sample points.
    plot_resolution : int, optional
        Resolution of interpolated grid, by default 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Interpolated X, Y meshgrids and Z values (normalized for log scale).
    """
    X, Y = np.meshgrid(
        np.linspace(x.min(), x.max(), plot_resolution),
        np.linspace(y.min(), y.max(), plot_resolution),
    )

    Z = griddata((x, y), losses, (X, Y), method="cubic")
    Z -= np.nanmin(Z)
    Z += np.nanmin(Z[Z > 0])

    return X, Y, Z


def _create_positive_log_normalized_grid(
    x: np.ndarray,
    y: np.ndarray,
    losses: np.ndarray,
    plot_resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid from scattered data and apply log normalization.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of sample points.
    y : np.ndarray
        Y coordinates of sample points.
    losses : np.ndarray
        Loss values at sample points.
    plot_resolution : int, optional
        Resolution of interpolated grid, by default 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Interpolated X, Y meshgrids and Z values (normalized for log scale).
    """
    X, Y, Z = _interpolate_2d_grid_from_scattered(
        x=x, y=y, losses=losses, plot_resolution=plot_resolution
    )

    # Ensure Z is positive for log normalization
    Z_nonzero = Z[Z > 0]
    eps = np.min(Z_nonzero) if Z_nonzero.size > 0 else 1e-1

    Z += eps

    return X, Y, Z


# =============================================================================


def plot_ertd_profile(
    d: DataLoader,
    ax: plt.Axes | None = None,
    include_best: bool = False,
    optimizers_to_plot: Iterable[str] | None = None,
    cost_metric: Literal["fevals", "iterations"] = "fevals",
    title_text: str | None = None,
    top_left_text: str | None = None,
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the run-time distribution for the given DataLoader.

    Parameters
    ----------
    d: DataLoader
        DataLoader instance
    ax: plt.Axes, optional
        Matplotlib axes
    include_best: bool, optional
        Include the best strategy in the plot, by default False
    optimizers_to_plot: list of str, optional
        List of optimizers to plot, by default None
    cost_metric: str, optional
        Cost metric to use for the x-axis, either 'fevals' or 'iterations',
        by default 'fevals'

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes

    Notes
    -----
    If no axes are provided, a new figure is created.
    """
    _ertd = d.ertd_profile(include_best=include_best)

    _ertd = _ertd.sel({"metric": cost_metric}).mean(PROBLEM)

    return plot_ertd(
        ertd=_ertd,
        ax=ax,
        cost_metric=cost_metric,
        title_text=title_text,
        top_left_text=top_left_text,
        show_legend=show_legend,
    )


# =============================================================================


def plot_2d_loss_landscape_only(
    task: Task,
    plot_resolution: int = 100,
    stepsize: float = 0.033,
    seed: int = 0,
    ax=None,
    colorbar: bool = True,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
):
    """
    Plot a 2D loss landscape for a given task.

    Parameters
    ----------
    task : Task
        The optimization task containing the model and evaluation function.
    plot_resolution : int, optional
        Resolution of the interpolated grid for plotting, by default 100.
    stepsize : float, optional
        Step size for grid sampling, by default 0.033.
    seed : int, optional
        Random seed for evaluation, by default 0.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure,
        by default None.
    colorbar : bool, optional
        Whether to display a colorbar, by default True.
    lower_bound : float, optional
        Lower bound for sampling, by default 0.0.
    upper_bound : float, optional
        Upper bound for sampling, by default 1.0.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes containing the plot.

    Notes
    -----
    The function uses logarithmic normalization to better visualize loss
    landscapes with large dynamic ranges. The loss values are shifted to
    be positive before applying log normalization.
    """
    samples, losses = _prepare_loss_landscape(
        task=task,
        stepsize=stepsize,
        seed=seed,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    x = samples[:, 0]
    y = samples[:, 1]

    X, Y, Z = _create_positive_log_normalized_grid(
        x=x, y=y, losses=losses, plot_resolution=plot_resolution
    )

    # Create the figure and set up the subplots
    fig, ax = _get_or_create_axes(ax=ax, figsize=(12, 6))

    heatmap = ax.pcolormesh(
        X, Y, Z, cmap="viridis", shading="auto", norm=LogNorm()
    )
    if colorbar:
        fig.colorbar(heatmap, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def plot_2d_loss_landscape_slice_grid(
    task: Task,
    plot_resolution: int = 100,
    stepsize: float = 0.033,
    seed: int = 0,
    z_slices: int | list[float] = 6,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    colorbar: bool = True,
):
    """
    Plot a grid of 2D (x, y) loss landscape slices for fixed z values,
    annotating the GLOBAL minimum with a red cross.
    """
    samples, losses = _prepare_loss_landscape(
        task=task,
        stepsize=stepsize,
        seed=seed,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]

    # ------------------------------------------------------------------
    # GLOBAL minimum across ALL samples
    # ------------------------------------------------------------------
    global_min_idx = np.argmin(losses)
    x_min = x[global_min_idx]
    y_min = y[global_min_idx]
    z_min = z[global_min_idx]

    # Choose z slices from actual grid
    unique_z = np.unique(z)

    if isinstance(z_slices, int):
        z_indices = np.linspace(0, len(unique_z) - 1, z_slices, dtype=int)
        z_values = unique_z[z_indices]
    else:
        z_values = np.asarray(z_slices)

    n_slices = len(z_values)
    ncols = int(np.ceil(np.sqrt(n_slices)))
    nrows = int(np.ceil(n_slices / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
    )

    all_Z = []
    slice_data = []

    # First pass: compute interpolated Z
    for z_val in z_values:
        mask = z == z_val

        xs, ys, ls = x[mask], y[mask], losses[mask]

        X, Y, Z = _interpolate_2d_grid_from_scattered(
            x=xs, y=ys, losses=ls, plot_resolution=plot_resolution
        )

        slice_data.append((X, Y, Z))
        all_Z.append(Z)

    # Shared log normalization
    Z_global_min = min(np.nanmin(Z) for Z in all_Z)
    Z_global_max = max(np.nanmax(Z) for Z in all_Z)
    norm = LogNorm(vmin=Z_global_min, vmax=Z_global_max)

    im = None

    for ax, z_val, data in zip(axes.flat, z_values, slice_data, strict=False):
        X, Y, Z = data

        im = ax.pcolormesh(
            X,
            Y,
            Z,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )

        # 🔴 Plot GLOBAL minimum only in the correct slice
        if z_val == z_min:
            ax.plot(
                x_min,
                y_min,
                marker="x",
                color="red",
                markersize=12,
                markeredgewidth=2.5,
            )

        ax.set_title(f"z = {z_val:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Disable unused axes
    for ax in axes.flat[n_slices:]:
        ax.axis("off")

    if colorbar and im is not None:
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)

    fig.tight_layout()
    return fig, axes, z_values


def plot_3d_loss_landscape_only(
    task: Task,
    plot_resolution: int = 100,
    stepsize: float = 0.033,
    seed: int = 0,
    ax=None,
):
    """
    Plot a 3D loss landscape for a given task.

    Parameters
    ----------
    task : Task
        The optimization task containing the model and evaluation function.
    plot_resolution : int, optional
        Resolution of the interpolated grid for plotting, by default 100.
    stepsize : float, optional
        Step size for grid sampling, by default 0.033.
    seed : int, optional
        Random seed for evaluation, by default 0.
    ax : plt.Axes, optional
        Matplotlib 3D axes to plot on. If None, creates a new figure,
        by default None.

    Returns
    -------
    tuple[plt.Figure, plt.Axes, np.ndarray] or plt.Axes
        If ax is None, returns (figure, axes, original_Z_values).
        If ax is provided, returns the axes only.

    Notes
    -----
    The function uses logarithmic normalization to better visualize loss
    landscapes with large dynamic ranges. The loss values are shifted to
    be positive before applying log normalization.
    """
    samples, losses = _prepare_loss_landscape(
        task=task,
        stepsize=stepsize,
        seed=seed,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    x = samples[:, 0]
    y = samples[:, 1]

    X, Y, Z = _create_positive_log_normalized_grid(
        x=x, y=y, losses=losses, plot_resolution=plot_resolution
    )
    Z_orig = Z.copy()

    # If no axis is provided, create one
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        created_fig = True

    # Plot on the provided or created axis
    _ = ax.plot_surface(X, Y, Z, cmap="viridis", norm=LogNorm(), zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if created_fig:
        return fig, ax, Z_orig
    else:
        return ax


# =============================================================================


def plot_ertd(
    ertd: xr.DataArray,
    ax: plt.Axes | None = None,
    cost_metric: Literal["fevals", "iterations"] = "fevals",
    title_text: str | None = None,
    top_left_text: str | None = None,
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the expected run-time distribution for multiple optimizers.

    Parameters
    ----------
    ertd : xr.DataArray
        DataArray containing expected run-time distribution data with
        dimensions for schedule (optimizer) and a 'cost_per_dim' coordinate.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure,
        by default None.
    cost_metric : {'fevals', 'iterations'}, optional
        Cost metric to display on x-axis, by default 'fevals'.
    title_text : str, optional
        The title to display above the plot, by default None.
    top_left_text : str, optional
        Text to display in the top-left corner (e.g., "5-D"), by default None.
    show_legend : bool, optional
        Whether to display the legend on the right side, by default True.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes containing the plot.

    Notes
    -----
    The plot shows the fraction of function-target pairs solved as a function
    of normalized cost (function evaluations or iterations per dimensionality).
    The x-axis uses logarithmic scaling. Colors and line styles are determined
    by the OPTIMIZER_COLORS and OPTIMIZER_LINESTYLES dictionaries.
    """
    fig, ax = _get_or_create_axes(ax=ax)

    # --- BBOB/COCO Style Grid Configuration ---
    ax.set_xscale("log")
    # Major grid: solid lines
    ax.grid(
        True,
        which="major",
        color="gray",
        linestyle="-",
        linewidth=0.5,
        alpha=0.7,
    )
    # Minor grid: dotted lines for logarithmic increments
    ax.grid(
        True,
        which="minor",
        color="gray",
        linestyle=":",
        linewidth=0.3,
        alpha=0.5,
    )

    # Ensure ticks point inward and appear on all sides
    ax.tick_params(
        axis="both", which="both", direction="in", top=True, right=True
    )

    x = ertd["cost_per_dim"]

    style = get_schedule_styles(ertd[SCHEDULE].values)

    for schedule in ertd[SCHEDULE].values:
        color = style[schedule].color
        linestyle = style[schedule].linestyle
        marker = style[schedule].marker
        y = ertd.sel({SCHEDULE: schedule})

        # FAILSAFE: Check if the entire series is NaN
        if y.isnull().all():
            # Optional: Add a log message or print here if you want to know
            # which failed
            continue

        ax.plot(
            x,
            y,
            label=schedule,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=0.1,
            linewidth=1.2,
        )

    # --- Labels and Optional Annotations ---
    if title_text:
        ax.set_title(title_text, fontsize=16)

    if top_left_text:
        # Places text like '5-D' in the top left corner of the plot area
        ax.text(
            0.02,
            0.98,
            top_left_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

    match cost_metric:
        case "iterations":
            ax.set_xlabel("Iterations / dimension")
        case "fevals":
            ax.set_xlabel("Number of function evaluations / dimension")
        case _:
            raise ValueError(
                f"Invalid cost_metric: {cost_metric}. "
                "Choose 'fevals' or 'iterations'."
            )

    ax.set_ylabel("Fraction of function, target pairs")

    # --- Legend and Limits ---
    if show_legend:
        # Legend is placed outside to the right to match the provided image
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            frameon=False,
            labelspacing=0.1,
            borderaxespad=0.0,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.1)  # Starts the x-axis at 10^-1

    return fig, ax
