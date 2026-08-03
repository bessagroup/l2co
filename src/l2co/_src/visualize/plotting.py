"""
Plotting module
"""
#                                                                       Modules
# =============================================================================

# Standard
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

# Third-party
import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from bessaplots import FigureGrid, TypstReport, savefig
from l2co_tasks import Task
from matplotlib.colors import (
    ListedColormap,
    LogNorm,
)
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from sklearn.metrics import r2_score

# Local
from l2co._src.batching import BatchState
from l2co._src.data import DataLoader
from l2co._src.naming import (
    BUDGET,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OUTPUT_DIM,
    OUTPUT_MIN,
    PROBLEM,
    SCHEDULE,
    UPDATE_STEP,
)
from l2co._src.optimizer_schedule import OptimizationStep
from l2co._src.performance import EPSILON
from l2co._src.sampler import grid_sampling
from l2co._src.update_class import evaluate
from l2co._src.visualize.colors import display_name, get_schedule_styles

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                                 Aspect ratios
# =============================================================================

# 1. Base widths from your IEEEtran.cls / Logs
# 3.486 inches
SINGLE_COL_WIDTH = 252.0 / 72.27
# 7.139 inches
DOUBLE_COL_WIDTH = 516.0 / 72.27


def set_size(
    fig: plt.Figure, n_side_by_side=1, span_columns=False, height: float = 0.8
):
    """
    Set figure size to fit in IEEE column layout.

    Parameters:
    - fig: The matplotlib figure object.
    - n_side_by_side: How many figures will sit horizontally (1, 2, or 3).
    - span_columns: False for single column (3.48"), True for full page (7.14")
    """

    total_width = DOUBLE_COL_WIDTH if span_columns else SINGLE_COL_WIDTH

    # 2. Account for LaTeX margins/gutters between subfigures
    # We use a 2% "safety margin" so LaTeX doesn't force a line break
    available_fraction = 0.98 / n_side_by_side
    target_width = total_width * available_fraction

    # 3. Set Height (Golden ratio is standard, but 0.8 is great for ERTD)
    target_height = target_width * height

    # 4. Apply dimensions and force 8pt font
    fig.set_size_inches(target_width, target_height)
    return fig


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
        width = 252.0 / 72.27
        # ~2.78 inches
        height = width * 0.8
        # fig, ax = plt.subplots(
        #     1, 1, figsize=figsize or set_size(width="paper", subplots=(1, 1))
        # )
        fig, ax = plt.subplots(1, 1, figsize=figsize or (width, height))
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
    xlim: tuple[float, float] = (10**-1, 10**5),
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
    title_text: str, optional
        Title text to display above the plot, by default None
    top_left_text: str, optional
        Text to display in the top-left corner (e.g., "5-D"), by default None
    show_legend: bool, optional
        Whether to display the legend on the right side, by default True
    xlim: tuple[float, float], optional
        Limits for the x-axis, by default (10**-1, 10**5)

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
        xlim=xlim,
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
    xlim: tuple[float, float] = (10**-1, 10**5),
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
    xlim : tuple[float, float], optional
        Limits for the x-axis, by default (10**-1, 10**5).

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
        # marker = style[schedule].marker
        marker = None
        pretty_label = display_name(schedule)
        y = ertd.sel({SCHEDULE: schedule})

        # FAILSAFE: Check if the entire series is NaN
        if y.isnull().all():
            # Optional: Add a log message or print here if you want to know
            # which failed
            continue

        ax.plot(
            x,
            y,
            label=pretty_label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=0.2,
            linewidth=1.2,
        )

    # --- Labels and Optional Annotations ---
    if title_text:
        ax.set_title(title_text)

    if top_left_text:
        # Places text like '5-D' in the top left corner of the plot area
        ax.text(
            0.02,
            0.98,
            top_left_text,
            transform=ax.transAxes,
            verticalalignment="top",
        )

    match cost_metric:
        case "iterations":
            # ax.set_xlabel("Iterations / dimension")
            ax.set_xlabel("iterations / dim")
        case "fevals":
            # ax.set_xlabel("Function evaluations / dimension")
            ax.set_xlabel("f-evals / dim")
        case _:
            raise ValueError(
                f"Invalid cost_metric: {cost_metric}. "
                f"Choose 'fevals' or 'iterations'."
            )

    # ax.set_ylabel("Fraction of function, target pairs")
    ax.set_ylabel("Fraction of targets")

    # --- Legend and Limits ---
    if show_legend:
        # loc="lower right" is usually best for ERTD plots as the curves
        # go from bottom-left to top-right.
        ax.legend(
            # Position inside the axes
            loc="best",
            # Enable the box
            frameon=True,
            # IEEE style usually prefers square corners
            fancybox=False,
            # Explicit frame color
            edgecolor="#BBBBBB",
            # Make it opaque so grid lines don't bleed
            framealpha=1.0,
            # Tighten vertical space between entries
            labelspacing=0.2,
            # Distance from the axis edge
            borderaxespad=0.5,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(*xlim)

    return fig, ax


def plot_r2(y_true: xr.DataArray, y_pred: xr.DataArray):
    """
    Plots Predicted vs True values for each label in a row.

    Parameters:
    y_true (xr.DataArray): True values, shape (sample, labels)
    y_pred (xr.DataArray): Predicted values, shape (sample, labels)
    """
    labels = y_true["label"].values
    n_labels = len(labels)

    # Create the figure and axes
    fig, axes = plt.subplots(
        1, n_labels, figsize=(5 * n_labels, 5), sharey=False
    )

    # Ensure axes is an array even if there is only 1 plot
    if n_labels == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        ax = axes[i]

        # Select data for the specific label
        # .values converts the DataArray slice to a numpy array for calculation
        y_t = y_true.sel({"label": label}).values
        y_p = y_pred.sel({"label": label}).values

        # 1. Calculate R2 score
        r2 = r2_score(y_t, y_p)

        # Scatter plot
        ax.scatter(y_t, y_p, alpha=0.5, s=20)

        # Define the limits for the diagonal line
        all_vals = np.concatenate([y_t, y_p])
        low, high = np.min(all_vals), np.max(all_vals)

        # Draw the 1:1 diagonal line (representing perfect R2)
        ax.plot(
            [low, high],
            [low, high],
            color="red",
            linestyle="--",
            label="1:1 Line",
        )

        # 4. Add the R2 text box in the top left
        text_str = f"$R^2 = {r2:.3f}$"
        # bbox creates the box styling
        props = dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor="silver"
        )

        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )

        # Formatting
        ax.set_title(f"Optimizer: {label}")
        ax.set_xlabel("True Values")
        if i == 0:
            ax.set_ylabel("Predicted Values")
        ax.grid(True, linestyle=":", alpha=0.7)

    return fig, axes


def plot_single_r2(
    y_true: xr.DataArray, y_pred: xr.DataArray, label: str, ax=None
):
    """
    Plots Predicted vs True values for a SINGLE label.
    Designed for use within a larger subfigure grid.
    """
    if ax is None:
        # Fallback if called standalone (using the 3-across width)
        fig, ax = plt.subplots(figsize=(3.486 / 3, 3.486 / 3))
    else:
        fig = ax.get_figure()

    # Select data for the specific label
    y_t = y_true.sel({"label": label}).values
    y_p = y_pred.sel({"label": label}).values

    # Calculate R2 score
    r2 = r2_score(y_t, y_p)

    # Scatter plot - smaller 's' and lower alpha for dense subfigures
    ax.scatter(y_t, y_p, s=3, edgecolors="none", color="k", rasterized=True)

    # Define the limits for the diagonal line
    all_vals = np.concatenate([y_t, y_p])
    low, high = np.min(all_vals), np.max(all_vals)

    # Draw the 1:1 diagonal line
    ax.plot(
        [low, high], [low, high], color="#EE6677", linestyle="--", linewidth=1
    )

    # Add the R2 text box (Font size 7 for IEEE subfigures)
    text_str = f"$R^2 = {r2:.3f}$"
    props = dict(
        boxstyle="square,pad=0.2",
        facecolor="white",
        alpha=0.9,
        edgecolor="silver",
    )

    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("True labels")
    ax.set_ylabel("Predicted labels")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(True, linestyle=":", alpha=0.6)
    # ax.tick_params(axis="both", which="major", labelsize=7)

    return fig, ax


def plot_metrics(metrics: dict[str, Iterable[float]]):
    """Plot training and validation loss curves.

    Parameters
    ----------
    metrics : dict[str, Iterable[float]]
        Must contain ``'train_loss'`` and ``'val_loss'`` keys.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes with log-scaled y-axis.
    """
    fig, ax = _get_or_create_axes()
    ax.plot(metrics["train_loss"], label="Train loss", color="#000000")
    ax.plot(metrics["val_loss"], label="Validation loss", color="#CC3311")

    ax.set_xlabel("Epochs")
    ax.set_xticklabels([])

    ax.set_ylabel("")
    ax.set_yscale("log")
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((-2, -2))
    # ax.yaxis.set_major_formatter(formatter)

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        framealpha=1.0,
        labelspacing=0.2,
        borderaxespad=0.5,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.3,
    )
    return fig, ax


def plot_update_step(
    d: DataLoader,
    schedule: str,
    optimizers: Iterable[OptimizationStep],
    cost_metric: Literal["fevals", "iterations"] | None = None,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    hide_when_done: bool = False,
    log_x: bool = False,
    xlim: tuple[float, float] | None = None,
):
    """Plot a colour-coded raster of optimizer selections.

    Each row represents a problem and each column a budget
    step, coloured by which optimizer was active. Budget cells
    that were never written by the rollout (the padding region
    after a problem terminates early) are rendered in white so
    that the per-optimizer colour mapping is independent of the
    order of ``optimizers``; white always wins over the
    ``hide_when_done`` black overlay, so the two sentinels stay
    distinguishable (black = solved from that step on, white =
    no data). Both sentinels get their own legend entries when
    they can appear.

    Parameters
    ----------
    d : DataLoader
        Loaded optimisation data.
    schedule : str
        Schedule name to visualise.
    optimizers : Iterable[OptimizationStep]
        Optimizers whose hashes map to colours.
    cost_metric : {'fevals', 'iterations'} or None, optional
        X-axis metric. ``None`` uses raw step index.
    ax : plt.Axes or None, optional
        Existing axes. Creates new figure when ``None``.
    show_legend : bool, optional
        Whether to display a legend, by default True.
    hide_when_done : bool, optional
        When ``True``, every step from the first one whose running
        ``OUTPUT_MIN`` reaches within ``EPSILON`` of the task's
        ``global_min`` onwards is rendered in black instead of the
        optimizer colour, indicating that the problem was already
        solved and subsequent optimizer choices are immaterial.
        Default is ``False``.
    log_x : bool, optional
        Render the x-axis on a logarithmic scale. Only meaningful
        together with a ``cost_metric``; the raw step index starts at
        zero, which a log axis would clip. Default is ``False``.
    xlim : tuple[float, float] or None, optional
        Explicit x-axis limits; ``None`` (default) keeps matplotlib's
        autoscaling. Pass :func:`plot_ertd`'s ``xlim`` together with
        ``log_x=True`` and the same ``cost_metric`` to align this
        raster with an ERTD profile of the same data.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes.
    """
    dims = d.task_labels.sel({FEATURE: "dimensionality"}, drop=True).astype(
        int
    )
    update_step = (
        d.data[UPDATE_STEP].isel({OUTPUT_DIM: 0}).sel({SCHEDULE: schedule})
    )
    if hide_when_done:
        output_min = (
            d.data[OUTPUT_MIN].isel({OUTPUT_DIM: 0}).sel({SCHEDULE: schedule})
        )
        global_min = d.task_labels.sel(
            {FEATURE: GLOBAL_MIN}, drop=True
        ).astype(float)
    else:
        output_min = None
        global_min = None
    match cost_metric:
        case "fevals":
            cost_metric = (
                d.data[FEVALS].isel({OUTPUT_DIM: 0}).sel({SCHEDULE: schedule})
                / dims
            )
            x_label = "f-evals / dim"
        case "iterations":
            cost_metric = (
                d.data[ITERATIONS]
                .isel({OUTPUT_DIM: 0})
                .sel({SCHEDULE: schedule})
                / dims
            )
            x_label = "iterations / dim"
        case None:
            cost_metric = None
            x_label = "Steps"
        case _:
            raise ValueError(f"Invalid cost_metric: {cost_metric}. ")

    # 1. Align Dimensions
    # We want (Problems, Steps) for the plot
    # If your xarray is (Steps, Problems), we transpose it
    if update_step.dims[0] != "problem":
        update_step = update_step.transpose("problem", ...)
        if cost_metric is not None:
            cost_metric = cost_metric.transpose("problem", ...)
        if output_min is not None:
            output_min = output_min.transpose("problem", ...)

    # 2. Prepare Mappings
    opt_hashes = [opt.hash for opt in optimizers]
    styles = get_schedule_styles([opt.name for opt in optimizers])
    hex_colors = [styles[opt.name].color for opt in optimizers]
    display_names = [display_name(opt.name) for opt in optimizers]

    # 3. Reclassification
    # Cells of `update_step` that were never written by the rollout (the
    # padding region after a problem terminates early) remain at the int32
    # zero used by `HistoryState.init`. Optimizer hashes are 6-hex sha256
    # prefixes, so 0 cannot collide with a real hash; we use -1 here as
    # an unambiguous "no optimizer" sentinel and mask those cells when
    # plotting so they render in a fixed background colour rather than
    # silently inheriting whichever optimizer happens to sit at index 0.
    z_data = update_step.astype(int).values
    indexed_data = np.full(z_data.shape, -1, dtype=int)
    for i, opt_hash in enumerate(opt_hashes):
        indexed_data[z_data == opt_hash] = i

    # Cells that match no optimizer hash (the never-written padding)
    # render via cmap.set_bad; computed before the done overpaint below
    # so "no data" always wins over "solved".
    unmatched_mask = indexed_data == -1

    # 3b. Optionally mark steps where the task is already solved.
    # We append black as an extra colormap entry at index len(optimizers).
    done_index = len(opt_hashes)
    if hide_when_done:
        # Take the running minimum along the budget axis so that once the
        # global minimum is first hit, every subsequent step stays black.
        # `np.fmin.accumulate` carries the running minimum *through* NaN
        # cells (fmin prefers the non-NaN operand), so the padding region
        # of a solved problem would also satisfy the comparison -- the
        # `~unmatched_mask` term keeps those cells white instead.
        running_min = xr.apply_ufunc(
            np.fmin.accumulate,
            output_min,
            input_core_dims=[[BUDGET]],
            output_core_dims=[[BUDGET]],
            kwargs={"axis": -1},
        )
        done_mask = (running_min <= global_min + EPSILON).values
        indexed_data[done_mask & ~unmatched_mask] = done_index
        cmap_colors = list(hex_colors) + ["black"]
    else:
        cmap_colors = list(hex_colors)

    # Mask unmatched cells so they render via cmap.set_bad rather than
    # being mapped to whichever optimizer sits at index 0.
    indexed_data = np.ma.array(indexed_data, mask=unmatched_mask)

    n_problems, n_steps = indexed_data.shape

    # 4. Coordinate Logic
    if cost_metric is not None:
        X = cost_metric.values.astype(float)
        # Forward-fill NaN along the BUDGET axis. NaNs appear when
        # `d.data` was assembled from problems with different BUDGET
        # sizes (xarray's outer-join concat pads short rows with NaN
        # in FEVALS / ITERATIONS, mirroring the same padding that
        # produces unmatched UPDATE_STEP cells above). Without this
        # fill, the midpoint cell edges computed below would be NaN
        # around the last visible cell and it would fail to render —
        # the Z mask only suppresses the masked cell, not its X-coord
        # influence on neighbours. Carry-forward collapses the last
        # visible cell to zero width but keeps every drawn cell
        # well-defined.
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_idx = np.where(~nan_mask, np.arange(X.shape[1])[None, :], 0)
            col_idx = np.maximum.accumulate(col_idx, axis=1)
            row_idx = np.arange(X.shape[0])[:, None]
            X = X[row_idx, col_idx]
            # Rows that are entirely NaN (a problem missing from this
            # schedule) have nothing meaningful to plot; fill with 0
            # so the per-row edges below stay finite (the fully masked
            # row is skipped when drawing anyway).
            X = np.nan_to_num(X, nan=0.0)
    else:
        X = None

    # 5. Plotting
    fig, ax = _get_or_create_axes(ax=ax)
    cmap = ListedColormap(cmap_colors)
    cmap.set_bad(color="white")

    # Pin vmin/vmax so the colormap mapping is stable regardless of
    # which indices actually appear in Z (e.g. no "done" cells, or
    # an optimizer that was never selected).
    mesh_kwargs = {
        "cmap": cmap,
        "edgecolors": "none",
        "vmin": 0,
        "vmax": len(cmap_colors) - 1,
    }
    if X is not None:
        # Every problem row has its own cost coordinates (optimizers
        # differ in fevals/iterations per step, and rows terminate at
        # different budgets), so the raster cannot be a single 2-D
        # pcolormesh: a QuadMesh shares each cell corner between
        # vertically adjacent cells and interpolates it from *both*
        # rows' centers, which slants every cell edge wherever
        # neighbouring rows disagree — the raster fills with diagonal
        # artifacts. Draw one flat-shaded single-row mesh per problem
        # instead, so every cell stays an axis-aligned rectangle on
        # that row's own cost grid.
        for i in range(n_problems):
            row = indexed_data[i : i + 1]
            if np.ma.getmaskarray(row).all():
                continue
            centers = X[i]
            edges = np.empty(n_steps + 1)
            if n_steps > 1:
                edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
                edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
                edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
            else:
                edges[:] = (centers[0] - 0.5, centers[0] + 0.5)
            if log_x:
                # A nonpositive left edge would invalidate the first
                # cell once the axis goes logarithmic.
                edges = np.maximum(edges, 0.5 * centers[0])
            ax.pcolormesh(
                edges,
                [i - 0.5, i + 0.5],
                row,
                shading="flat",
                **mesh_kwargs,
            )
        # Skipped (fully masked) rows draw nothing; pin the limits so
        # they still occupy their slot instead of shrinking the axis.
        ax.set_ylim(-0.5, n_problems - 0.5)
    else:
        # Raw step index: one shared grid for all rows, a single mesh.
        X_grid, Y_grid = np.meshgrid(
            np.linspace(0.0, 1.0, n_steps), np.arange(n_problems)
        )
        ax.pcolormesh(
            X_grid,
            Y_grid,
            indexed_data,
            shading="nearest",
            **mesh_kwargs,
        )

    # 6. IEEE Style Legend
    if show_legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=name,
                markerfacecolor=color,
                markersize=8,
                linestyle="None",
            )
            for color, name in zip(hex_colors, display_names, strict=True)
        ]
        if hide_when_done:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label="Global min. reached",
                    markerfacecolor="black",
                    markersize=8,
                    linestyle="None",
                )
            )
        if unmatched_mask.any():
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label="No data",
                    markerfacecolor="white",
                    markeredgecolor="#BBBBBB",
                    markersize=8,
                    linestyle="None",
                )
            )
        # The raster fills the entire axes, so an inside legend would
        # always cover data rows; anchor it above the axes instead.
        # Savers that use bbox_inches="tight" (e.g. f3dasm's
        # figure_store) grow the saved figure to include it.
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncols=3,
            frameon=True,
            fancybox=False,
            edgecolor="#BBBBBB",
            framealpha=1.0,
            labelspacing=0.2,
            borderaxespad=0.0,
            handletextpad=0.4,
            columnspacing=1.0,
            fontsize="small",
        )

    # 7. Scientific Polish
    if log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylabel("Problems")
    ax.set_xlabel(x_label)

    # Clean up Y-axis
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.invert_yaxis()

    return fig, ax


# =============================================================================
#                                              Grouped ERTD plotting utilities
# =============================================================================


def _add_eval_line(ax, line_val):
    if line_val is not None:
        ax.axvline(
            x=line_val,
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=5,
        )


def _save_and_collect(
    fig: plt.Figure,
    name: str,
    caption: str,
    save_dir: Path,
    figures: list,
    paths: list,
    captions: list,
    grid: FigureGrid | None = None,
    first_column: bool = True,
    last_row: bool = True,
    **savefig_kwargs,
):
    """Save a figure and append to collection lists.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    name : str
        Base file name (without extension).
    caption : str
        Caption text for report generation.
    save_dir : Path
        Directory to write the file to.
    figures : list
        Accumulator for figure objects.
    paths : list
        Accumulator for saved file paths.
    captions : list
        Accumulator for caption strings.
    grid : FigureGrid or None, optional
        When provided, sizing and saving are handled by
        :class:`~bessaplots.FigureGrid` instead of
        :func:`~bessaplots.savefig`.
    first_column : bool, optional
        Whether this is the first figure in a row.
    last_row : bool, optional
        Whether this figure is in the bottom row of the grid.
    **savefig_kwargs
        Forwarded to :func:`~bessaplots.savefig` when
        *grid* is ``None``.
    """
    save_path = save_dir / f"{name}.pdf"
    if grid is not None:
        fmt = savefig_kwargs.get("format", "pgf")
        grid.save(
            fig,
            str(save_path),
            first_column=first_column,
            last_row=last_row,
            format=fmt,
        )
    else:
        savefig(fig=fig, path=str(save_path), **savefig_kwargs)
    paths.append(save_path.resolve())
    figures.append(fig)
    captions.append(caption)
    plt.close(fig)


def ertd_per_group(
    d,
    groups: list[tuple[dict, str]],
    plot_actions: str | None = None,
    optimizers=None,
    line: float | None = None,
    save_dir: Path | None = None,
    n_side_by_side: int = 4,
    height: float = 1.0,
    sharey: bool = False,
    sharex: bool = False,
    **savefig_kwargs,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot and save ERTD profiles for each group.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing optimization results.
    groups : list[tuple[dict, str]]
        Each entry is (filter_dict, name) where filter_dict is
        passed to ``d.select_feature`` and name is used for
        file naming and captions.
    plot_actions : str or None, optional
        If not None, also plot actions for the specified schedule.
    optimizers : optional
        Optimizers to pass to ``plot_update_step``.
    line : float or None, optional
        X-coordinate for a vertical reference line.
    save_dir : Path or None, optional
        Directory to save plots to. Defaults to cwd.
    n_side_by_side : int, optional
        Number of figures per row. The default is ``4``.
    height : float, optional
        Height of each figure in the grid. The default is ``1.0``.
    sharey : bool, optional
        When ``True``, only the first figure in each row keeps its
        y-axis label and tick labels.  Default is ``False``.
    sharex : bool, optional
        When ``True``, only the bottom row keeps its x-axis label
        and tick labels.  Default is ``False``.

    Returns
    -------
    tuple[list[Figure], list[Path], list[str]]
        List of figures, list of absolute paths, list of captions.
    """
    save_dir = Path(save_dir) if save_dir is not None else Path.cwd()
    figures, paths, captions = [], [], []

    if sharey or sharex:
        fig_grid = FigureGrid(
            n_figures=n_side_by_side,
            paper_size=savefig_kwargs.get("paper_size", "letter"),
            span_columns=True,
            height=height,
            sharey=sharey,
            sharex=sharex,
        )
    else:
        fig_grid = None

    n_groups = len(groups)
    n_rows = math.ceil(n_groups / n_side_by_side)

    ertd_idx = 0
    for filter_dict, name in groups:
        d_group = d.select_feature(filter_dict)

        fig, ax = plot_ertd_profile(d_group, show_legend=False)
        _add_eval_line(ax, line)

        row_idx = ertd_idx // n_side_by_side
        is_first_col = ertd_idx % n_side_by_side == 0
        is_last_row = row_idx == n_rows - 1

        _save_and_collect(
            fig,
            f"rtd_{name}",
            f"{name}",
            save_dir,
            figures,
            paths,
            captions,
            grid=fig_grid,
            first_column=is_first_col,
            last_row=is_last_row,
            span_columns=True,
            height=height,
            n_side_by_side=n_side_by_side,
            **savefig_kwargs,
        )
        ertd_idx += 1

        if plot_actions is not None:
            fig_act, ax_act = plot_update_step(
                d_group,
                schedule=plot_actions,
                optimizers=optimizers,
                cost_metric="iterations",
                show_legend=False,
            )

            _save_and_collect(
                fig_act,
                f"actions_{plot_actions}_{name}",
                f"Actions {plot_actions}: {name}",
                save_dir,
                figures,
                paths,
                captions,
                span_columns=True,
                height=height,
                n_side_by_side=n_side_by_side,
                **savefig_kwargs,
            )

    return figures, paths, captions


def generate_report(
    paths: list[str | Path],
    captions: list[str],
    title: str = "",
    author: str = "",
    columns: int = 2,
    save_path: str | Path = "report",
) -> Path:
    """Generate a Typst PDF report from a grid of figures.

    Parameters
    ----------
    paths : list of str or Path
        Paths to figure files (e.g. PDF or PNG).
    captions : list of str
        Per-figure captions. Must have the same length as *paths*.
    title : str, optional
        Report title. The default is ``""``.
    author : str, optional
        Author name. The default is ``""``.
    columns : int, optional
        Number of grid columns. The default is ``2``.
    save_path : str or Path, optional
        Output path stem (without extension). The default is
        ``"report"``.

    Returns
    -------
    Path
        Path to the compiled PDF.
    """
    report = TypstReport(title=title, author=author)
    report.add_figures(paths, columns=columns, subcaptions=captions)
    return report.save(save_path)
