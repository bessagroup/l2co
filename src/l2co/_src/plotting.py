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

# Standard
import matplotlib.animation as animation
import math
from typing import Iterable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
# Third-party
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from f3dasm import ExperimentData
from f3dasm.design import Domain
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from scipy.interpolate import griddata

# Local
from .data_loader import DataLoader, OptimizationDynamics
from .naming import (BUDGET, INPUT_DIM, OUTPUT_DIM, PROBLEM, REALIZATION,
                     SCHEDULE)
from .sampler import grid_sampling
from .task import Task
from .tree_utils import experimentdata_to_xarray
from .update_state import evaluate

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
        SCHEDULE).mean(PROBLEM).isel({OUTPUT_DIM: 0})

    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=set_size(width='paper', subplots=(1, 1)))
    else:
        fig = ax.get_figure()

    if optimizers_to_plot is None:
        for optimizer in _rtd[SCHEDULE]:
            ax.plot(_rtd.sel({SCHEDULE: optimizer}), label=optimizer.values)

    else:
        for optimizer in optimizers_to_plot:
            ax.plot(_rtd.sel({SCHEDULE: optimizer}), label=optimizer)

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

    for optimizer in _perf[SCHEDULE]:
        ax.plot(rel_error, _perf.sel({SCHEDULE: optimizer}),
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
        {SCHEDULE: 0}).T
    Y = optimization_dynamics._Y.isel({OUTPUT_DIM: 0}).T

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

        ax.plot(X_traj.isel({PROBLEM: i}), color=OPTIMIZER_COLORS[optimizer])
        ax.set_title(f'#{i}')
        ax.set_xticks([])

        # [left, bottom, width, height]
        bar_ax = ax.inset_axes([0, -0.3, 1, 0.3])
        bar_ax.bar(
            range(Y_substract_mean.optimizer.size), Y_substract_mean.isel(
                {PROBLEM: i}),
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

# =============================================================================


def format_plot(ax: plt.Axes, xlim_right: int, ylabel: str,
                xlabel: str, title: str):
    """
    Format a plot with consistent styling.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to format.
    xlim_right (float): The right limit for the x-axis.
    ylabel (str): The label for the y-axis.
    xlabel (str): The label for the x-axis.
    title (str): The title of the plot.
    """
    ax.set_ylim(0, 1)
    ax.set_xlim(0, xlim_right)  # Ensure x-axis starts at 0

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=14)

    # Increase font size of tick labels
    ax.tick_params(axis='both', labelsize=12)

    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    # Set x-axis grid every 100
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    # Set y-axis grid every 0.1
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Modify the existing legend if present
    legend = ax.get_legend()
    if legend:
        custom_labels = {
            "adam_learning_rate=0.001": "Adam (lr=1e-3)",
            "adam_learning_rate=0.1": "Adam (lr=1e-1)",
            "pso_popsize=30": "PSO",
            "cmaes": "CMA-ES",
            "lbfgsb": "L-BFGS-B",
            "sbs": "Single Best Solver",
            "l2co": "L2CO",
        }

        legend.set_frame_on(True)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_facecolor('white')
        legend.get_title().set_fontsize(14)
        for text in legend.get_texts():
            old_label = text.get_text()
            new_label = custom_labels.get(
                old_label, old_label)  # Replace if found
            text.set_text(new_label)
            text.set_fontsize(12)


def plot_2d_loss_landscape(task: Task, plot_resolution: int = 100,
                           stepsize: float = 0.033, seed: int = 0):
    sampler = GridWeightSampler(
        stepsize=stepsize, lower_bound=0.0, upper_bound=1.0)

    evaluate_block = EvaluateModel(
        task=task,
        seed=seed,
    )

    domain = Domain()
    domain.add_parameter('x')
    domain.add_output('y')
    domain.add_output('opt_step')
    data = ExperimentData(domain=domain)

    data = sampler.call(data=data, model=task.model)
    data = evaluate_block.call(data=data)
    data = experimentdata_to_xarray(data=data, popsizes=None, budgets=None,
                                    include_input=True, n_iterations=None)

    x = data['input'].isel({INPUT_DIM: 0}).values
    y = data['input'].isel({INPUT_DIM: 1}).values
    z = data['output'].isel({OUTPUT_DIM: 0}).values
    x.shape, y.shape, z.shape

    import numpy as np
    from scipy.interpolate import griddata

    X, Y = np.meshgrid(np.linspace(min(x), max(x), plot_resolution),
                       np.linspace(min(y), max(y), plot_resolution))

    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Ensure Z is positive
    Z_min = np.min(Z)
    Z = Z - Z_min

    Z_nonzero = Z[Z > 0]
    eps = np.min(Z_nonzero) if Z_nonzero.size > 0 else 1e-1

    Z += eps

    # Create the figure and set up the subplots
    fig = plt.figure(figsize=(12, 6))

    # 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', norm=LogNorm())
    fig.colorbar(surf, ax=ax1)
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.set_title('3D Surface Plot')

    # 2D Heatmap
    ax2 = fig.add_subplot(122)
    heatmap = ax2.pcolormesh(X, Y, Z, cmap='viridis',
                             shading='auto', norm=LogNorm())
    fig.colorbar(heatmap, ax=ax2)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_title('2D Heatmap')
    return fig


def plot_2d_loss_landscape_only(task: Task, plot_resolution: int = 100,
                                stepsize: float = 0.033, seed: int = 0,
                                ax=None, colorbar: bool = True):
    params, static = eqx.partition(task.model, eqx.is_inexact_array)
    samples = grid_sampling(
        params=params, n_points_per_dim=int(1 // stepsize),
        lower_bound=0.0, upper_bound=1.0)

    z = evaluate(params=samples, task=task, key=jr.key(seed))
    x = samples[:, 0]
    y = samples[:, 1]

    import numpy as np
    from scipy.interpolate import griddata

    X, Y = np.meshgrid(np.linspace(min(x), max(x), plot_resolution),
                       np.linspace(min(y), max(y), plot_resolution))

    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Ensure Z is positive
    Z_min = np.min(Z)
    Z = Z - Z_min

    Z_nonzero = Z[Z > 0]
    eps = np.min(Z_nonzero) if Z_nonzero.size > 0 else 1e-1

    Z += eps

    # Create the figure and set up the subplots
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    heatmap = ax.pcolormesh(X, Y, Z, cmap='viridis',
                            shading='auto', norm=LogNorm())
    if colorbar:
        fig.colorbar(heatmap, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def plot_3d_loss_landscape_only(task: Task, plot_resolution: int = 100,
                                stepsize: float = 0.033, seed: int = 0,
                                ax=None):
    params, static = eqx.partition(task.model, eqx.is_inexact_array)
    samples = grid_sampling(
        params=params, n_points_per_dim=int(1 // stepsize),
        lower_bound=0.0, upper_bound=1.0)

    z = evaluate(params=samples, task=task, key=jr.key(seed))
    x = samples[:, 0]
    y = samples[:, 1]

    import numpy as np
    from scipy.interpolate import griddata

    X, Y = np.meshgrid(np.linspace(min(x), max(x), plot_resolution),
                       np.linspace(min(y), max(y), plot_resolution))

    Z = griddata((x, y), z, (X, Y), method='cubic')
    Z_orig = Z.copy()

    # Ensure Z is positive
    Z_min = np.min(Z)
    Z = Z - Z_min

    Z_nonzero = Z[Z > 0]
    eps = np.min(Z_nonzero) if Z_nonzero.size > 0 else 1e-1

    Z += eps

    # If no axis is provided, create one
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        created_fig = True

    # Plot on the provided or created axis
    _ = ax.plot_surface(X, Y, Z, cmap='viridis', norm=LogNorm(),
                        zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if created_fig:
        return fig, ax, Z_orig
    else:
        return ax


def plot_2d_trajectories(raw_results_list: List[xr.DataArray], task: Task,
                         realization: int):
    # Get the default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    X, Y, Z = plot_2d_loss_landscape_only(task=task)

    def get_grid(n):
        return (math.ceil(n / 3), min(n, 3))
    nrows, ncols = get_grid(len(raw_results_list))
    fig, axes = plt.subplots(
        *get_grid(len(raw_results_list)), figsize=(ncols*5, nrows*5))

    if len(raw_results_list) == 1:
        axes = np.array([axes])

    for ax, r_i in zip(axes.flatten(), raw_results_list):
        heatmap = ax.pcolormesh(X, Y, Z, cmap='viridis',
                                shading='auto', norm=LogNorm())
        fig.colorbar(heatmap, ax=ax)
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        switching_iters = np.cumsum(
            [int(round(pop * b * r_i.attrs['n_iterations']))
             for pop, b in zip(r_i.attrs['popsizes'], r_i.attrs['budgets'])])
        for j, (color, current) in enumerate(zip(colors, switching_iters)):
            prev = 0 if j == 0 else switching_iters[j-1]
            current
            x0 = r_i['input'].isel(
                {REALIZATION: realization, INPUT_DIM: 0,
                 BUDGET: range(prev, current)})
            x1 = r_i['input'].isel(
                {REALIZATION: realization, INPUT_DIM: 1,
                 BUDGET: range(prev, current)})
            y = r_i['output'].isel(
                {REALIZATION: realization, OUTPUT_DIM: 0,
                 BUDGET: range(prev, current)})
            ax.scatter(x0.values, x1.values, c=color, s=1, alpha=1.0)
            ax.scatter(x0.values[y.argmin()], x1.values[y.argmin()],
                       c='k', s=10, alpha=1.0, marker='x')

    return fig


def make_trajectory_animation(task, data, ax):
    fig, ax = plot_2d_loss_landscape_only(task=task, ax=ax, colorbar=False)
    scatter = ax.scatter([], [], c='#CC3311', s=20)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,

    def update(frame):
        scatter.set_offsets(data[:frame + 1])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(
        data), init_func=init, blit=True, interval=200)

    return ani


def make_3d_trajectory_animation(task, data, loss):
    fig, ax, Z = plot_3d_loss_landscape_only(task=task)
    ax.view_init(elev=45, azim=-75)  # rotate the view

    # Normalize Z offset
    L = loss.ravel()
    Z_min = np.min(Z)
    L = L - Z_min
    L_nonzero = L[L > 0]
    eps = np.min(L_nonzero) if L_nonzero.size > 0 else 1e-1
    L += eps

    # Combine (x, y, z) points
    data = np.concatenate([data, L[:, None]], axis=1)  # shape (N, 3)
    xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]

    # Create initial empty 3D scatter
    scatter = ax.scatter([], [], [], c='#CC3311', s=20,
                         zorder=10, depthshade=False)

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,

    def update(frame):
        scatter._offsets3d = (
            xs[:frame + 1],
            ys[:frame + 1],
            zs[:frame + 1],
        )

        return scatter,

    ani = animation.FuncAnimation(
        fig, update, frames=len(data),
        init_func=init, blit=False, interval=200
    )

    return ani
