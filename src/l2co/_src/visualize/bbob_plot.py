import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
from bessaplots import savefig

import l2co.plotting


def _add_eval_line(ax, line_val):
    """
    Add a vertical evaluation line to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to add the line to.
    line_val : float or None
        X-coordinate for the vertical line. If None, no line is added.

    Returns
    -------
    None
    """
    if line_val is not None:
        ax.axvline(
            x=line_val, color="black", linestyle="--", linewidth=2, zorder=5
        )


def actions_per_category(
    d,
    strategy: str | None = None,
    optimizers=None,
):
    groups = [
        {
            "fn_name": [
                "sphere",
                "ellipsoid_seperable",
                "rastrigin_seperable",
                "skew_rastrigin_bueche",
                "linear_slope",
            ]
        },
        {
            "fn_name": [
                "attractive_sector",
                "step_ellipsoid",
                "rosenbrock",
                "rosenbrock_rotated",
            ]
        },
        {
            "fn_name": [
                "ellipsoid",
                "discuss",
                "bent_cigar",
                "sharp_ridge",
                "sum_of_different_powers",
            ]
        },
        {
            "fn_name": [
                "rastrigin",
                "weierstrass",
                "schaffer_f7_condition_10",
                "schaffer_f7_condition_1000",
                "griewank_rosenbrock_f8f2",
            ]
        },
        {
            "fn_name": [
                "schwefel_xsinx",
                "gallagher_101_peaks",
                "gallagher_21_peaks",
                "katsuura",
                "lunacek_bi_rastrigin",
            ]
        },
    ]
    names = [
        "separable",
        "low_conditioning",
        "high_conditioning",
        "multimodal_adequate",
        "multimodal_weak",
    ]

    # Hardcoded figure size

    for i, group in enumerate(groups):
        d_group = d.select_feature(group)

        fig_print, ax_print = l2co.plotting.plot_update_step(
            d_group,
            schedule=strategy,
            optimizers=optimizers,
            cost_metric="iterations",
            show_legend=False,
        )

        savefig(
            fig=fig_print,
            name=f"actions_{strategy}_{names[i]}",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )


def ertd_per_category(
    d,
    plot_actions: str | None = None,
    optimizers=None,
    line: float | None = None,
):
    """
    Plot ERTD profiles grouped by BBOB function category.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing BBOB benchmark function results.
    plot_actions : str or None, optional
        If not None, make a plot of the actions of the specified optimizer.
    line : float or None, optional
        X-coordinate for a vertical reference line, by default None.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and array of axes for the category plots.

    Notes
    -----
    Creates 5 subplots for BBOB categories: separable, low conditioning,
    high conditioning, multimodal adequate, and multimodal weak.
    Also saves individual plots for each category.
    """
    groups = [
        {
            "fn_name": [
                "sphere",
                "ellipsoid_seperable",
                "rastrigin_seperable",
                "skew_rastrigin_bueche",
                "linear_slope",
            ]
        },
        {
            "fn_name": [
                "attractive_sector",
                "step_ellipsoid",
                "rosenbrock",
                "rosenbrock_rotated",
            ]
        },
        {
            "fn_name": [
                "ellipsoid",
                "discuss",
                "bent_cigar",
                "sharp_ridge",
                "sum_of_different_powers",
            ]
        },
        {
            "fn_name": [
                "rastrigin",
                "weierstrass",
                "schaffer_f7_condition_10",
                "schaffer_f7_condition_1000",
                "griewank_rosenbrock_f8f2",
            ]
        },
        {
            "fn_name": [
                "schwefel_xsinx",
                "gallagher_101_peaks",
                "gallagher_21_peaks",
                "katsuura",
                "lunacek_bi_rastrigin",
            ]
        },
    ]
    names = [
        "separable",
        "low_conditioning",
        "high_conditioning",
        "multimodal_adequate",
        "multimodal_weak",
    ]

    # Hardcoded figure size
    fig, axes = plt.subplots(1, len(groups), figsize=(20, 4), sharey=True)

    for i, (ax, group) in enumerate(zip(axes, groups, strict=True)):
        d_group = d.select_feature(group)
        l2co.plotting.plot_ertd_profile(
            d_group,
            show_legend=(i == len(groups) - 1),
            title_text=names[i],
        )
        _add_eval_line(ax, line)

        fig_print, ax_print = l2co.plotting.plot_ertd_profile(
            d_group, show_legend=False
        )

        savefig(
            fig=fig_print,
            name=f"rtd_{names[i]}",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )

        if plot_actions is not None:
            fig_print, ax_print = l2co.plotting.plot_update_step(
                d_group,
                schedule=plot_actions,
                optimizers=optimizers,
                cost_metric="iterations",
                show_legend=False,
            )

            savefig(
                fig=fig_print,
                name=f"actions_{plot_actions}_{names[i]}",
                n_side_by_side=4,
                span_columns=True,
                height=1.0,
            )
    return fig, axes


def ertd_per_dimension(
    d,
    line=None,
    plot_actions: str | None = None,
    optimizers=None,
    dims: Iterable[int] | None = None,
):
    """
    Plot ERTD profiles grouped by problem dimensionality.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing optimization results.
    line : float or None, optional
        X-coordinate for a vertical reference line, by default None.
    dims : Iterable[int] or None, optional
        List of dimensions to plot. If None, uses [2, 3, 5, 10, 20, 40].

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and array of axes for the dimension plots.

    Notes
    -----
    Creates subplots for each specified dimension, showing ERTD
    profiles for problems of that dimensionality. Also saves
    individual plots for each dimension.
    """
    dims = dims if dims is not None else [2, 3, 5, 10, 20, 40]
    # Hardcoded figure size
    fig, axes = plt.subplots(1, len(dims), figsize=(25, 4), sharey=True)

    for i, (ax, dim) in enumerate(zip(axes, dims, strict=True)):
        d_group = d.select_feature({"dimensionality": [dim]})
        l2co.plotting.plot_ertd_profile(
            d_group,
            ax=ax,
            show_legend=(i == len(dims) - 1),
            title_text=f"{dim}D",
        )
        _add_eval_line(ax, line)

        fig_print, ax_print = l2co.plotting.plot_ertd_profile(
            d_group, show_legend=False
        )

        savefig(
            fig=fig_print,
            name=f"rtd_{dim}d",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )

        if plot_actions is not None:
            fig_print, ax_print = l2co.plotting.plot_update_step(
                d_group,
                schedule=plot_actions,
                optimizers=optimizers,
                cost_metric="iterations",
                show_legend=False,
            )

            savefig(
                fig=fig_print,
                name=f"actions_{plot_actions}_{dim}d",
                n_side_by_side=4,
                span_columns=True,
                height=1.0,
            )
    return fig, axes


def actions_per_dimension(
    d,
    strategy: str | None = None,
    optimizers=None,
    dims: Iterable[int] | None = None,
):
    dims = dims if dims is not None else [2, 3, 5, 10, 20, 40]
    # Hardcoded figure size

    for _i, dim in enumerate(dims):
        d_group = d.select_feature({"dimensionality": [dim]})
        fig_print, ax_print = l2co.plotting.plot_update_step(
            d_group,
            schedule=strategy,
            optimizers=optimizers,
            cost_metric="iterations",
            show_legend=False,
        )

        savefig(
            fig=fig_print,
            name=f"actions_{strategy}_{dim}d",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )


def ertd_per_function_raster(
    d, plot_actions: str | None = None, optimizers=None, cols=6, line=None
):
    """
    Create raster plot of ERTD profiles for all BBOB functions.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing BBOB benchmark function results.
    cols : int, optional
        Number of columns in the raster grid, by default 6.
    line : float or None, optional
        X-coordinate for a vertical reference line, by default None.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and array of axes for the function plots.

    Notes
    -----
    Creates a grid layout with one subplot per BBOB function,
    displaying ERTD profiles. Rows are calculated based on the
    number of functions and specified columns. Also saves
    individual plots for each function.
    """
    all_fns = (
        [
            "sphere",
            "ellipsoid_seperable",
            "rastrigin_seperable",
            "skew_rastrigin_bueche",
            "linear_slope",
        ]
        + [
            "attractive_sector",
            "step_ellipsoid",
            "rosenbrock",
            "rosenbrock_rotated",
        ]
        + [
            "ellipsoid",
            "discuss",
            "bent_cigar",
            "sharp_ridge",
            "sum_of_different_powers",
        ]
        + [
            "rastrigin",
            "weierstrass",
            "schaffer_f7_condition_10",
            "schaffer_f7_condition_1000",
            "griewank_rosenbrock_f8f2",
        ]
        + [
            "schwefel_xsinx",
            "gallagher_101_peaks",
            "gallagher_21_peaks",
            "katsuura",
            "lunacek_bi_rastrigin",
        ]
    )

    num_fns = len(all_fns)
    rows = math.ceil(num_fns / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3.5), sharey=True, sharex=True
    )

    for _, fn_name in enumerate(all_fns):
        d_fn = d.select_feature({"fn_name": [fn_name]})
        fig_print, ax_print = l2co.plotting.plot_ertd_profile(
            d_fn, show_legend=False
        )

        savefig(
            fig=fig_print,
            name=f"rtd_{fn_name}",
            n_side_by_side=4,
            span_columns=True,
        )
        if plot_actions is not None:
            fig_print, ax_print = l2co.plotting.plot_update_step(
                d_fn,
                schedule=plot_actions,
                optimizers=optimizers,
                cost_metric="iterations",
                show_legend=False,
            )

            savefig(
                fig=fig_print,
                name=f"actions_{plot_actions}_{fn_name}",
                n_side_by_side=4,
                span_columns=True,
                height=1.0,
            )

    return fig, axes


def actions_per_function_raster(
    d,
    strategy: str | None = None,
    optimizers=None,
    cols=6,
):
    all_fns = (
        [
            "sphere",
            "ellipsoid_seperable",
            "rastrigin_seperable",
            "skew_rastrigin_bueche",
            "linear_slope",
        ]
        + [
            "attractive_sector",
            "step_ellipsoid",
            "rosenbrock",
            "rosenbrock_rotated",
        ]
        + [
            "ellipsoid",
            "discuss",
            "bent_cigar",
            "sharp_ridge",
            "sum_of_different_powers",
        ]
        + [
            "rastrigin",
            "weierstrass",
            "schaffer_f7_condition_10",
            "schaffer_f7_condition_1000",
            "griewank_rosenbrock_f8f2",
        ]
        + [
            "schwefel_xsinx",
            "gallagher_101_peaks",
            "gallagher_21_peaks",
            "katsuura",
            "lunacek_bi_rastrigin",
        ]
    )

    for _i, fn_name in enumerate(all_fns):
        d_fn = d.select_feature({"fn_name": [fn_name]})
        fig_print, ax_print = l2co.plotting.plot_update_step(
            d_fn,
            schedule=strategy,
            optimizers=optimizers,
            cost_metric="iterations",
            show_legend=False,
        )

        savefig(
            fig=fig_print,
            name=f"actions_{strategy}_{fn_name}",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )
