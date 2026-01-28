import math

import matplotlib.pyplot as plt

import l2co.plotting
from l2co._src.visualize.plotting import savefig


def _add_eval_line(ax, line_val):
    if line_val is not None:
        ax.axvline(
            x=line_val, color="black", linestyle="--", linewidth=2, zorder=5
        )


def ertd_per_category(d, line=None):
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
                "sum_of_different_power",
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
            d_group, show_legend=True
        )

        savefig(
            fig=fig_print,
            name=f"rtd_{names[i]}",
            n_side_by_side=3,
            span_columns=True,
        )
    return fig, axes


def ertd_per_dimension(d, line=None):
    dims = [2, 3, 5, 10, 20, 40]
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
            d_group, show_legend=True
        )

        savefig(
            fig=fig_print,
            name=f"rtd_{dim}d",
            n_side_by_side=3,
            span_columns=True,
        )
    return fig, axes


def ertd_per_function_raster(d, cols=6, line=None):
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
            "sum_of_different_power",
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
    axes_flat = axes.flatten()

    for i, fn_name in enumerate(all_fns):
        ax = axes_flat[i]
        d_fn = d.select_feature({"fn_name": [fn_name]})

        if d_fn.n_realization == 0:
            ax.axis("off")
            continue

        # We handle legend visibility manually or let the library decide
        l2co.plotting.plot_ertd_profile(
            d_fn, ax=ax, show_legend=False, title_text=fn_name
        )

        _add_eval_line(ax, line)

    # Hide unused trailing axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    return fig, axes
