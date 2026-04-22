"""Visualization utilities for BBOB benchmark results.

Provides functions to plot ERTD profiles and optimizer actions grouped
by BBOB function category, dimensionality, or individual function.
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt

from l2co._src.visualize.plotting import ertd_per_group

BBOB_CATEGORY_GROUPS = [
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

BBOB_CATEGORY_NAMES = [
    "separable",
    "low_conditioning",
    "high_conditioning",
    "multimodal_adequate",
    "multimodal_weak",
]

BBOB_ALL_FNS = [
    fn for group in BBOB_CATEGORY_GROUPS for fn in group["fn_name"]
]


def ertd_per_category(
    d,
    plot_actions: str | None = None,
    optimizers=None,
    line: float | None = None,
    save_dir: Path | None = None,
    sharey: bool = False,
    sharex: bool = False,
    **savefig_kwargs,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot ERTD profiles grouped by BBOB function category.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing optimization results.
    plot_actions : str or None, optional
        If not None, also plot actions for the specified schedule.
    optimizers : optional
        Optimizers to pass to ``plot_update_step``.
    line : float or None, optional
        X-coordinate for a vertical reference line.
    save_dir : Path or None, optional
        Directory to save plots to.
    sharey : bool, optional
        Strip y-axis labels from all but the first column.
        Default is ``False``.
    sharex : bool, optional
        Strip x-axis labels from all but the bottom row.
        Default is ``False``.
    **savefig_kwargs
        Forwarded to :func:`~bessaplots.savefig` (or
        :class:`~bessaplots.FigureGrid`).
    """
    groups = list(zip(BBOB_CATEGORY_GROUPS, BBOB_CATEGORY_NAMES, strict=True))
    return ertd_per_group(
        d,
        groups,
        plot_actions=plot_actions,
        optimizers=optimizers,
        line=line,
        save_dir=save_dir,
        sharey=sharey,
        sharex=sharex,
        **savefig_kwargs,
    )


def ertd_per_dimension(
    d,
    line=None,
    plot_actions: str | None = None,
    optimizers=None,
    dims: Iterable[int] | None = None,
    save_dir: Path | None = None,
    sharey: bool = False,
    sharex: bool = False,
    **savefig_kwargs,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot ERTD profiles grouped by problem dimensionality.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing optimization results.
    line : float or None, optional
        X-coordinate for a vertical reference line.
    plot_actions : str or None, optional
        If not None, also plot actions for the specified schedule.
    optimizers : optional
        Optimizers to pass to ``plot_update_step``.
    dims : Iterable[int] or None, optional
        Dimensionalities to plot. Defaults to
        ``[2, 3, 5, 10, 20, 40]``.
    save_dir : Path or None, optional
        Directory to save plots to.
    sharey : bool, optional
        Strip y-axis labels from all but the first column.
        Default is ``False``.
    sharex : bool, optional
        Strip x-axis labels from all but the bottom row.
        Default is ``False``.
    **savefig_kwargs
        Forwarded to :func:`~bessaplots.savefig` (or
        :class:`~bessaplots.FigureGrid`).
    """
    dims = dims if dims is not None else [2, 3, 5, 10, 20, 40]
    groups = [({"dimensionality": [dim]}, f"{dim}d") for dim in dims]
    return ertd_per_group(
        d,
        groups,
        plot_actions=plot_actions,
        optimizers=optimizers,
        line=line,
        save_dir=save_dir,
        sharey=sharey,
        sharex=sharex,
        **savefig_kwargs,
    )


def ertd_per_function(
    d,
    line=None,
    plot_actions: str | None = None,
    optimizers=None,
    fns: Iterable[str] | None = None,
    save_dir: Path | None = None,
    sharey: bool = False,
    sharex: bool = False,
    **savefig_kwargs,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot ERTD profiles for individual BBOB functions.

    Parameters
    ----------
    d : DataLoader
        DataLoader containing optimization results.
    line : float or None, optional
        X-coordinate for a vertical reference line.
    plot_actions : str or None, optional
        If not None, also plot actions for the specified schedule.
    optimizers : optional
        Optimizers to pass to ``plot_update_step``.
    fns : Iterable[str] or None, optional
        BBOB functions to plot. Defaults to all 24 BBOB functions.
    save_dir : Path or None, optional
        Directory to save plots to.
    sharey : bool, optional
        Strip y-axis labels from all but the first column.
        Default is ``False``.
    sharex : bool, optional
        Strip x-axis labels from all but the bottom row.
        Default is ``False``.
    **savefig_kwargs
        Forwarded to :func:`~bessaplots.savefig` (or
        :class:`~bessaplots.FigureGrid`).
    """
    fns = fns if fns is not None else BBOB_ALL_FNS
    groups = [({"fn_name": [fn]}, fn) for fn in fns]
    return ertd_per_group(
        d,
        groups,
        plot_actions=plot_actions,
        optimizers=optimizers,
        line=line,
        save_dir=save_dir,
        sharey=sharey,
        sharex=sharex,
        **savefig_kwargs,
    )
