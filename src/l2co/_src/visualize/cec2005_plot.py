"""Visualization utilities for CEC 2005 benchmark results.

Provides functions to plot ERTD profiles and optimizer actions grouped
by CEC 2005 function category, dimensionality, or individual function.
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt

from l2co._src.visualize.plotting import ertd_per_group

CEC2005_CATEGORY_GROUPS = [
    {"fn_name": ["f1", "f2", "f3", "f4", "f5"]},
    {
        "fn_name": [
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
            "f12",
            "f13",
            "f14",
        ]
    },
    {
        "fn_name": [
            "f15",
            "f16",
            "f17",
            "f18",
            "f19",
            "f20",
            "f21",
            "f22",
            "f23",
            "f24",
            "f25",
        ]
    },
]

CEC2005_CATEGORY_NAMES = [
    "unimodal",
    "multimodal_basic",
    "multimodal_composition",
]

CEC2005_ALL_FNS = [
    fn for group in CEC2005_CATEGORY_GROUPS for fn in group["fn_name"]
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
    """Plot ERTD profiles grouped by CEC2005 function category.

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
    groups = list(
        zip(
            CEC2005_CATEGORY_GROUPS,
            CEC2005_CATEGORY_NAMES,
            strict=True,
        )
    )
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
        Dimensionalities to plot. Defaults to ``[10, 30, 50]``.
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
    dims = dims if dims is not None else [10, 30, 50]
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
    """Plot ERTD profiles for individual CEC2005 functions.

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
        CEC2005 functions to plot. Defaults to all 25 CEC2005 functions.
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
    fns = fns if fns is not None else CEC2005_ALL_FNS
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
