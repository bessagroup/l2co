from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from bessaplots import savefig

import l2co.plotting
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


def actions_per_category(
    d,
    strategy: str | None = None,
    optimizers=None,
):
    for group, name in zip(
        CEC2005_CATEGORY_GROUPS,
        CEC2005_CATEGORY_NAMES,
        strict=True,
    ):
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
            path=f"actions_{strategy}_{name}",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )
        plt.close(fig_print)


def ertd_per_category(
    d,
    plot_actions: str | None = None,
    optimizers=None,
    line: float | None = None,
    save_dir: Path | None = None,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot ERTD profiles grouped by CEC2005 function category."""
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
    )


def ertd_per_dimension(
    d,
    line=None,
    plot_actions: str | None = None,
    optimizers=None,
    dims: Iterable[int] | None = None,
    save_dir: Path | None = None,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot ERTD profiles grouped by problem dimensionality."""
    dims = dims if dims is not None else [10, 30, 50]
    groups = [({"dimensionality": [dim]}, f"{dim}d") for dim in dims]
    return ertd_per_group(
        d,
        groups,
        plot_actions=plot_actions,
        optimizers=optimizers,
        line=line,
        save_dir=save_dir,
    )


def actions_per_dimension(
    d,
    strategy: str | None = None,
    optimizers=None,
    dims: Iterable[int] | None = None,
):
    dims = dims if dims is not None else [10, 30, 50]

    for dim in dims:
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
            path=f"actions_{strategy}_{dim}d",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )
        plt.close(fig_print)


def ertd_per_function_raster(
    d,
    plot_actions: str | None = None,
    optimizers=None,
    line=None,
    save_dir: Path | None = None,
) -> tuple[list[plt.Figure], list[Path], list[str]]:
    """Plot ERTD profiles for all individual CEC2005 functions."""
    groups = [({"fn_name": [fn]}, fn) for fn in CEC2005_ALL_FNS]
    return ertd_per_group(
        d,
        groups,
        plot_actions=plot_actions,
        optimizers=optimizers,
        line=line,
        save_dir=save_dir,
    )


def actions_per_function_raster(
    d,
    strategy: str | None = None,
    optimizers=None,
):
    for fn_name in CEC2005_ALL_FNS:
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
            path=f"actions_{strategy}_{fn_name}",
            n_side_by_side=4,
            span_columns=True,
            height=1.0,
        )
        plt.close(fig_print)
