# Standard
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

# Local
from l2co.optimizers import optimizers

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

colors = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
]
marker = [".", "v", "s", "*", "d", "x", "X"]
line_styles = ["-", "--", "-.", ":"]


@dataclass
class OptStyle:
    color: str = "#000000"
    marker: str | None = None
    linestyle: str = "-"


BASE_OPT_STYLES = {
    optimizer: OptStyle(color=c, marker=m)
    for optimizer, c, m in zip(
        optimizers.keys(), itertools.cycle(colors), itertools.cycle(marker)
    )
}

# Overwrite specific colors/markers if desired
BASE_OPT_STYLES.update(
    {
        "adam": OptStyle(color="#4477AA", marker="o"),
        "pso": OptStyle(color="#CCBB44", marker="s"),
        "lbfgs": OptStyle(color="#228833", marker="^"),
        "sepcmaes": OptStyle(color="#EE6677", marker="x"),
        "stage0": OptStyle(color="#BBBBBB", marker=None, linestyle="--"),
        "l2co": OptStyle(color="#000000", marker=None, linestyle="-"),
        "rl2co": OptStyle(color="#FF0000", marker=None, linestyle="-"),
    }
)


LABEL_MAP = {
    "sepcmaes": "Sep-CMA-ES",
    # "l2co": r"\textbf{L2CO (Ours)}",  # You can even use LaTeX bold here
    "lbfgs": "L-BFGS",
    "pso": "PSO",
    "adam_learning_rate=0.01": "Adam",
    "adam_learning_rate=0.3": "Adam",
    "stage0": r"L2CO ($g_\text{first}$)",
    "l2co": r"L2CO ($g_\text{first} \rightarrow g_\text{second}$)",
    "rl2co": "RL2CO",
}


def get_schedule_styles(schedule_names: Iterable[str]) -> dict[str, OptStyle]:
    """
    Map schedule names to OptStyle objects with appropriate styling.

    Parameters
    ----------
    schedule_names : Iterable[str]
        Collection of schedule/optimizer names to style.

    Returns
    -------
    dict[str, OptStyle]
        Mapping from schedule names to OptStyle objects with assigned
        colors, markers, and linestyles.

    Notes
    -----
    Base optimizer names (e.g., 'adam') determine color and marker
    from BASE_OPT_STYLES. Variants of the same optimizer
    (e.g., 'adam_lr=0.01', 'adam_lr=0.3') cycle through linestyles
    to maintain visual distinction.
    """
    schedule_styles = {}
    # Track how many variants of each base name we have encountered
    variant_tracker = {}

    for full_name in schedule_names:
        # 1. Identify the base name (split at the first underscore)
        base_name = full_name.split("_")[0]

        # 2. Retrieve base color/marker from the global registry
        # Fallback to a default if the optimizer is not in the imported list
        base_style = BASE_OPT_STYLES.get(base_name, OptStyle())

        # if linestyle is already defined in base_style, use it directly
        if base_style.linestyle != "-":
            schedule_styles[full_name] = base_style
            continue

        # 3. Handle Linestyle cycling for variants
        idx = variant_tracker.get(base_name, 0)
        chosen_linestyle = line_styles[idx % len(line_styles)]
        variant_tracker[base_name] = idx + 1

        # 4. Create the final OptStyle for this specific schedule
        schedule_styles[full_name] = OptStyle(
            color=base_style.color,
            marker=base_style.marker,
            linestyle=chosen_linestyle,
        )

    return schedule_styles
