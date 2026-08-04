"""Optimizer color, marker, and linestyle definitions for plots."""

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
    """Visual style for a single optimizer in plots.

    Attributes
    ----------
    color : str
        Hex colour code, by default ``"#000000"``.
    marker : str or None
        Matplotlib marker symbol, by default ``None``.
    linestyle : str
        Matplotlib linestyle string, by default ``"-"``.
    """

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
        "randomsearch": OptStyle(color="#888888", marker=None, linestyle=":"),
        # Best-solver envelope (added by ertd_profile(include_best=True)):
        # a dashed line so it reads as an oracle reference, not one of the
        # portfolio's own solid optimizer curves.
        "sbs": OptStyle(color="#000000", marker=None, linestyle="--"),
    }
)


#: Display labels for schedules, resolved by :func:`display_name`.
#:
#: Prefer a **bare optimizer name** (``"adam"``): ``display_name`` falls
#: back to it for every hyperparameter variant, so one entry covers
#: ``adam_learning_rate=0.01`` and any future learning rate. Add a
#: suffixed key only to give one specific variant its own label.
LABEL_MAP = {
    "sepcmaes": "Sep-CMA-ES",
    "randomsearch": "Random search",
    # "l2co": r"\textbf{L2CO (Ours)}",  # You can even use LaTeX bold here
    "lbfgs": "L-BFGS",
    "pso": "PSO",
    "adam": "Adam",
    # The headroom-balanced quartet (l2co_experiments optimizers=headroom4);
    # evosax spells the last two CR_FM_NES and MR15_GA.
    "rprop": "Rprop",
    "crfmnes": "CR-FM-NES",
    "mr15ga": "MR15-GA",
    "stage0": r"L2CO ($g_\text{first}$)",
    "l2co": r"L2CO ($g_\text{first} \rightarrow g_\text{second}$)",
    "rl2co": "RL2CO",
}


def display_name(schedule_name: str) -> str:
    """Human-readable label for a schedule, falling back when unlisted.

    ``LABEL_MAP`` is hand-maintained and only covers the optimizers that
    have needed a prettier label so far, hyperparameter suffix included
    (``adam_learning_rate=0.01``). Treating a miss as an error would make
    that table a hard dependency of every optimizer portfolio -- swapping
    in an optimizer nobody has labelled yet would crash the figure rather
    than draw it with a plain name. So resolution degrades instead:

    1. the full schedule name, suffix and all;
    2. the base optimizer name -- the same ``split("_")[0]`` rule
       :func:`get_schedule_styles` uses to find a colour, so a labelled
       optimizer keeps its label across hyperparameter variants;
    3. the raw name, which is always readable if not pretty.

    Parameters
    ----------
    schedule_name : str
        Schedule/optimizer name, e.g. ``"lbfgs"`` or
        ``"rprop_learning_rate=0.001"``.

    Returns
    -------
    str
        Label to show in a legend or axis.
    """
    if schedule_name in LABEL_MAP:
        return LABEL_MAP[schedule_name]
    return LABEL_MAP.get(schedule_name.split("_")[0], schedule_name)


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
