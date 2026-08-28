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

#: Marker rotation for the second and later variants of one optimizer
#: (see :func:`get_schedule_styles`). Ordered for visual distinctness at
#: figure scale, and deliberately separate from :data:`marker`, whose
#: order pins the per-optimizer assignments baked into
#: :data:`BASE_OPT_STYLES` and must not be reshuffled.
variant_markers = ["o", "s", "^", "d", "v", "X", "*", "."]


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
        # The headroom-balanced quartet (l2co_experiments
        # optimizers=headroom4) plus the uniform-random rl2co
        # schedule. On an ERTD panel these are five solid,
        # marker-free 1.0pt curves sharing 3.417in with black rl2co,
        # black dotted vbs and grey dotted randomsearch, so hue is
        # the only channel separating them and they have to be solved
        # for jointly rather than picked one at a time.
        #
        # Chosen by searching an OKLCH grid (L 0.44-0.72, in-gamut
        # chroma) filtered to >=3:1 contrast on white, OKLab dE >=10
        # from randomsearch's #888888 and >=45 from black, for the
        # 5-subset maximising the *worst* pair under normal,
        # protanopic and deuteranopic vision (Machado 2009 at
        # severity 1.0), subject to tritan dE >=8 and >=50 deg hue
        # spacing. Worst pair is dE 14.3. The palette this replaced
        # sat at 7.7, below the dE 8 floor for thin categorical
        # lines: lbfgs/rprop collapsed to 7.7 under deuteranopia and
        # lbfgs/crfmnes to 7.6 under tritanopia.
        #
        # Cool for the gradient-based half, warm for the
        # derivative-free half, magenta for the schedule that is
        # neither. Worst-pair dE does not depend on which optimizer
        # takes which colour, so the assignment is free and may as
        # well carry the family split.
        #
        # Two constraints worth not rediscovering. A true green
        # cannot join this set -- every green/amber pairing tested
        # scored dE 2.5-7.9, because red-green dichromats resolve a
        # single chromatic axis and green and amber sit on the same
        # side of it; mr15ga's olive is that green pushed far enough
        # down in lightness to survive. And holding lbfgs at the old
        # #0077BB caps the whole set at dE 5.6, since that blue
        # crowds whatever indigo the remaining slots need, which is
        # why the blue moved rather than staying pinned.
        #
        # rprop, crfmnes and mr15ga had no entry here before: they
        # were taking positional colours from `colors` through
        # itertools.cycle over optimizers.keys(), which is how they
        # became #AA4499, #117733 and #332288. Pinning them also
        # stops registry order from repainting them. Their markers
        # are the ones that cycle handed out, so nothing but colour
        # changes.
        "lbfgs": OptStyle(color="#0AA0D2", marker="^"),
        "rprop": OptStyle(color="#5E56DE", marker="d"),
        "crfmnes": OptStyle(color="#CC7200", marker="*"),
        "mr15ga": OptStyle(color="#4C5A04", marker="X"),
        "sepcmaes": OptStyle(color="#EE6677", marker="x"),
        "stage0": OptStyle(color="#BBBBBB", marker=None, linestyle="--"),
        "l2co": OptStyle(color="#000000", marker=None, linestyle="-"),
        # Black, not #FF0000. crfmnes's #CC7200 is dE 0.3 from pure
        # red under deuteranopia, so the trained policy and one of
        # its own portfolio members would be the same line; the
        # legend strip (media/ertd/legend.pgf in RL2CO_paper) already
        # drew rl2co black, and the manuscript describes it as black.
        "rl2co": OptStyle(color="#000000", marker="o", linestyle="-"),
        "randomsearch": OptStyle(color="#888888", marker=None, linestyle=":"),
        # The two oracle references, dashed and dotted so neither reads as
        # one of the portfolio's own solid optimizer curves.
        #
        # "sbs" -- the single best solver, one optimizer for the whole task
        # set (`calculate_sbs`).
        "sbs": OptStyle(color="#000000", marker=None, linestyle="--"),
        # "vbs" -- the virtual best solver, the per-task oracle selection
        # `ertd_profile(include_best=True)` adds. Dotted like randomsearch,
        # the other line that bounds rather than competes, and black to
        # randomsearch's grey.
        "vbs": OptStyle(color="#000000", marker=None, linestyle=":"),
        # Variant pin (a key with an underscore suffix, resolved ahead of
        # the base optimizer -- see get_schedule_styles). Every rl2co
        # schedule otherwise collapses onto base "rl2co" and so onto one
        # black solid line, which puts the uniform-random baseline in
        # the trained policy's colour. Giving it its own colour and marker
        # separates the two on colour, dash pattern, and shape at once.
        "rl2co_random": OptStyle(color="#B12083", marker="s", linestyle="--"),
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
    "sbs": "SBS",
    "vbs": "VBS",
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
    (e.g., 'adam_lr=0.01', 'adam_lr=0.3') step through linestyles
    *and* markers to maintain visual distinction: the first variant
    keeps the base marker, later ones take distinct shapes from
    :data:`variant_markers`. A base style with no marker stays
    unmarked, so the deliberately plain baselines (``randomsearch``,
    ``l2co``, ``stage0``) do not sprout shapes.

    A full schedule name present in ``BASE_OPT_STYLES`` (a *variant
    pin*, e.g. ``rl2co_random``) is honoured verbatim and takes
    precedence over its base optimizer's style. Its linestyle and
    marker are then withheld from the cycle, so a pinned variant never
    collides with a cycled sibling regardless of portfolio order.
    """
    names = list(schedule_names)

    def _is_pin(full_name: str) -> bool:
        """Whether this name has its own entry, not just its base's."""
        return (
            full_name != full_name.split("_")[0]
            and full_name in BASE_OPT_STYLES
        )

    schedule_styles = {}
    # Per base name, the linestyles / markers already handed out. Every
    # pin in the selection is registered up front, so the cycle skips
    # them no matter where they sit in the portfolio order.
    used_linestyles: dict[str, set[str]] = {}
    used_markers: dict[str, set[str]] = {}
    for full_name in names:
        if not _is_pin(full_name):
            continue
        base_name = full_name.split("_")[0]
        pinned = BASE_OPT_STYLES[full_name]
        used_linestyles.setdefault(base_name, set()).add(pinned.linestyle)
        if pinned.marker is not None:
            used_markers.setdefault(base_name, set()).add(pinned.marker)

    for full_name in names:
        # 1. Identify the base name (split at the first underscore)
        base_name = full_name.split("_")[0]
        taken_ls = used_linestyles.setdefault(base_name, set())
        taken_mk = used_markers.setdefault(base_name, set())

        # 2. A variant pin wins outright; it was registered above.
        if _is_pin(full_name):
            schedule_styles[full_name] = BASE_OPT_STYLES[full_name]
            continue

        # 3. Retrieve base color/marker from the global registry
        # Fallback to a default if the optimizer is not in the imported list
        base_style = BASE_OPT_STYLES.get(base_name, OptStyle())

        # if linestyle is already defined in base_style, use it directly
        if base_style.linestyle != "-":
            schedule_styles[full_name] = base_style
            continue

        # 4. Take the first linestyle this base has not used yet, falling
        # back to reuse once all four are spoken for.
        chosen_linestyle = next(
            (ls for ls in line_styles if ls not in taken_ls),
            line_styles[len(taken_ls) % len(line_styles)],
        )
        taken_ls.add(chosen_linestyle)

        # 5. Same for the marker, preferring the base optimizer's own
        # shape so a single-variant portfolio looks exactly as before.
        chosen_marker = base_style.marker
        if chosen_marker is not None and chosen_marker in taken_mk:
            chosen_marker = next(
                (m for m in variant_markers if m not in taken_mk),
                chosen_marker,
            )
        if chosen_marker is not None:
            taken_mk.add(chosen_marker)

        # 6. Create the final OptStyle for this specific schedule
        schedule_styles[full_name] = OptStyle(
            color=base_style.color,
            marker=chosen_marker,
            linestyle=chosen_linestyle,
        )

    return schedule_styles
