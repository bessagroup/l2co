"""Optimizer color, marker, and linestyle definitions for plots."""

# Standard
import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

# Third-party
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.population_based import population_based_algorithms

# Local
from l2co_optimizers import (
    lbfgs_mapping,
    normalized_evosax,
    optax_mapping,
    optimizers,
    rbf_trust_region_mapping,
    shade_mapping,
    turbo_mapping,
)

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
        # The three oracle references: black, and each in its own broken
        # line so none reads as one of the portfolio's solid curves.
        #
        # "sbs" -- the single best solver, one optimizer for the whole task
        # set (`DataLoader.single_best_schedule`).
        "sbs": OptStyle(color="#000000", marker=None, linestyle="--"),
        # "vbs" -- the per-run virtual best solver, the oracle selection
        # `ertd_profile(include_best=True)` adds. Dotted like randomsearch,
        # the other line that bounds rather than competes, and black to
        # randomsearch's grey.
        "vbs": OptStyle(color="#000000", marker=None, linestyle=":"),
        # "vbs_task" -- the per-task virtual best solver,
        # `ertd_profile(include_task_best=True)`. Dash-dot: between the
        # dashed sbs and the dotted vbs, as it is between them in what it
        # may know. An exact key, so it pins ahead of base "vbs".
        "vbs_task": OptStyle(color="#000000", marker=None, linestyle="-."),
        # Variant pin (a key with an underscore suffix, resolved ahead of
        # the base optimizer -- see get_schedule_styles). Every rl2co
        # schedule otherwise collapses onto base "rl2co" and so onto one
        # black solid line, which puts the uniform-random baseline in
        # the trained policy's colour. Giving it its own colour and marker
        # separates the two on colour, dash pattern, and shape at once.
        "rl2co_random": OptStyle(color="#B12083", marker="s", linestyle="--"),
    }
)


#                                                        Optimizer categories
# =============================================================================
# The alternative to one hand-tuned colour per optimizer: one hue per
# *kind* of optimizer, so a panel of sixty curves reads as five groups
# (see get_schedule_styles(by="category")).

CATEGORY_GRADIENT = "Gradient-based"
CATEGORY_QUASI_NEWTON = "Quasi-Newton"
CATEGORY_DISTRIBUTION = "Distribution-based"
CATEGORY_POPULATION = "Population-based"
CATEGORY_MODEL_BASED = "Model-based"

#: Every categorised optimizer, per category, in the fixed order its
#: members take their styles in. Read off the registries rather than
#: listed by hand, so an optimizer added to one lands in its category;
#: sorted, so a member's style depends only on the category's
#: membership and never on which schedules a figure happens to plot.
#: Random search is left out: it is a reference line, not a competitor,
#: and keeps its grey dotted style.
CATEGORY_MEMBERS: dict[str, tuple[str, ...]] = {
    CATEGORY_GRADIENT: tuple(sorted(optax_mapping)),
    CATEGORY_QUASI_NEWTON: tuple(sorted(lbfgs_mapping)),
    CATEGORY_DISTRIBUTION: tuple(
        sorted(
            name
            for name, cls in normalized_evosax.items()
            if cls in distribution_based_algorithms.values()
            and name != "randomsearch"
        )
    ),
    CATEGORY_POPULATION: tuple(
        sorted(
            [
                name
                for name, cls in normalized_evosax.items()
                if cls in population_based_algorithms.values()
            ]
            + list(shade_mapping)
        )
    ),
    CATEGORY_MODEL_BASED: tuple(
        sorted([*rbf_trust_region_mapping, *turbo_mapping])
    ),
}

#: Optimizer registry key to its category.
OPTIMIZER_CATEGORIES: dict[str, str] = {
    member: category
    for category, members in CATEGORY_MEMBERS.items()
    for member in members
}

#: One hue per category, stepped in lightness (dark to light) so members
#: of a large category can differ in shade as well as dash pattern.
#:
#: Found by searching OKLCH lightness for fixed hue families -- blue,
#: purple, orange, green and gold, at least 40 degrees of hue apart so
#: each reads as its own family rather than a lighter or darker member
#: of another -- for the set maximising the worst cross-category pair,
#: black and randomsearch's grey included, under normal, protanopic and
#: deuteranopic vision (Machado 2009 at severity 1.0). Shades within a
#: category step by OKLCH L 0.06. Worst cross-category pair under normal
#: vision is OKLab dE 15.1 (population's lighter green against
#: model-based gold).
#:
#: Under red-green colour blindness the set cannot hold: gold against
#: the lightest orange falls to dE 4.0, and purple against the blues to
#: 4.4. Once each large category spends lightness on its own members,
#: dichromats keep only one chromatic axis to separate categories on,
#: and five do not fit on it. So the category is a grouping cue, never
#: the only way to tell a curve apart: that is left to the direct labels
#: (plot_ertd(right_labels=True)) or the legend.
#:
#: Quasi-Newton's purple sits next to gradient-based blue on purpose:
#: L-BFGS is gradient-based too, only with curvature information.
CATEGORY_PALETTES: dict[str, tuple[str, ...]] = {
    CATEGORY_GRADIENT: ("#0458b0", "#1369cb", "#2b7cdf", "#3f8ff4", "#5ea3fd"),
    CATEGORY_QUASI_NEWTON: ("#6c2893",),
    CATEGORY_DISTRIBUTION: (
        "#892d01",
        "#a33806",
        "#bf4306",
        "#d45624",
        "#e9693a",
    ),
    CATEGORY_POPULATION: ("#089d49", "#29b159"),
    CATEGORY_MODEL_BASED: ("#bd9f13",),
}

#: Dash patterns a category cycles its members through, after its
#: shades. Six, so the largest category (27 gradient methods, 5 shades)
#: still gives every member its own shade and pattern.
category_line_styles: list[str | tuple] = [
    "-",
    "--",
    ":",
    "-.",
    (0, (5, 1.5, 1, 1.5, 1, 1.5)),  # dash-dot-dot
    (0, (9, 2.5)),  # long dash
]


def optimizer_category(schedule_name: str) -> str | None:
    """The category a schedule's optimizer belongs to.

    Parameters
    ----------
    schedule_name : str
        Schedule name, e.g. ``"adam_learning_rate=0.001"`` or
        ``"rbf_trust_region"``.

    Returns
    -------
    str or None
        One of the ``CATEGORY_*`` names, or ``None`` for a schedule
        outside every category (random search, the oracles, learned
        policies, an optimizer registered from outside l2co).
    """
    optimizer = _categorised_optimizer(schedule_name)
    return None if optimizer is None else OPTIMIZER_CATEGORIES[optimizer]


def _categorised_optimizer(schedule_name: str) -> str | None:
    """The categorised optimizer a schedule runs, longest name first.

    A schedule is its optimizer's name, optionally followed by
    ``_key=value`` settings. Matching the longest registered name that
    the schedule equals or extends with an underscore is what resolves
    ``rbf_trust_region`` whole, where splitting at the first underscore
    would look for ``rbf``, and keeps ``adamw_...`` off ``adam``.
    """
    candidates = [
        name
        for name in OPTIMIZER_CATEGORIES
        if schedule_name == name or schedule_name.startswith(name + "_")
    ]
    return max(candidates, key=len) if candidates else None


def _styles_by_category(
    names: list[str], styles: dict[str, OptStyle]
) -> dict[str, OptStyle]:
    """Restyle every categorised schedule in its category's hue.

    A member takes shade ``slot % n_shades`` and dash pattern
    ``slot // n_shades``, with ``slot`` its place in
    :data:`CATEGORY_MEMBERS`, so every member of a category differs from
    every other in shade, pattern or both. A second or later variant of
    one optimizer keeps its shade and moves on to the next pattern.
    Markers are left as ``styles`` assigned them.
    """
    variants: dict[str, int] = {}
    restyled = dict(styles)
    for name in names:
        optimizer = _categorised_optimizer(name)
        if optimizer is None:
            continue
        category = OPTIMIZER_CATEGORIES[optimizer]
        shades = CATEGORY_PALETTES[category]
        slot = CATEGORY_MEMBERS[category].index(optimizer)
        variant = variants.get(optimizer, 0)
        variants[optimizer] = variant + 1
        restyled[name] = OptStyle(
            color=shades[slot % len(shades)],
            marker=styles[name].marker,
            linestyle=category_line_styles[
                (slot // len(shades) + variant) % len(category_line_styles)
            ],
        )
    return restyled


#: Display labels for schedules, resolved by :func:`display_name`.
#:
#: Prefer a **bare optimizer name** (``"adam"``): ``display_name`` falls
#: back to it for every hyperparameter variant, so one entry covers
#: ``adam_learning_rate=0.01`` and any future learning rate. Add a
#: suffixed key only to give one specific variant its own label. An
#: optimizer whose own name has an underscore (``rbf_trust_region``)
#: needs its full name as the key, since the fallback splits at the
#: first one.
#:
#: Labels follow the source paper's abbreviation, hyphenated the way the
#: literature writes it (evosax's ``CMA_ES`` is CMA-ES), and sentence
#: case where there is no established abbreviation. Every label is also
#: rendered by LaTeX under the dissertation style (``text.usetex``), so
#: none may carry a bare ``_``, ``&``, ``%`` or ``#``.
LABEL_MAP = {
    # --- evosax, distribution-based ---
    "ars": "ARS",
    "asebo": "ASEBO",
    "cmaes": "CMA-ES",
    "crfmnes": "CR-FM-NES",
    # evosax's DiscoveredES (Lange et al., 2023); BBOB readers know "DES"
    # as the unrelated Differential Evolution Strategy.
    "des": "Discovered ES",
    "esmc": "ESMC",
    "gradientlessdescent": "GLD",
    "guidedes": "Guided ES",
    "hillclimbing": "Hill climbing",
    "iamalgamfull": "iAMaLGaM (full)",
    "iamalgamunivariate": "iAMaLGaM (univariate)",
    "lmmaes": "LM-MA-ES",
    "maes": "MA-ES",
    "noisereusees": "NRES",
    # Salimans et al. (2017); evosax spells it Open_ES.
    "openes": "OpenAI-ES",
    "persistentes": "PES",
    "pgpe": "PGPE",
    "rmes": "Rm-ES",
    "sepcmaes": "Sep-CMA-ES",
    "simulatedannealing": "Simulated annealing",
    "simplees": "Simple ES",
    "snes": "SNES",
    "xnes": "xNES",
    # --- evosax, population-based ---
    "differentialevolution": "DE",
    "diffusionevolution": "Diffusion evolution",
    "gesmrga": "GESMR-GA",
    "mr15ga": "MR15-GA",
    "pso": "PSO",
    "samrga": "SAMR-GA",
    "simplega": "Simple GA",
    # --- optax, gradient-based ---
    "adabelief": "AdaBelief",
    "adadelta": "Adadelta",
    "adafactor": "Adafactor",
    "adagrad": "AdaGrad",
    "adam": "Adam",
    "adamax": "Adamax",
    "adamaxw": "AdamaxW",
    "adamw": "AdamW",
    "adan": "Adan",
    "amsgrad": "AMSGrad",
    "fromage": "Fromage",
    "lamb": "LAMB",
    "lars": "LARS",
    "lion": "Lion",
    "nadam": "NAdam",
    "nadamw": "NAdamW",
    "noisysgd": "Noisy SGD",
    "novograd": "NovoGrad",
    "optimisticadam": "Optimistic Adam",
    "optimisticgradientdescent": "Optimistic GD",
    "radam": "RAdam",
    "rmsprop": "RMSProp",
    "rprop": "Rprop",
    "sgd": "SGD",
    "signsgd": "signSGD",
    "sm3": "SM3",
    "yogi": "Yogi",
    # --- second-order and model-based ---
    "lbfgs": "L-BFGS",
    "rbf_trust_region": "RBF trust region",
    "turbo": "TuRBO",
    # --- baselines, oracles and learned policies ---
    "randomsearch": "Random search",
    "sbs": "SBS",
    "vbs": "VBS (per run)",
    "vbs_task": "VBS (per task)",
    # "l2co": r"\textbf{L2CO (Ours)}",  # You can even use LaTeX bold here
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


def get_schedule_styles(
    schedule_names: Iterable[str],
    by: Literal["optimizer", "category"] = "optimizer",
) -> dict[str, OptStyle]:
    """
    Map schedule names to OptStyle objects with appropriate styling.

    Parameters
    ----------
    schedule_names : Iterable[str]
        Collection of schedule/optimizer names to style.
    by : {"optimizer", "category"}, optional
        What a curve's colour identifies, by default ``"optimizer"``:
        each optimizer's own colour, as tuned in ``BASE_OPT_STYLES``.
        ``"category"`` colours by kind of optimizer instead -- one hue
        per category in :data:`CATEGORY_PALETTES`, members told apart by
        shade and dash pattern -- so a panel of many curves reads as a
        handful of groups. Schedules outside every category (random
        search, the oracles, learned policies) keep their own style
        either way.

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

    A schedule name that either matches a compound (underscore-bearing)
    key in ``BASE_OPT_STYLES`` exactly, or extends one with a further
    ``"_"``-delimited suffix (a *variant pin*, e.g. ``rl2co_random``
    covering ``rl2co_random_v2`` and ``rl2co_random_10``), takes its
    colour from that pin and takes precedence over its base optimizer's
    style. When a name extends more than one such key, the longest key
    wins. A pinned style's linestyle and marker are withheld from the
    base optimizer's own cycle, so a pinned variant never collides with
    a cycled sibling regardless of portfolio order.

    When more than one selected schedule pins to the *same* compound
    key (e.g. the ``rl2co_random_10`` / ``_100`` / ``_1000``
    iteration-budget ablation, all pinned to ``rl2co_random``), they
    would otherwise all render as one identical curve -- the pinned
    style is a colour/shape for the *family*, not a fingerprint for
    each member. So within a pin group of two or more, linestyle and
    marker cycle across the group the same way a base optimizer's own
    hyperparameter variants do (starting from the pin's own linestyle
    and marker), while every member keeps the pin's colour. A pin group
    of exactly one is unaffected and keeps the pinned style verbatim.
    """
    names = list(schedule_names)
    compound_keys = [k for k in BASE_OPT_STYLES if "_" in k]

    def _pin_key(full_name: str) -> str | None:
        """The compound key this name pins to, longest match wins."""
        candidates = [
            k
            for k in compound_keys
            if full_name == k or full_name.startswith(k + "_")
        ]
        return max(candidates, key=len) if candidates else None

    pin_groups: dict[str, list[str]] = {}
    for full_name in names:
        pin_key = _pin_key(full_name)
        if pin_key is not None:
            pin_groups.setdefault(pin_key, []).append(full_name)
    pin_group_linestyles: dict[str, set[str]] = {}
    pin_group_markers: dict[str, set[str]] = {}

    schedule_styles = {}
    # Per base name, the linestyles / markers already handed out. Every
    # pin in the selection is registered up front, so the cycle skips
    # them no matter where they sit in the portfolio order.
    used_linestyles: dict[str, set[str]] = {}
    used_markers: dict[str, set[str]] = {}
    for full_name in names:
        pin_key = _pin_key(full_name)
        if pin_key is None:
            continue
        base_name = full_name.split("_")[0]
        pinned = BASE_OPT_STYLES[pin_key]
        used_linestyles.setdefault(base_name, set()).add(pinned.linestyle)
        if pinned.marker is not None:
            used_markers.setdefault(base_name, set()).add(pinned.marker)

    for full_name in names:
        # 1. Identify the base name (split at the first underscore)
        base_name = full_name.split("_")[0]
        taken_ls = used_linestyles.setdefault(base_name, set())
        taken_mk = used_markers.setdefault(base_name, set())

        # 2. A variant pin wins outright; it was registered above. A
        # solo pin keeps the family style verbatim; a pin group of two
        # or more cycles linestyle/marker across its members so they
        # stay distinguishable from each other, not just from the rest
        # of the base optimizer's family.
        pin_key = _pin_key(full_name)
        if pin_key is not None:
            pinned = BASE_OPT_STYLES[pin_key]
            if len(pin_groups[pin_key]) == 1:
                schedule_styles[full_name] = pinned
                continue

            taken_pin_ls = pin_group_linestyles.setdefault(pin_key, set())
            taken_pin_mk = pin_group_markers.setdefault(pin_key, set())

            chosen_pin_ls = next(
                (ls for ls in line_styles if ls not in taken_pin_ls),
                line_styles[len(taken_pin_ls) % len(line_styles)],
            )
            taken_pin_ls.add(chosen_pin_ls)

            chosen_pin_mk = pinned.marker
            if chosen_pin_mk is not None and chosen_pin_mk in taken_pin_mk:
                chosen_pin_mk = next(
                    (m for m in variant_markers if m not in taken_pin_mk),
                    chosen_pin_mk,
                )
            if chosen_pin_mk is not None:
                taken_pin_mk.add(chosen_pin_mk)

            schedule_styles[full_name] = OptStyle(
                color=pinned.color,
                marker=chosen_pin_mk,
                linestyle=chosen_pin_ls,
            )
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

    match by:
        case "optimizer":
            return schedule_styles
        case "category":
            return _styles_by_category(names, schedule_styles)
        case _:
            raise ValueError(
                f"by must be 'optimizer' or 'category'; got {by!r}."
            )
