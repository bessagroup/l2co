"""Unit tests for colouring ERTD curves by optimizer category.

Covers where the categories come from (the optimizer registries, with
random search and the learned policies left out), that members of one
category never share a style, that a member's style does not depend on
which schedules are plotted, that the palette keeps its normal-vision
floor between categories, and the key ``plot_ertd`` draws for it.
"""

import itertools

import matplotlib
import numpy as np
import pytest

import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from l2co._src.naming import SCHEDULE  # noqa: E402
from l2co._src.visualize.colors import (  # noqa: E402
    BASE_OPT_STYLES,
    CATEGORY_DIRECT_SEARCH,
    CATEGORY_GRADIENT,
    CATEGORY_MEMBERS,
    CATEGORY_MODEL_BASED,
    CATEGORY_PALETTES,
    CATEGORY_QUASI_NEWTON,
    OPTIMIZER_CATEGORIES,
    get_schedule_styles,
    optimizer_category,
)
from l2co._src.visualize.plotting import plot_ertd  # noqa: E402
from l2co.optimizers import meta_optimizers  # noqa: E402
from l2co_optimizers import normalize_key, optimizers  # noqa: E402

# =============================================================================


def _style_key(style):
    return (style.color, str(style.linestyle))


#                                                                  Categories
# =============================================================================


def test_every_registered_optimizer_has_a_category_but_the_references():
    categorised = {normalize_key(name) for name in OPTIMIZER_CATEGORIES}
    registered = set(optimizers) | set(meta_optimizers)
    assert registered - categorised == {"l2co", "randomsearch"}


@pytest.mark.parametrize(
    "name, category",
    [
        ("bfgs", CATEGORY_QUASI_NEWTON),
        ("dfp", CATEGORY_QUASI_NEWTON),
        ("ipopt", CATEGORY_QUASI_NEWTON),
        ("lbfgsb", CATEGORY_QUASI_NEWTON),
        ("slsqp", CATEGORY_QUASI_NEWTON),
        ("tnc", CATEGORY_QUASI_NEWTON),
        ("trustconstr", CATEGORY_QUASI_NEWTON),
        ("trustkrylov", CATEGORY_QUASI_NEWTON),
        ("nonlinearcg", CATEGORY_GRADIENT),
        ("cobyqa", CATEGORY_MODEL_BASED),
        ("neldermead", CATEGORY_DIRECT_SEARCH),
        ("powell", CATEGORY_DIRECT_SEARCH),
    ],
)
def test_optimizers_named_by_hand_land_in_their_categories(name, category):
    # Named in colors.py because l2co-optimizers keeps their tables
    # private; a misspelt name would categorise nothing.
    assert name in optimizers
    assert optimizer_category(name) == category


def test_no_optimizer_sits_in_two_categories():
    members = [m for ms in CATEGORY_MEMBERS.values() for m in ms]
    assert len(members) == len(set(members))


def test_quasi_newton_is_its_own_category():
    assert optimizer_category("lbfgs") == CATEGORY_QUASI_NEWTON
    assert optimizer_category("adam_learning_rate=0.001") == CATEGORY_GRADIENT


def test_an_optimizer_with_an_underscore_is_categorised_whole():
    # Splitting at the first underscore would look up "rbf".
    assert optimizer_category("rbf_trust_region") == CATEGORY_MODEL_BASED


def test_a_longer_name_is_not_mistaken_for_its_prefix():
    styles = get_schedule_styles(
        ["adam_learning_rate=0.001", "adamw_learning_rate=0.001"],
        by="category",
    )
    assert _style_key(styles["adam_learning_rate=0.001"]) != _style_key(
        styles["adamw_learning_rate=0.001"]
    )


def test_references_and_learned_policies_have_no_category():
    for name in ["randomsearch", "vbs", "sbs", "rl2co_10", "l2co_s0-1"]:
        assert optimizer_category(name) is None


#                                                            Category styles
# =============================================================================


@pytest.mark.parametrize("category", sorted(CATEGORY_MEMBERS))
def test_members_of_a_category_never_share_a_style(category):
    styles = get_schedule_styles(CATEGORY_MEMBERS[category], by="category")
    keys = [_style_key(style) for style in styles.values()]
    assert len(keys) == len(set(keys))
    assert {s.color for s in styles.values()} <= set(
        CATEGORY_PALETTES[category]
    )


def test_categories_share_no_colour():
    colours = [c for shades in CATEGORY_PALETTES.values() for c in shades]
    assert len(colours) == len(set(colours))


def test_a_members_style_does_not_depend_on_what_else_is_plotted():
    everything = [m for ms in CATEGORY_MEMBERS.values() for m in ms]
    full = get_schedule_styles(everything, by="category")
    subset = get_schedule_styles(["cmaes", "sgd", "turbo"], by="category")
    for name, style in subset.items():
        assert _style_key(style) == _style_key(full[name])


def test_variants_of_one_optimizer_keep_its_shade_and_differ_in_pattern():
    names = ["adam_learning_rate=0.001", "adam_learning_rate=0.01"]
    a, b = get_schedule_styles(names, by="category").values()
    assert a.color == b.color
    assert a.linestyle != b.linestyle


def test_uncategorised_schedules_keep_their_own_style():
    names = ["randomsearch", "vbs", "cmaes"]
    by_category = get_schedule_styles(names, by="category")
    assert by_category["randomsearch"] == BASE_OPT_STYLES["randomsearch"]
    assert by_category["vbs"] == BASE_OPT_STYLES["vbs"]


def test_the_default_styling_is_unchanged():
    names = ["lbfgs", "rprop_learning_rate=0.001", "crfmnes", "mr15ga"]
    assert get_schedule_styles(names) == get_schedule_styles(
        names, by="optimizer"
    )
    assert get_schedule_styles(["lbfgs"])["lbfgs"].color == "#0AA0D2"


def test_an_unknown_styling_is_rejected():
    with pytest.raises(ValueError, match="optimizer' or 'category"):
        get_schedule_styles(["cmaes"], by="family")


#                                                             Palette floor
# =============================================================================


def _oklab(hex_colour: str) -> np.ndarray:
    rgb = np.array(matplotlib.colors.to_rgb(hex_colour))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = (
        np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        )
        @ lin
    )
    return np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    ) @ np.cbrt(lms)


def test_categories_stay_apart_for_full_colour_vision():
    # OKLab dE x100 >= 15 between any two categories, the reference
    # lines' black and grey counted as one more: the floor below which
    # neighbours are hard to tell apart even with full colour vision.
    groups = [*CATEGORY_PALETTES.values(), ("#000000", "#888888")]
    worst = min(
        100 * np.linalg.norm(_oklab(a) - _oklab(b))
        for g1, g2 in itertools.combinations(groups, 2)
        for a in g1
        for b in g2
    )
    assert worst >= 15.0


#                                                              Category key
# =============================================================================


def _profile(names: list[str]) -> xr.DataArray:
    x = np.logspace(-1, 5, 20)
    return xr.DataArray(
        np.vstack(
            [
                np.linspace(0.0, (i + 1) / len(names), x.size)
                for i in range(len(names))
            ]
        ),
        dims=(SCHEDULE, "cost_per_dim"),
        coords={SCHEDULE: names, "cost_per_dim": x},
    )


def test_the_key_names_the_categories_drawn_in_fixed_order():
    names = ["turbo", "sgd_learning_rate=0.001", "cmaes", "randomsearch"]
    fig, ax = plt.subplots()
    plot_ertd(
        ertd=_profile(names),
        ax=ax,
        right_labels=True,
        style_by="category",
    )
    key = ax.get_legend()
    assert [t.get_text() for t in key.get_texts()] == [
        "Gradient-based",
        "Distribution-based",
        "Model-based",
    ]
    plt.close(fig)


def test_the_key_keeps_the_boxed_legend():
    fig, ax = plt.subplots()
    plot_ertd(
        ertd=_profile(["cmaes", "lbfgs"]),
        ax=ax,
        show_legend=True,
        style_by="category",
    )
    boxed = [a for a in ax.artists if isinstance(a, matplotlib.legend.Legend)]
    assert len(boxed) == 1
    assert ax.get_legend() is not None
    plt.close(fig)


def test_no_key_without_category_styling():
    fig, ax = plt.subplots()
    plot_ertd(ertd=_profile(["cmaes", "lbfgs"]), ax=ax, right_labels=True)
    assert ax.get_legend() is None
    plt.close(fig)
