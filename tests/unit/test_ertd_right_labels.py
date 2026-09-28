"""Unit tests for the right-hand curve-label column of ``plot_ertd``.

Covers the label-spreading geometry (no overlap, ordering preserved,
bounds respected, the infeasible fallback), the curve-end pick that
anchors each label, and a render of the whole thing: that the column
really is empty of data, grid lines and right-hand ticks, that every
curve is named exactly once in black, that each connector is drawn in
its curve's own style, and that the boxed-legend path is left alone.
"""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from l2co._src.naming import SCHEDULE  # noqa: E402
from l2co._src.visualize.colors import display_name  # noqa: E402
from l2co._src.visualize.plotting import (  # noqa: E402
    MIN_RIGHT_LABEL_FONTSIZE,
    _curve_end,
    _CurveEnd,
    _disambiguate,
    _settings,
    _spread_labels,
    plot_ertd,
)

# =============================================================================


def _gaps(positions: list[float]) -> np.ndarray:
    return np.diff(np.sort(np.asarray(positions)))


#                                                             _spread_labels
# =============================================================================


def test_spread_labels_leaves_roomy_targets_alone():
    targets = [0.1, 0.5, 0.9]
    assert _spread_labels(targets, min_gap=0.05) == pytest.approx(targets)


def test_spread_labels_separates_coincident_targets():
    # The saturating case: every curve finishes at 1.0.
    positions = _spread_labels([1.0] * 5, min_gap=0.1)
    assert _gaps(positions) == pytest.approx(0.1)
    assert min(positions) >= 0.0 and max(positions) <= 1.0


def test_spread_labels_keeps_input_order():
    # Third target is the highest, so it must come back the highest.
    positions = _spread_labels([0.5, 0.5, 0.9, 0.1], min_gap=0.2)
    assert positions[2] == max(positions)
    assert positions[3] == min(positions)


def test_spread_labels_respects_bounds_when_crowded_at_the_top():
    positions = _spread_labels([0.95, 0.96, 0.97, 0.98], min_gap=0.2, hi=1.0)
    assert max(positions) <= 1.0 + 1e-12
    assert min(positions) >= 0.0 - 1e-12
    assert _gaps(positions).min() >= 0.2 - 1e-12


def test_spread_labels_respects_bounds_when_crowded_at_the_bottom():
    positions = _spread_labels([0.0, 0.01, 0.02], min_gap=0.3, lo=0.0, hi=1.0)
    assert min(positions) >= 0.0 - 1e-12
    assert max(positions) <= 1.0 + 1e-12
    assert _gaps(positions).min() >= 0.3 - 1e-12


def test_spread_labels_falls_back_to_even_spacing_when_it_cannot_fit():
    # 5 labels needing 0.3 apiece do not fit in a unit interval; the
    # least-bad answer is even spacing rather than a crash.
    positions = _spread_labels([0.5] * 5, min_gap=0.3)
    assert positions == pytest.approx(np.linspace(0.0, 1.0, 5))


def test_spread_labels_handles_no_labels():
    assert _spread_labels([], min_gap=0.1) == []


#                                                                 _curve_end
# =============================================================================


def test_curve_end_takes_the_last_on_axis_point():
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    end = _curve_end(x=x, y=y, xmax=100.0)
    assert end is not None
    assert end[0] == 100.0
    assert end[1] == pytest.approx(0.3)
    # Mean is over the on-axis points only.
    assert end[2] == pytest.approx(0.2)


def test_curve_end_skips_trailing_nans():
    x = np.array([1.0, 10.0, 100.0])
    y = np.array([0.5, 0.6, np.nan])
    end = _curve_end(x=x, y=y, xmax=1e5)
    assert end is not None and end[0] == 10.0 and end[1] == pytest.approx(0.6)


def test_curve_end_is_none_when_nothing_is_drawn():
    x = np.array([1.0, 10.0])
    assert _curve_end(x=x, y=np.array([np.nan, np.nan]), xmax=1e5) is None
    # Everything past the right edge is off-axis, so nothing to label.
    assert _curve_end(x=x, y=np.array([0.1, 0.2]), xmax=0.5) is None


#                                                        plot_ertd rendering
# =============================================================================


def _profile(n_schedules: int, saturating: bool = False) -> xr.DataArray:
    x = np.logspace(-1, 5, 50)
    rows = [
        np.ones_like(x)
        if saturating
        else np.linspace(0.0, (i + 1) / n, x.size)
        for i, n in ((i, n_schedules) for i in range(n_schedules))
    ]
    return xr.DataArray(
        np.vstack(rows),
        dims=(SCHEDULE, "cost_per_dim"),
        coords={
            SCHEDULE: [f"opt{i}" for i in range(n_schedules)],
            "cost_per_dim": x,
        },
    )


def _long_named_profile() -> xr.DataArray:
    """A profile whose first schedule cannot fit the default column."""
    ertd = _profile(5)
    return ertd.assign_coords(
        {
            SCHEDULE: ["a_very_long_optimizer_name_indeed"]
            + list(ertd[SCHEDULE].values[1:])
        }
    )


def test_right_labels_names_every_curve_once():
    fig, ax = plt.subplots()
    plot_ertd(ertd=_profile(6), ax=ax, right_labels=True, show_legend=False)
    named = {t.get_text() for t in ax.texts}
    assert named == {f"opt{i}" for i in range(6)}
    assert ax.get_legend() is None
    plt.close(fig)


def test_right_labels_reserve_an_empty_column():
    fig, ax = plt.subplots()
    xlim = (1e-1, 1e5)
    plot_ertd(
        ertd=_profile(4),
        ax=ax,
        right_labels=True,
        right_label_band=0.25,
        xlim=xlim,
    )
    # The axis runs past the data so the column is empty ...
    assert ax.get_xlim()[1] > xlim[1]
    # ... and it stays empty: no tick, and so no grid line, lands in it.
    assert max(ax.get_xticks()) <= xlim[1]
    assert max(ax.get_xticks(minor=True)) <= xlim[1]
    # The reserved share is what was asked for.
    left, right = (np.log10(v) for v in ax.get_xlim())
    assert (right - np.log10(xlim[1])) / (right - left) == pytest.approx(0.25)
    plt.close(fig)


def test_right_labels_take_precedence_over_the_boxed_legend():
    fig, ax = plt.subplots()
    plot_ertd(ertd=_profile(3), ax=ax, right_labels=True, show_legend=True)
    assert ax.get_legend() is None
    assert len(ax.texts) == 3
    plt.close(fig)


def test_boxed_legend_still_draws_without_right_labels():
    fig, ax = plt.subplots()
    plot_ertd(ertd=_profile(3), ax=ax, show_legend=True)
    assert ax.get_legend() is not None
    assert ax.get_xlim()[1] == pytest.approx(1e5)
    plt.close(fig)


def test_right_labels_stack_saturated_curves_without_overlap():
    # Every curve ends at 1.0, the case a boxed legend hides and a naive
    # direct label collapses onto one line.
    fig, ax = plt.subplots()
    plot_ertd(
        ertd=_profile(8, saturating=True),
        ax=ax,
        right_labels=True,
        right_label_fontsize=4.0,
    )
    heights = sorted(t.get_position()[1] for t in ax.texts)
    assert len(heights) == 8
    assert np.diff(heights).min() > 0.0
    plt.close(fig)


def test_right_labels_warn_when_too_many_curves_to_fit_the_height():
    fig, ax = plt.subplots(figsize=(3.0, 1.0))
    with pytest.warns(UserWarning, match="fewer schedules"):
        plot_ertd(ertd=_profile(60), ax=ax, right_labels=True)
    plt.close(fig)


def test_right_labels_warn_when_a_name_is_too_long_for_the_column():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    with pytest.warns(UserWarning, match="right_label_band"):
        plot_ertd(ertd=_long_named_profile(), ax=ax, right_labels=True)
    plt.close(fig)


def test_pinned_label_fontsize_is_used_verbatim():
    fig, ax = plt.subplots()
    plot_ertd(
        ertd=_profile(4), ax=ax, right_labels=True, right_label_fontsize=6.5
    )
    assert {t.get_fontsize() for t in ax.texts} == {6.5}
    plt.close(fig)


@pytest.mark.parametrize("band", [0.0, 1.0, -0.1, 1.5])
def test_invalid_label_band_is_rejected(band):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="right_label_band"):
        plot_ertd(
            ertd=_profile(3), ax=ax, right_labels=True, right_label_band=band
        )
    plt.close(fig)


def test_an_overlong_name_shrinks_rather_than_overflowing():
    # Shrinking is lossless; letting the column run past the axes would
    # be cropped away by the saved canvas, silently losing the name.
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    with pytest.warns(UserWarning):
        plot_ertd(ertd=_long_named_profile(), ax=ax, right_labels=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_right = ax.get_window_extent(renderer).x1
    for text in ax.texts:
        assert text.get_window_extent(renderer).x1 <= axes_right + 1.0
    assert min(t.get_fontsize() for t in ax.texts) < MIN_RIGHT_LABEL_FONTSIZE
    plt.close(fig)


def test_colliding_display_names_are_told_apart_by_their_settings():
    # display_name maps every adam_* variant onto "Adam"; a column of
    # rows all reading "Adam" names nothing, so each names its setting.
    x = np.logspace(-1, 5, 20)
    names = [
        "adam_learning_rate=0.001",
        "adam_learning_rate=0.01",
        "lbfgs",
    ]
    ertd = xr.DataArray(
        np.vstack([np.linspace(0.0, 0.5, x.size) for _ in names]),
        dims=(SCHEDULE, "cost_per_dim"),
        coords={SCHEDULE: names, "cost_per_dim": x},
    )
    fig, ax = plt.subplots()
    plot_ertd(ertd=ertd, ax=ax, right_labels=True)
    labelled = {t.get_text() for t in ax.texts}
    assert labelled == {
        r"Adam ($\eta$ = 0.001)",
        r"Adam ($\eta$ = 0.01)",
        # Uncontested, so it keeps its pretty label.
        "L-BFGS",
    }
    plt.close(fig)


def test_the_boxed_legend_keeps_the_collapsed_labels():
    # The fallback is scoped to the label column: a small-N legend is
    # where collapsing variants onto one family name is the point.
    x = np.logspace(-1, 5, 20)
    names = ["adam_learning_rate=0.001", "adam_learning_rate=0.01"]
    ertd = xr.DataArray(
        np.vstack([np.linspace(0.0, 0.5, x.size) for _ in names]),
        dims=(SCHEDULE, "cost_per_dim"),
        coords={SCHEDULE: names, "cost_per_dim": x},
    )
    fig, ax = plt.subplots()
    plot_ertd(ertd=ertd, ax=ax, show_legend=True)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Adam",
        "Adam",
    ]
    plt.close(fig)


def _connectors(ax) -> list:
    """The label connectors: the only lines placed in axes coordinates."""
    return [line for line in ax.lines if line.get_transform() is ax.transAxes]


def _styled_profile() -> xr.DataArray:
    """Curves in solid, dashed, dash-dot and dotted styles."""
    x = np.logspace(-1, 5, 20)
    names = [
        "adam_learning_rate=0.001",
        "adam_learning_rate=0.01",
        "adam_learning_rate=0.1",
        "randomsearch",
    ]
    return xr.DataArray(
        np.vstack([np.linspace(0.0, 0.2 * (i + 1), x.size) for i in range(4)]),
        dims=(SCHEDULE, "cost_per_dim"),
        coords={SCHEDULE: names, "cost_per_dim": x},
    )


def test_connectors_are_drawn_exactly_like_their_curves():
    # A thinner or solid connector reads as a different line, so a
    # dotted curve must stay dotted all the way to its name.
    fig, ax = plt.subplots()
    plot_ertd(ertd=_styled_profile(), ax=ax, right_labels=True)
    connectors = _connectors(ax)
    curves = [line for line in ax.lines if line not in connectors]

    def style(line):
        return (
            matplotlib.colors.to_hex(line.get_color()),
            line.get_linestyle(),
            line.get_linewidth(),
        )

    assert len(connectors) == len(curves) == 4
    assert sorted(map(style, connectors)) == sorted(map(style, curves))
    assert {line.get_linestyle() for line in connectors} >= {"--", ":"}
    plt.close(fig)


def test_connectors_slant_across_a_fifth_of_the_column():
    # The slant is what separates neighbouring connectors; squeezed into
    # a sliver of the column they become near-vertical and unreadable.
    band = 0.4
    fig, ax = plt.subplots()
    plot_ertd(
        ertd=_profile(6), ax=ax, right_labels=True, right_label_band=band
    )
    for connector in _connectors(ax):
        xs = connector.get_xdata()
        assert xs[1] - xs[0] >= 0.2 * band
    plt.close(fig)


def test_label_text_is_black():
    fig, ax = plt.subplots()
    plot_ertd(ertd=_styled_profile(), ax=ax, right_labels=True)
    black = matplotlib.colors.to_rgba("black")
    assert {matplotlib.colors.to_rgba(t.get_color()) for t in ax.texts} == {
        black
    }
    plt.close(fig)


def test_label_column_has_no_horizontal_grid_or_right_ticks():
    band = 0.3
    fig, ax = plt.subplots()
    plot_ertd(
        ertd=_profile(4), ax=ax, right_labels=True, right_label_band=band
    )
    fig.canvas.draw()
    column_left = ax.transAxes.transform((1.0 - band, 0.0))[0]
    ticks = [*ax.yaxis.get_major_ticks(), *ax.yaxis.get_minor_ticks()]
    assert ticks
    # Every tick is checked after the draw, so ticks matplotlib only
    # created while drawing are covered too. (A rectangular clip lands
    # on the clip *box*; matplotlib keeps the clip path for other shapes.)
    for tick in ticks:
        assert tick.gridline.get_clip_box().x1 <= column_left + 1e-6
        # The right-hand spine borders the labels, not the data.
        assert not tick.tick2line.get_visible()
    # The left spine keeps its ticks.
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_visible()
    plt.close(fig)


def test_boxed_legend_keeps_right_ticks_and_full_grid():
    fig, ax = plt.subplots()
    plot_ertd(ertd=_profile(3), ax=ax, show_legend=True)
    fig.canvas.draw()
    tick = ax.yaxis.get_major_ticks()[0]
    assert tick.tick2line.get_visible()
    assert tick.gridline.get_clip_box().x1 == pytest.approx(ax.bbox.x1)
    plt.close(fig)


#                                                 _settings / _disambiguate
# =============================================================================


def _labelled(*names: str) -> list[str]:
    ends = [
        _CurveEnd(
            name=name,
            label=display_name(name),
            color="k",
            linestyle="-",
            marker=None,
            linewidth=1.0,
            x=1.0,
            y=0.5,
            mean=0.5,
        )
        for name in names
    ]
    return [end.label for end in _disambiguate(ends)]


def test_settings_parse_keys_that_hold_underscores():
    assert _settings("adam_learning_rate=0.001_b1=0.9") == {
        "learning_rate": "0.001",
        "b1": "0.9",
    }
    assert _settings("lbfgs") == {}
    assert _settings("adadelta") == {}


def test_settings_the_group_shares_are_left_out():
    # b1 is the same for both, so naming it would only add width.
    assert _labelled(
        "adam_learning_rate=0.001_b1=0.9", "adam_learning_rate=0.01_b1=0.9"
    ) == [r"Adam ($\eta$ = 0.001)", r"Adam ($\eta$ = 0.01)"]


def test_an_unlisted_setting_shows_under_its_own_name_without_underscores():
    # A bare underscore would break the dissertation style's LaTeX pass.
    assert _labelled(
        "adam_learning_rate=0.001_weight_decay=0.1",
        "adam_learning_rate=0.001_weight_decay=0.01",
    ) == ["Adam (weight decay = 0.1)", "Adam (weight decay = 0.01)"]


def test_a_collision_the_settings_cannot_separate_keeps_raw_names():
    # Both resolve to "RL2CO" through the base-name fallback, and carry
    # no key=value settings to tell them apart.
    assert _labelled("rl2co_run-a", "rl2co_run-b") == [
        "rl2co_run-a",
        "rl2co_run-b",
    ]


def test_uncontested_labels_are_left_alone():
    assert _labelled("adam_learning_rate=0.001", "lbfgs") == ["Adam", "L-BFGS"]
