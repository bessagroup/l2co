"""Time-shared portfolio pseudo-schedules built from stored runs.

A time-shared portfolio runs K optimizers side by side on one problem,
in round robin and without passing any state between them, and keeps
the best point any of them has found (Gomes & Selman 2001, *Algorithm
portfolios*). :func:`time_shared_portfolio` builds its run histories
offline from the members' stored ones, with no new optimization, as a
reference next to the virtual and single best solvers (l2co#115).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from l2co._src.naming import (
    BUDGET,
    FEVALS,
    ITERATIONS,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    SCHEDULE,
)

#: Granularities of a member's turn: any number of fevals (idealised), or
#: whole iterations (a CR-FM-NES generation cannot be split).
GRANULARITIES = ("fevals", "iterations")


def _at(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``values[..., k, idx[..., k, i]]``: gather along the last axis."""
    return np.take_along_axis(values, idx, axis=-1)


def _last_at_or_before(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the last ``grid[..., j] <= x[..., i]`` (-1 when none).

    ``grid`` is ``(..., budget)`` and nondecreasing; ``x`` is
    ``(..., n)``. A NaN on either side compares false.
    """
    return (grid[..., None, :] <= x[..., :, None]).sum(axis=-1) - 1


def _interp_from_origin(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray
) -> np.ndarray:
    """Batched linear interpolation of ``fp(xp)`` at ``x``, through (0, 0).

    The curve runs from the origin through the stored points and is
    flat after the last one. All arrays share their leading dims.
    """
    zeros = np.zeros_like(xp[..., :1])
    xp = np.concatenate([zeros, xp], axis=-1)
    fp = np.concatenate([zeros, fp], axis=-1)
    lo = np.clip(_last_at_or_before(xp, x), 0, xp.shape[-1] - 1)
    hi = np.minimum(lo + 1, xp.shape[-1] - 1)
    x_lo, x_hi = _at(xp, lo), _at(xp, hi)
    gap = x_hi - x_lo
    frac = np.where(gap > 0, (x - x_lo) / np.where(gap > 0, gap, 1.0), 0.0)
    at_lo = _at(fp, lo)
    # On a stored point the next one is not read (it may be NaN).
    step = np.clip(frac, 0.0, 1.0) * (_at(fp, hi) - at_lo)
    return np.where(frac > 0, at_lo + step, at_lo)


def _feval_turns(
    fevals: np.ndarray,
    iterations: np.ndarray,
    total: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each member's position at total budget ``F_i`` under feval sharing.

    Member k has run ``min(w_k F_i, its own fevals)`` evaluations.
    Returns, per ``(..., member, step)``: the last stored point reached
    (-1: none yet), the fevals spent and the iterations made (linear in
    fevals between stored points).
    """
    share = weights[:, None] * total[..., None, :]
    # Cap at the last known count: a NaN tail spoils only its own steps.
    own = np.fmax.reduce(fevals, axis=-1)[..., None]
    spent = np.minimum(share, own)
    reached = _last_at_or_before(fevals, share)
    made = _interp_from_origin(spent, fevals, iterations)
    return reached, spent, made


def _iteration_turns(
    fevals: np.ndarray,
    iterations: np.ndarray,
    total: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each member's position at total budget ``F_i`` under whole turns.

    Members take turns in order; in round ``c`` member k runs up to
    ``floor(c w_k / min(w))`` iterations in all (fewer once it has run
    out), so equal weights give one iteration per turn. A turn costs
    the fevals its iterations cost, linear in iterations between stored
    points. Step i stops after the last turn that fits in ``F_i``.
    Returns what :func:`_feval_turns` does.
    """
    n_members = weights.size
    pace = weights / weights.min()
    own = np.nan_to_num(np.fmax.reduce(iterations, axis=-1), nan=0.0)
    rounds = int(np.ceil(np.max(own / pace, initial=0.0)))

    def made_after(turn: np.ndarray) -> np.ndarray:
        """Iterations of each member after ``turn`` turns in all."""
        k = np.arange(n_members)[:, None]
        rnd = (turn[..., None, :] - k - 1) // n_members + 1  # rounds had
        made = np.floor(np.maximum(rnd, 0) * pace[:, None] + 1e-9)
        return np.minimum(made, own[..., None])

    def cost(turn: np.ndarray) -> np.ndarray:
        spent = _interp_from_origin(made_after(turn), iterations, fevals)
        return spent.sum(axis=-2)

    # Largest turn count whose cost fits in F_i (cost grows with turns).
    lo = np.zeros(total.shape, dtype=np.int64)
    hi = np.full(total.shape, rounds * n_members, dtype=np.int64)
    while (lo < hi).any():
        mid = (lo + hi + 1) // 2
        fits = cost(mid) <= total
        lo = np.where(fits, mid, lo)
        hi = np.where(fits, hi, mid - 1)

    made = made_after(lo)
    spent = _interp_from_origin(made, iterations, fevals)
    reached = _last_at_or_before(iterations, made)
    return reached, spent, made


_TURNS = {"fevals": _feval_turns, "iterations": _iteration_turns}


def _interleave(
    fevals: np.ndarray,
    iterations: np.ndarray,
    *losses: np.ndarray,
    weights: np.ndarray,
    granularity: str,
) -> tuple[np.ndarray, ...]:
    """The portfolio's runs from its members' ``(..., member, budget)`` runs.

    Returns ``fevals``, ``iterations`` and each of ``losses`` as
    ``(..., budget)``. ``losses[0]`` is ``output_min``: a run that
    lacks it on any member is missing (NaN) as a whole.
    """
    total = fevals.sum(axis=-2)  # F_i: every member's i-th point
    reached, spent, made = _TURNS[granularity](
        fevals, iterations, total, weights
    )
    out = [spent.sum(axis=-2), made.sum(axis=-2)]
    for loss in losses:
        best = np.fmin.accumulate(loss, axis=-1)
        member_best = np.where(
            reached >= 0, _at(best, np.maximum(reached, 0)), np.inf
        )
        portfolio = member_best.min(axis=-2)
        out.append(np.where(np.isinf(portfolio), np.nan, portfolio))
    missing = np.isnan(losses[0]).all(axis=-1).any(axis=-1)[..., None]
    missing = missing | np.isnan(total)  # a step some member lacks
    return tuple(np.where(missing, np.nan, o) for o in out)


def time_shared_portfolio(
    ds: xr.Dataset,
    members: Sequence[str],
    weights: Sequence[float] | None = None,
    granularity: str = "fevals",
) -> xr.Dataset:
    """Pseudo-schedule of a time-shared portfolio of stored schedules.

    Run r of the portfolio interleaves run r of every member (runs are
    paired along every dimension but ``schedule`` and ``budget``; the
    databank's common random numbers give paired runs the same
    starting points). Member k gets the share ``w_k`` of the function
    evaluations, so with ``best_k(f)`` member k's best-so-far after f
    fevals the portfolio's is

        best_portfolio(F) = min_k best_k(w_k * F).

    Step i of the result is the portfolio at the total budget ``F_i``,
    the sum of every member's fevals at its own step i, so the members'
    grids are the shared one: the portfolio keeps their ``budget``
    coordinate, and members that spend the same fevals per step are
    each at their own step i there. ``fevals`` and ``iterations`` are
    what the members have spent in all by then -- the portfolio's own
    count, at most ``F_i``, less once a member has run out. ``F_i`` runs
    up to the sum of the members' budgets, and at a total budget F each
    member is used up to ``w_k * F`` only: cut the portfolio at the
    evaluation budget B as for a real run (``calculate_ert(...,
    evaluation_budget=B)``) and each member's steps past ``w_k * B``
    go unused.

    Between stored steps a member's best-so-far is a step function, the
    store's convention: at f fevals it is the best-so-far at the
    member's last stored step with at most f fevals (none before its
    first). Its fevals and iterations are linear between stored steps
    and from (0, 0) to the first.

    With equal weights and ``granularity="fevals"``, every target the
    fastest member reaches after h fevals the portfolio reaches within
    ``K * h`` when the members spend the same fevals per step; otherwise
    within the next step of the shared grid after ``K * h``.

    Parameters
    ----------
    ds : xr.Dataset
        Stored runs with ``schedule`` and ``budget`` dims and the
        variables ``output_min``, ``fevals``, ``iterations`` and, for
        noisy tasks, ``output_min_true`` -- a databank read
        (``l2co.databank.load_databank_dataset``, or a
        ``DataLoader``'s ``data``) or a loader's dataset. Gaps or
        losses alike: the portfolio takes minima only. Lazy data stays
        lazy.
    members : sequence of str
        The member schedules, by name. Their order is the turn order.
    weights : sequence of float or None, optional
        Each member's share, positive, normalized to sum 1. ``None``
        (default) shares equally.
    granularity : {"fevals", "iterations"}, optional
        ``"fevals"`` (default) is the idealized portfolio: a member's
        turn may stop inside an iteration, so the members spend exactly
        their shares. ``"iterations"`` runs whole iterations, as a real
        round robin would (a CR-FM-NES generation cannot be split): the
        members take turns in order; in round c member k runs up to
        ``floor(c * w_k / min(w))`` iterations in all, so the weights
        scale each turn's length (one iteration a turn for equal
        weights), and a turn costs the fevals its iterations cost,
        linear in iterations between stored steps. Step i stops after
        the last turn that fits in ``F_i``.

    Returns
    -------
    xr.Dataset
        One schedule, ``portfolio(<m1>+<m2>+...)``, with ``output_min``,
        ``fevals``, ``iterations`` and, when ``ds`` has it,
        ``output_min_true`` -- built from the members' noise-free series
        alike, so :func:`l2co.data.scoring_loss` scores it as it scores
        a real run (ADR 0024). The variables of ``ds`` without a
        ``schedule`` dim (labels, coordinates) are carried over; its
        other per-schedule variables are not. ``attrs`` mark it as
        derived, not run: ``derived="time_shared_portfolio"`` and the
        members, normalized weights and granularity. Concatenate it
        with the members along ``schedule`` to score it next to them.
        A run that a member lacks (``output_min`` NaN at every step) is
        NaN in every variable; a step whose ``fevals`` a member lacks is
        NaN in every variable at that step only.

    Raises
    ------
    ValueError
        If ``weights`` are not one positive number per member, or
        ``granularity`` is not one of ``"fevals"``, ``"iterations"``.
    KeyError
        If a member is not a schedule of ``ds``.
    """
    members = list(members)
    name = f"portfolio({'+'.join(members)})"
    if weights is None:
        weights = np.ones(len(members))
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(members),) or not (w > 0).all():
        raise ValueError(
            f"weights must be {len(members)} positive numbers, one per "
            f"member, got {weights!r}."
        )
    w = w / w.sum()
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {GRANULARITIES}, got {granularity!r}."
        )
    sel = ds.sel({SCHEDULE: members})
    losses = [v for v in (OUTPUT_MIN, OUTPUT_MIN_TRUE) if v in sel]
    names = [FEVALS, ITERATIONS, *losses]
    core = [SCHEDULE, BUDGET]
    sel = sel[names]
    if any(sel[v].chunks for v in names):  # lazy: one chunk per history
        sel = sel.unify_chunks().chunk({SCHEDULE: -1, BUDGET: -1})
    out = xr.apply_ufunc(
        _interleave,
        *(sel[v] for v in names),
        kwargs={"weights": w, "granularity": granularity},
        input_core_dims=[core] * len(names),
        output_core_dims=[[BUDGET]] * len(names),
        dask="parallelized",
        output_dtypes=[float] * len(names),
    )
    runs = {
        v: da.expand_dims({SCHEDULE: [name]})
        for v, da in zip(names, out, strict=True)
    }
    # Per-run labels (no schedule dim) carry over; other per-schedule
    # variables of the members have no portfolio counterpart.
    shared = ds.drop_vars([v for v in ds.variables if SCHEDULE in ds[v].dims])
    result = shared.assign(runs)
    result.attrs = {
        "derived": "time_shared_portfolio",
        "portfolio_members": members,
        "portfolio_weights": w.tolist(),
        "portfolio_granularity": granularity,
    }
    return result
