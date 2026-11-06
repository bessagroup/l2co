"""The protocol stamp: which run protocol a store's numbers belong to.

A store's numbers depend on the run protocol that made them: the
evaluation budget, the starting points, how noisy tasks are scored, the
task hashes. Two stores written under different protocols are not
comparable, yet they can share a dataset name -- the dataset paper's
re-run writes its stores under their old names in a new root. So every
new store carries a stamp naming the protocol of the code that wrote it,
and the databank refuses a store whose stamp is missing or different
(ADR 0025):

- every reader entry point (:func:`l2co._src.databank.reader._open_store`),
  so a launch pointed at the wrong root stops at its first read;
- ``mode="append"``, before anything is restored, snapshotted or grown;
- ``mode="replace"``, before anything is deleted: removing an old store
  is a deliberate act, done by hand.

:func:`l2co._src.databank.verify.verify_databank` audits any store and
reports a stamp problem as a finding instead.

The stamp is a small JSON file inside the store
(:func:`l2co._src.databank.layout.stamp_path`), written once when a store
is laid out. It is not a zarr root attribute: every ``to_zarr(mode="a")``
the writer makes replaces those.

:data:`DATABANK_PROTOCOL` is set by hand. Whoever changes what a stored
number means bumps it in the same change. A store without a stamp was
written by l2co 1.8.x or earlier.

The stamp can also record the caller's **run settings** (ADR 0027): a
plain JSON mapping of everything that defines a cell (the budget rule,
the sampler and its parameters, the seed, ``n_realizations``), opaque to
l2co. It is written at creation and compared for equality on every
append and by :func:`check_run_settings`, which a create step calls
before any cell runs. A store without settings and a caller without
settings behave as before.

:func:`check_f_star` is the create step's check of each task's f*
against the store's ``f_star`` variable, the same one an append makes
(:class:`FStarError`, ADR 0027). It is not part of the stamp.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from l2co._src.databank import layout
from l2co.naming import F_STAR, TASK_ID

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: The run protocol this code reads and writes. Set by hand; bump it in
#: any change after which newly written numbers are not comparable with
#: stored ones. ``"dataset-paper-2026-11"`` is the dataset paper's
#: protocol revision (2026-11); ``".1"`` stores losses as gaps to f*
#: (ADR 0027).
DATABANK_PROTOCOL: str = "dataset-paper-2026-11.1"

#: The stamp key holding the caller's run settings (ADR 0027).
RUN_SETTINGS_KEY = "run_settings"

#: What follows a refusal, by what was refused.
_ADVICE: dict[str, str] = {
    "read": (
        " Read it with the l2co that wrote it (l2co 1.8.x for a store "
        "without a stamp), or point databank_root at a store written "
        "under protocol '{protocol}'."
    ),
    "append": (
        " Nothing was written. Append to a store written under protocol "
        "'{protocol}', or start a new store under a new databank_root."
    ),
    "replace": (
        " mode='replace' would delete it, so nothing was deleted. If "
        "replacing it is intended, delete it by hand first."
    ),
}


class StoreProtocolError(RuntimeError):
    """A store's protocol stamp is missing or names another protocol."""


class RunSettingsError(StoreProtocolError):
    """A store's stamp records other run settings than the caller's."""


class FStarError(ValueError):
    """A store holds another f* for a task than the caller's (ADR 0027)."""


def normalize_run_settings(
    run_settings: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return ``run_settings`` as it reads back from JSON.

    Tuples become lists and keys become strings, so two mappings that
    are equal as JSON compare equal. ``None`` stays ``None``.

    Raises
    ------
    TypeError
        ``run_settings`` is not a mapping, or not JSON-serializable.
    """
    if run_settings is None:
        return None
    if not isinstance(run_settings, Mapping):
        raise TypeError(
            "run_settings must be a mapping (a plain JSON object), got "
            f"{type(run_settings).__name__}."
        )
    try:
        return json.loads(json.dumps(dict(run_settings), allow_nan=False))
    except (TypeError, ValueError) as err:
        raise TypeError(
            f"run_settings must be plain JSON (stored in the stamp): {err}"
        ) from err


def _l2co_version() -> str:
    try:
        return version("l2co")
    except PackageNotFoundError:
        return "unknown"


def write_stamp(
    store: Path, run_settings: Mapping[str, Any] | None = None
) -> None:
    """Stamp a newly laid-out store with :data:`DATABANK_PROTOCOL`.

    Parameters
    ----------
    store : pathlib.Path
        The store directory (``<root>/<dataset_name>.zarr``).
    run_settings : Mapping or None, optional
        The caller's run settings, stored under ``"run_settings"``. Left
        out of the stamp when ``None``.
    """
    stamp = {
        "protocol": DATABANK_PROTOCOL,
        "l2co_version": _l2co_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    settings = normalize_run_settings(run_settings)
    if settings is not None:
        stamp[RUN_SETTINGS_KEY] = settings
    path = Path(store) / layout.STAMP_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=2) + "\n")
    os.replace(tmp, path)


def store_stamp(root: Path, dataset_name: str) -> dict | None:
    """Return a store's protocol stamp.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    dict or None
        ``{"protocol", "l2co_version", "created"}``, plus
        ``"run_settings"`` when the store was created with them, or
        ``None`` for a store written before stores were stamped.

    Raises
    ------
    FileNotFoundError
        The store does not exist.
    ValueError
        The stamp file is not readable JSON.
    """
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        raise FileNotFoundError(f"No zarr store found at {store}.")
    return _read_stamp(store)


def _read_stamp(store: Path) -> dict | None:
    path = Path(store) / layout.STAMP_NAME
    if not path.exists():
        return None
    try:
        stamp = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"unreadable protocol stamp {path}: {err}") from err
    if not isinstance(stamp, dict):
        raise ValueError(f"unreadable protocol stamp {path}: not an object")
    return stamp


def protocol_problem(store: Path) -> str | None:
    """Describe what is wrong with a store's stamp, or ``None`` if nothing.

    The phrase completes "the store ...", e.g. "carries no protocol
    stamp ...". Shared by :func:`check_protocol` and the verify finding.
    """
    try:
        stamp = _read_stamp(store)
    except ValueError as err:
        return f"has an {err}"
    if stamp is None:
        return (
            "carries no protocol stamp: it was written before stores were "
            "stamped (by l2co 1.8.x or earlier)"
        )
    stored = stamp.get("protocol")
    if stored == DATABANK_PROTOCOL:
        return None
    return (
        f"was written under protocol '{stored}' (by l2co "
        f"{stamp.get('l2co_version', 'unknown')})"
    )


def check_protocol(
    store: Path,
    dataset_name: str,
    *,
    action: Literal["read", "append", "replace"],
) -> None:
    """Refuse a store whose stamp is missing or names another protocol.

    Parameters
    ----------
    store : pathlib.Path
        The (existing) store directory.
    dataset_name : str
        Name of the dataset, for the message.
    action : {"read", "append", "replace"}
        What was about to happen; picks the advice in the message.

    Raises
    ------
    StoreProtocolError
        The stamp is missing, unreadable, or names another protocol.
    """
    problem = protocol_problem(store)
    if problem is None:
        return
    raise StoreProtocolError(
        f"databank '{dataset_name}' at {store} {problem}. This code reads "
        f"and writes protocol '{DATABANK_PROTOCOL}', and numbers from the "
        "two protocols are not comparable."
        + _ADVICE[action].format(protocol=DATABANK_PROTOCOL)
    )


def run_settings_problem(
    store: Path, run_settings: Mapping[str, Any] | None
) -> str | None:
    """Describe how ``run_settings`` differs from the store's, or ``None``.

    Compared for equality as JSON, without interpretation. Neither side
    having settings is a match.
    """
    incoming = normalize_run_settings(run_settings)
    stamp = _read_stamp(store) or {}
    stored = stamp.get(RUN_SETTINGS_KEY)
    if stored == incoming:
        return None
    if stored is None:
        return (
            "records no run settings, but the caller passed "
            f"{json.dumps(incoming, sort_keys=True)}"
        )
    if incoming is None:
        return (
            f"records the run settings {json.dumps(stored, sort_keys=True)}, "
            "but the caller passed none"
        )
    keys = sorted(
        k
        for k in set(stored) | set(incoming)
        if stored.get(k, None) != incoming.get(k, None)
        or (k in stored) != (k in incoming)
    )
    diffs = "; ".join(
        f"{k}: store {json.dumps(stored.get(k), sort_keys=True)}, caller "
        f"{json.dumps(incoming.get(k), sort_keys=True)}"
        for k in keys
    )
    return f"records other run settings ({diffs})"


def check_run_settings(
    root: Path,
    dataset_name: str,
    run_settings: Mapping[str, Any] | None,
) -> None:
    """Refuse run settings that differ from an existing store's.

    The create step calls this before any cell runs, so a launch with
    other settings stops before it spends compute (ADR 0027); the dump's
    append makes the same check. A store that does not exist yet has
    nothing to compare with and passes. The store's protocol is checked
    first (:func:`check_protocol`), so an old store is refused for its
    protocol, not for its missing settings.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.
    run_settings : Mapping or None
        The caller's run settings (plain JSON), or ``None``.

    Raises
    ------
    StoreProtocolError
        The store's stamp is missing or names another protocol.
    RunSettingsError
        The store's run settings differ from ``run_settings``; neither
        having settings is a match.
    TypeError
        ``run_settings`` is not a JSON mapping.
    """
    normalize_run_settings(run_settings)
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        return
    check_protocol(store, dataset_name, action="append")
    _refuse_other_settings(store, dataset_name, run_settings)


def _refuse_other_settings(
    store: Path,
    dataset_name: str,
    run_settings: Mapping[str, Any] | None,
) -> None:
    problem = run_settings_problem(store, run_settings)
    if problem is None:
        return
    raise RunSettingsError(
        f"databank '{dataset_name}' at {store} {problem}. A store holds "
        "cells made under one set of run settings. Nothing was written. "
        "Use the store's settings, or start a new store under a new "
        "databank_root or dataset name."
    )


def check_f_star(
    root: Path,
    dataset_name: str,
    f_star_by_task: Mapping[str, float],
) -> None:
    """Refuse an f* that differs from the one a store holds for a task.

    A store keeps one f* per task hash and holds that task's cells as
    gaps to it, so an append refuses a task whose f* differs from the
    stored one (ADR 0027). The create step calls this before any cell
    runs, so a corrected minimum stops the launch before it spends
    compute. The comparison is exact in float64, and NaN (no finite f*)
    equals NaN. A store that does not exist yet, and a task the store
    does not hold, pass. The store's protocol is checked first
    (:func:`check_protocol`).

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory.
    dataset_name : str
        Name of the dataset.
    f_star_by_task : Mapping
        The caller's f* per task hash: the task's ``global_min`` as
        float64, NaN for a task without a finite one.

    Raises
    ------
    StoreProtocolError
        The store's stamp is missing or names another protocol.
    RuntimeError
        The store is marked mid-append: a dump crashed while appending,
        and its f* values are not final until the next dump restores it.
    FStarError
        The store holds another f* for one or more of the tasks; the
        message names each with its stored and incoming value.
    """
    store = layout.zarr_store_path(root, dataset_name)
    if not store.exists():
        return
    check_protocol(store, dataset_name, action="append")
    # A crashed append leaves the store half-grown: the tasks it added
    # hold f* NaN until the next dump restores the rollback. Comparing
    # against that would refuse a re-submit the append itself accepts.
    if layout.append_marker_path(root, dataset_name).exists():
        raise RuntimeError(
            f"databank '{dataset_name}' at {store} is mid-append (a dump "
            "crashed while appending); its f* values are not final. Re-run "
            "the dump to restore it first, then check again."
        )
    _refuse_other_f_star(store, dataset_name, f_star_by_task)


def _same_f_star(a: float, b: float) -> bool:
    """Exact float64 equality, with NaN (no finite f*) equal to NaN."""
    return a == b or (bool(np.isnan(a)) and bool(np.isnan(b)))


def _one_f_star_per_task(
    f_stars: Iterable[tuple[str, float]], dataset_name: str
) -> dict[str, float]:
    """The incoming f* per task hash, refusing a hash given two of them.

    A dump's runs can list a task hash more than once; they must agree on
    its f*, or one value would silently win (ADR 0027). Same comparison as
    against the store: exact float64, NaN equals NaN.
    """
    first: dict[str, float] = {}
    differing: dict[str, list[float]] = {}
    for task, f in f_stars:
        f = float(f)
        if task not in first:
            first[task] = f
        elif not _same_f_star(first[task], f) and not any(
            _same_f_star(seen, f) for seen in differing.get(task, [])
        ):
            differing.setdefault(task, [first[task]]).append(f)
    if not differing:
        return first
    detail = ", ".join(
        f"{t}: " + " and ".join(repr(v) for v in values)
        for t, values in sorted(differing.items())
    )
    raise FStarError(
        f"the runs to dump into databank '{dataset_name}' give more than "
        f"one f* (global_min) for task(s) {detail}. A store holds one f* "
        "per task, and its cells of that task are all gaps to it. Nothing "
        "was written (l2co ADR 0027)."
    )


def _refuse_other_f_star(
    store: Path,
    dataset_name: str,
    f_star_by_task: Mapping[str, float],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_zarr(store, consolidated=True)
    stored = dict(
        zip(
            (str(t) for t in ds[TASK_ID].values),
            np.asarray(ds[F_STAR].values, dtype=np.float64).tolist(),
            strict=True,
        )
    )
    differing = {
        t: (stored[t], float(f))
        for t, f in sorted(f_star_by_task.items())
        if t in stored and not _same_f_star(stored[t], float(f))
    }
    if not differing:
        return
    detail = ", ".join(
        f"{t}: stored {was!r}, incoming {now!r}"
        for t, (was, now) in differing.items()
    )
    raise FStarError(
        f"databank '{dataset_name}' at {store} holds another f* "
        f"(global_min) for task(s) {detail}. A store's cells of a task are "
        "all stored as gaps to one f*, and an append cannot change a "
        "store's f* (a tool to update it is future work). Nothing was "
        "written (l2co ADR 0027)."
    )
