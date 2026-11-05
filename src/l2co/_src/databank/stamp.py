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
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from l2co._src.databank import layout

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: The run protocol this code reads and writes. Set by hand; bump it in
#: any change after which newly written numbers are not comparable with
#: stored ones. ``"dataset-paper-2026-11"`` is the dataset paper's
#: protocol revision (2026-11).
DATABANK_PROTOCOL: str = "dataset-paper-2026-11"

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


def _l2co_version() -> str:
    try:
        return version("l2co")
    except PackageNotFoundError:
        return "unknown"


def write_stamp(store: Path) -> None:
    """Stamp a newly laid-out store with :data:`DATABANK_PROTOCOL`.

    Parameters
    ----------
    store : pathlib.Path
        The store directory (``<root>/<dataset_name>.zarr``).
    """
    stamp = {
        "protocol": DATABANK_PROTOCOL,
        "l2co_version": _l2co_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
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
        ``{"protocol", "l2co_version", "created"}``, or ``None`` for a
        store written before stores were stamped.

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
