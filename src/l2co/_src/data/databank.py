"""Write a folder-structured databank from an ``entire_dataset.odata``.

Given the combined ``xr.Dataset`` produced by
``experiments/create_optimization_dataset``, this module splits it along
``SCHEDULE`` and writes one NetCDF per optimizer schedule plus the
metadata needed to reproduce the run.

Output layout::

    <databank_root>/<dataset_name>/
        <schedule_name>.nc          # one NetCDF per SCHEDULE value
        ...
        .data/
            info.json               # run-level info (caller-provided)
            <schedule_name>.json    # OptimizationStep recipe per schedule
            ...

The ``<schedule_name>`` matches the ``SCHEDULE`` coord value in the
source dataset, which the producer sets to
:attr:`l2co.optimization.OptimizationStep.name` (see
``l2co._src.data_validation.create_combined_odata``).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
import logging
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

# Third-party
import xarray as xr

# Local
from l2co._src.data.loader import DataLoader
from l2co._src.optimizer_schedule import OptimizationStep
from l2co.naming import REALIZATION, SCHEDULE, TASK_ID, TASK_LABELS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

# Mapping from ``location`` keyword to the databank root directory on disk.
# Add a new entry here when introducing another machine.
_LOCATION_ROOTS: dict[str, Path] = {
    "local": Path("/Volumes/T7/data/databank"),
    "brown": Path("/oscar/data/mbessa/mvander7/databank"),
}

# Dims and variables a databank-bound dataset must carry. Checked at write
# time so malformed inputs fail fast rather than producing broken slices.
_REQUIRED_DIMS: tuple[str, ...] = (SCHEDULE, TASK_ID, REALIZATION)
_REQUIRED_VARS: tuple[str, ...] = (TASK_LABELS,)

# Characters disallowed inside a schedule name (path separators and any
# control character).
_FORBIDDEN_NAME_CHARS = re.compile(r"[\\/\x00-\x1f]")


def _diff_info(existing: dict, provided: dict) -> str:
    """Summarise how a provided ``info`` dict differs from an existing one.

    Walks both dicts recursively and returns a human-readable, multi-line
    description of every key that was added, removed, or changed. Used to
    make ``mode="append"`` mismatches actionable instead of just reporting
    that the two dicts are unequal.

    Parameters
    ----------
    existing : dict
        The ``info`` already on disk (loaded from ``info.json``).
    provided : dict
        The ``info`` passed to :func:`dump_databank` for this call.

    Returns
    -------
    str
        One ``- <path>: ...`` line per difference, sorted by key path.
    """
    lines: list[str] = []

    def _walk(old: Any, new: Any, path: str) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                child = f"{path}.{key}" if path else str(key)
                if key not in old:
                    lines.append(
                        f"  - {child}: existing=[not present], "
                        f"provided={new[key]!r}"
                    )
                elif key not in new:
                    lines.append(
                        f"  - {child}: existing={old[key]!r}, "
                        "provided=[not provided]"
                    )
                else:
                    _walk(old[key], new[key], child)
        elif old != new:
            lines.append(f"  - {path}: existing={old!r}, provided={new!r}")

    _walk(existing, provided, "")
    return "\n".join(lines) if lines else "  (no field-level differences)"


def _list_schedule_ncs(dataset_root: Path) -> list[Path]:
    """Return ``*.nc`` slices in ``dataset_root``, sorted.

    Skips macOS AppleDouble sidecars (``._*.nc``) that the OS creates
    on non-HFS+ volumes (e.g. exFAT external drives); those files are
    not valid NetCDF and would crash :func:`xarray.open_dataset`.
    """
    return sorted(
        p for p in dataset_root.glob("*.nc") if not p.name.startswith("._")
    )


def _validate_schedule_names(names: list[str]) -> None:
    """Reject schedule names that would be unsafe as filenames."""
    for name in names:
        if not name:
            raise ValueError("Empty schedule name is not allowed.")
        if name in (".", ".."):
            raise ValueError(f"Schedule name '{name}' is reserved.")
        if name.startswith("."):
            raise ValueError(
                f"Schedule name '{name}' starts with '.'; leading "
                "dots are reserved for hidden files."
            )
        if ".." in name:
            raise ValueError(
                f"Schedule name '{name}' contains '..'; path "
                "traversal is not allowed."
            )
        if _FORBIDDEN_NAME_CHARS.search(name):
            raise ValueError(
                f"Schedule name '{name}' contains a path separator "
                "or control character."
            )


def _validate_input_dataset(ds: xr.Dataset) -> None:
    """Assert the input has the dims and variables the databank needs."""
    missing_dims = [d for d in _REQUIRED_DIMS if d not in ds.dims]
    if missing_dims:
        raise ValueError(
            f"Input dataset is missing required dim(s): {missing_dims}. "
            f"Expected all of {list(_REQUIRED_DIMS)}."
        )
    missing_vars = [v for v in _REQUIRED_VARS if v not in ds.variables]
    if missing_vars:
        raise ValueError(
            f"Input dataset is missing required variable(s): "
            f"{missing_vars}. Expected all of {list(_REQUIRED_VARS)}."
        )


def _check_append_compatibility(ds: xr.Dataset, dataset_root: Path) -> None:
    """Compare ``ds`` against any existing slice in ``dataset_root``.

    Picks the first existing ``*.nc`` slice (ignoring macOS
    ``._*.nc`` AppleDouble sidecars) as a sentinel and verifies that
    the incoming dataset has the same non-``SCHEDULE`` dim set, dim
    sizes, coord values, data variables, and ``task_labels``. Raises
    :class:`ValueError` on any mismatch.
    """
    existing_files = _list_schedule_ncs(dataset_root)
    if not existing_files:
        return

    sentinel_path = existing_files[0]
    with xr.open_dataset(sentinel_path, engine="netcdf4") as existing:
        ds_dims = set(ds.dims) - {SCHEDULE}
        existing_dims = set(existing.dims) - {SCHEDULE}
        if ds_dims != existing_dims:
            raise ValueError(
                f"Append schema drift: incoming dims "
                f"{sorted(ds_dims)} differ from existing "
                f"{sorted(existing_dims)} in {sentinel_path.name}."
            )

        for dim in ds_dims:
            if ds.sizes[dim] != existing.sizes[dim]:
                raise ValueError(
                    f"Append schema drift: size of dim '{dim}' is "
                    f"{ds.sizes[dim]} incoming vs "
                    f"{existing.sizes[dim]} in {sentinel_path.name}."
                )
            if (
                dim in ds.coords
                and dim in existing.coords
                and not ds[dim].equals(existing[dim])
            ):
                raise ValueError(
                    f"Append schema drift: coord values for dim "
                    f"'{dim}' differ between incoming and "
                    f"{sentinel_path.name}."
                )

        if set(ds.data_vars) != set(existing.data_vars):
            raise ValueError(
                f"Append schema drift: data variables "
                f"{sorted(ds.data_vars)} differ from existing "
                f"{sorted(existing.data_vars)} in "
                f"{sentinel_path.name}."
            )

        if (
            TASK_LABELS in ds
            and TASK_LABELS in existing
            and not ds[TASK_LABELS].equals(existing[TASK_LABELS])
        ):
            raise ValueError(
                f"Append schema drift: '{TASK_LABELS}' differs "
                "between incoming and existing — the new schedules "
                "appear to come from a different task set."
            )


def _audit_dataset_folder(dataset_root: Path) -> None:
    """Warn about files that don't fit the databank layout.

    macOS AppleDouble sidecars (``._*``) are silently ignored: they
    are filesystem noise from non-HFS+ volumes, not databank content.
    """
    recipe_dir = dataset_root / ".data"
    schedule_names = {p.stem for p in _list_schedule_ncs(dataset_root)}

    allowed_root = {f"{n}.nc" for n in schedule_names} | {".data"}
    for p in dataset_root.iterdir():
        if p.name.startswith("._"):
            continue
        if p.name not in allowed_root:
            logger.warning(
                f"dump_databank: unexpected entry in dataset root: {p}"
            )

    if not recipe_dir.is_dir():
        return
    allowed_data = {f"{n}.json" for n in schedule_names} | {"info.json"}
    for p in recipe_dir.iterdir():
        if p.name.startswith("._"):
            continue
        if p.name not in allowed_data:
            logger.warning(f"dump_databank: unexpected entry in .data/: {p}")


# =============================================================================


def dump_databank(
    odata: xr.Dataset | str | Path,
    location: Literal["local", "brown"],
    dataset_name: str,
    optimizers: list[OptimizationStep],
    info: dict[str, Any],
    *,
    mode: Literal["error", "replace", "append"] = "error",
) -> Path:
    """Split an ``entire_dataset.odata`` into a per-schedule databank.

    For every value along the ``SCHEDULE`` dimension a NetCDF slice is
    written as ``<dataset_name>/<schedule_name>.nc``. The matching
    :class:`OptimizationStep` (keyed by
    :attr:`OptimizationStep.name`) is serialised as JSON under
    ``<dataset_name>/.data/<schedule_name>.json``, and the caller-supplied
    ``info`` dict is dumped to ``<dataset_name>/.data/info.json``.

    Parameters
    ----------
    odata : xarray.Dataset | str | pathlib.Path
        Either the loaded combined dataset or a path to its ``.odata``
        / ``.nc`` file (opened with :func:`xarray.open_dataset`).
    location : {"local", "brown"}
        Which databank to write to; maps to a fixed root directory via
        :data:`_LOCATION_ROOTS`.
    dataset_name : str
        Name of the dataset folder to create (e.g. ``"bbob"``).
    optimizers : list[OptimizationStep]
        Optimization-step recipes to serialise. Must contain exactly one
        entry per ``SCHEDULE`` value in ``odata``; matching is by
        :attr:`OptimizationStep.name`.
    info : dict[str, Any]
        Run-level metadata (e.g. ``n_realizations``, ``seed``,
        ``budget_steps``, ``budgets``, ``samplers``). Dumped verbatim to
        ``info.json`` via :func:`json.dump`; values must be
        JSON-serialisable (checked up front before any writes).
    mode : {"error", "replace", "append"}, optional
        How to handle an existing dataset folder. ``"error"`` (default)
        raises :class:`FileExistsError`. ``"replace"`` removes the
        existing folder and rewrites from scratch. ``"append"`` keeps
        existing files and writes only schedules that are not already
        on disk; any schedule whose ``<name>.nc`` already exists is
        skipped with a warning. In ``"append"`` mode, ``info`` must
        equal the existing ``info.json`` exactly (same run conditions);
        the existing file is then left in place rather than rewritten.

    Returns
    -------
    pathlib.Path
        Absolute path to the written dataset folder.

    Raises
    ------
    FileExistsError
        The dataset folder already exists and ``mode="error"``.
    KeyError
        ``location`` is not a known key of :data:`_LOCATION_ROOTS`.
    ValueError
        Any of: the input dataset is missing a required dim or
        variable; a ``SCHEDULE`` coord value is an unsafe filename
        (empty, ``.`` / ``..``, leading dot, path separator, or
        control character); the set of ``OptimizationStep.name``
        values does not match the ``SCHEDULE`` coord values exactly;
        ``info`` contains values that are not JSON-serialisable;
        ``mode="append"`` and the provided ``info`` differs from the
        existing ``info.json``; ``mode="append"`` and the incoming
        slice's dims, coords, data variables, or ``task_labels``
        differ from an existing on-disk schedule.
    """
    # --- Validate `info` is JSON-serialisable before any FS writes ---
    # Done up front so a bad `info` can't leave the dataset folder
    # half-written after every .nc slice is on disk.
    try:
        info_serialized = json.dumps(info, indent=2)
    except TypeError as e:
        raise ValueError(f"'info' is not JSON-serialisable: {e}") from e

    with ExitStack() as stack:
        # --- Resolve the source dataset; close it if we opened it ---
        if isinstance(odata, (str, Path)):
            ds = stack.enter_context(xr.open_dataset(odata, engine="netcdf4"))
        else:
            ds = odata

        # --- Validate inputs before touching the filesystem ---
        _validate_input_dataset(ds)

        schedule_names = [str(s) for s in ds[SCHEDULE].values]
        _validate_schedule_names(schedule_names)

        optimizers_by_name = {step.name: step for step in optimizers}
        if len(optimizers_by_name) != len(optimizers):
            raise ValueError(
                "Duplicate OptimizationStep.name values in 'optimizers'."
            )
        missing = set(schedule_names) - set(optimizers_by_name)
        extra = set(optimizers_by_name) - set(schedule_names)
        if missing or extra:
            raise ValueError(
                "Mismatch between SCHEDULE coords and provided "
                f"OptimizationStep names. Missing recipes for: "
                f"{sorted(missing)}. Extra recipes not in dataset: "
                f"{sorted(extra)}."
            )

        # --- Resolve and prepare the target directory ---
        dataset_root = _LOCATION_ROOTS[location] / dataset_name
        recipe_dir = dataset_root / ".data"
        logger.info(
            f"dump_databank: writing dataset '{dataset_name}' to "
            f"{dataset_root} (mode={mode})"
        )
        if dataset_root.exists():
            if mode == "error":
                raise FileExistsError(
                    f"Dataset folder {dataset_root} already exists; pass "
                    "mode='replace' to wipe and rewrite, or mode='append' "
                    "to extend it with new schedules."
                )
            elif mode == "replace":
                logger.info(
                    f"dump_databank: removing existing folder {dataset_root}"
                )
                shutil.rmtree(dataset_root)
                recipe_dir.mkdir(parents=True)
            elif mode == "append":
                info_path = recipe_dir / "info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        existing_info = json.load(f)
                    if existing_info != info:
                        raise ValueError(
                            f"Provided 'info' does not match existing "
                            f"{info_path}; append mode requires identical "
                            "run-level info. Differences "
                            "(existing vs provided):\n"
                            f"{_diff_info(existing_info, info)}"
                        )
                _check_append_compatibility(ds, dataset_root)
                recipe_dir.mkdir(parents=True, exist_ok=True)
        else:
            recipe_dir.mkdir(parents=True)

        # --- Write per-schedule NetCDF slices and JSON recipes ---
        n_written = 0
        n_skipped = 0
        for name in schedule_names:
            nc_path = dataset_root / f"{name}.nc"
            json_path = recipe_dir / f"{name}.json"
            if mode == "append" and nc_path.exists():
                logger.warning(
                    f"dump_databank: schedule '{name}' already exists at "
                    f"{nc_path}; skipping (mode='append' never overwrites)."
                )
                n_skipped += 1
                continue
            ds.sel({SCHEDULE: [name]}).to_netcdf(nc_path, engine="netcdf4")
            optimizers_by_name[name].save_to_json(json_path)
            logger.info(f"dump_databank: wrote schedule '{name}' → {nc_path}")
            n_written += 1

        # --- Write run-level info ---
        # Skip rewriting in append mode when info.json already exists:
        # it was verified to match above, so re-truncating would only
        # risk corrupting the canonical record we just validated
        # against. When writing, go via a temp file + atomic rename so
        # a crash mid-write doesn't leave a half-written info.json.
        info_path = recipe_dir / "info.json"
        if info_path.exists():
            logger.info(
                f"dump_databank: kept existing {info_path} (matches input)"
            )
        else:
            tmp_path = info_path.with_suffix(".json.tmp")
            tmp_path.write_text(info_serialized)
            tmp_path.replace(info_path)
            logger.info(f"dump_databank: wrote {info_path}")

        _audit_dataset_folder(dataset_root)

        skipped_str = f", {n_skipped} skipped" if n_skipped else ""
        logger.info(
            f"dump_databank: done — {n_written} schedule(s) written"
            f"{skipped_str} → {dataset_root}"
        )
        return dataset_root.resolve()


def load_databank_dataset(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[str],
    location: Literal["local", "brown"],
) -> xr.Dataset:
    """Open every per-schedule NetCDF and concatenate along ``SCHEDULE``.

    Reads the per-schedule slices written by :func:`dump_databank` and
    returns them concatenated as a single raw :class:`xarray.Dataset`.
    Each slice is fully materialised in memory and its NetCDF file
    handle is closed before this function returns, so the result is
    independent of concurrent writers on disk (e.g. another process
    calling :func:`dump_databank` mid-flight). Trades file-handle
    safety for memory: the entire requested subset is held in RAM.

    Use this when you need the bare ``xr.Dataset`` (e.g. to re-write it
    to a ``.odata`` file); use :func:`load_databank` when you want it
    wrapped in a :class:`DataLoader`.

    Variables and coords that do not carry the ``SCHEDULE`` dim (e.g.
    ``task_labels``, ``target_value``) are expected to be identical
    across slices. The concat is done with ``compat="override"`` so
    conflicting values are silently resolved by taking the **first**
    schedule's copy; this hides producer-side drift across schedules
    rather than failing loudly. Inspect the source files directly if
    you suspect a mismatch.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder inside the databank
        (e.g. ``"bbob"``).
    optimizers : list[OptimizationStep] | list[str]
        Schedules to load, identified either by
        :class:`OptimizationStep` (``step.name`` is used) or by name
        string. Order is preserved along the resulting ``SCHEDULE``
        dim. Must be non-empty and contain no duplicate names.
    location : {"local", "brown"}
        Which databank to read from; maps to a fixed root directory
        via :data:`_LOCATION_ROOTS`.

    Returns
    -------
    xarray.Dataset
        The concatenated per-schedule dataset.

    Raises
    ------
    ValueError
        ``optimizers`` is empty or contains duplicate names, or
        ``location`` is not a known key of :data:`_LOCATION_ROOTS`.
    FileNotFoundError
        The dataset folder does not exist, or a requested schedule's
        ``<name>.nc`` is missing.
    """
    if not optimizers:
        raise ValueError("'optimizers' must contain at least one entry.")

    names = [
        opt.name if isinstance(opt, OptimizationStep) else opt
        for opt in optimizers
    ]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate schedule name(s) in 'optimizers': {duplicates}."
        )

    dataset_root = databank_dataset_root(location, dataset_name)

    slices = []
    for name in names:
        nc_path = dataset_root / f"{name}.nc"
        if not nc_path.is_file():
            raise FileNotFoundError(
                f"Schedule '{name}' not found at {nc_path}"
            )
        slices.append(xr.load_dataset(nc_path, engine="netcdf4"))
        logger.info(f"load_databank_dataset: loaded {nc_path}")

    ds = xr.concat(
        slices,
        dim=SCHEDULE,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    logger.info(
        f"load_databank_dataset: loaded {len(slices)} schedule(s) "
        f"from {dataset_root}"
    )
    return ds


def load_databank(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[str],
    location: Literal["local", "brown"],
) -> DataLoader:
    """Open every per-schedule NetCDF and wrap it in a :class:`DataLoader`.

    Thin wrapper over :func:`load_databank_dataset`: loads the requested
    schedules into a single concatenated :class:`xarray.Dataset` and
    returns ``DataLoader.from_xarray(ds)``. See
    :func:`load_databank_dataset` for the loading semantics (in-memory
    materialisation, ``compat="override"`` concat behaviour) and the
    full parameter / exception contract, which is identical.

    Returns
    -------
    DataLoader
        Loader wrapping the concatenated per-schedule dataset.
    """
    ds = load_databank_dataset(
        dataset_name=dataset_name,
        optimizers=optimizers,
        location=location,
    )
    return DataLoader.from_xarray(ds)


# =============================================================================


def databank_dataset_root(
    location: Literal["local", "brown"],
    dataset_name: str,
) -> Path:
    """
    Return the absolute path to a dataset folder in the databank.

    Parameters
    ----------
    location : {"local", "brown"}
        Which databank to read from; maps to a fixed root directory via
        :data:`_LOCATION_ROOTS`.
    dataset_name : str
        Name of the dataset folder inside the databank (e.g. ``"bbob_test
        "``).

    Returns
    -------
    pathlib.Path
        Absolute path to the dataset folder.

    Raises
    ------
    KeyError
        ``location`` is not a known key of :data:`_LOCATION_ROOTS`.
    FileNotFoundError
        The dataset folder does not exist at the expected path.
    """
    if location not in _LOCATION_ROOTS:
        raise ValueError(
            f"Unknown location {location!r}; expected one of "
            f"{sorted(_LOCATION_ROOTS)}."
        )
    path = _LOCATION_ROOTS[location] / dataset_name
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    return path
