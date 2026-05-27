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
        JSON-serialisable.
    mode : {"error", "replace", "append"}, optional
        How to handle an existing dataset folder. ``"error"`` (default)
        raises :class:`FileExistsError`. ``"replace"`` removes the
        existing folder and rewrites from scratch. ``"append"`` keeps
        existing files and writes only schedules that are not already
        on disk; any schedule whose ``<name>.nc`` already exists is
        skipped with a warning. In ``"append"`` mode, ``info`` must
        equal the existing ``info.json`` exactly (same run conditions).

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
        ``mode="append"`` and the provided ``info`` differs from the
        existing ``info.json``; ``mode="append"`` and the incoming
        slice's dims, coords, data variables, or ``task_labels``
        differ from an existing on-disk schedule.
    """
    # --- Resolve the source dataset ---
    if isinstance(odata, (str, Path)):
        ds = xr.open_dataset(odata, engine="netcdf4")
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
                        "run-level info."
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
    info_path = recipe_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"dump_databank: wrote {info_path}")

    _audit_dataset_folder(dataset_root)

    skipped_str = f", {n_skipped} skipped" if n_skipped else ""
    logger.info(
        f"dump_databank: done — {n_written} schedule(s) written"
        f"{skipped_str} → {dataset_root}"
    )
    return dataset_root.resolve()


def load_databank(
    dataset_name: str,
    optimizers: list[OptimizationStep] | list[str],
    location: Literal["local", "brown"],
) -> DataLoader:
    """Load selected schedules from a databank into a :class:`DataLoader`.

    Opens each ``<dataset_name>/<schedule_name>.nc`` slice produced by
    :func:`dump_databank`, concatenates them along the ``SCHEDULE``
    dimension, and wraps the result via :meth:`DataLoader.from_xarray`.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder inside the databank (e.g.
        ``"bbob_test"``).
    optimizers : list[OptimizationStep] | list[str]
        Schedules to load. Each entry is either an
        :class:`OptimizationStep` (its :attr:`~OptimizationStep.name`
        is used as the filename stem) or the schedule name as a
        string. The two forms can be mixed.
    location : {"local", "brown"}
        Which databank to read from; maps to a fixed root directory
        via :data:`_LOCATION_ROOTS`.

    Returns
    -------
    DataLoader
        DataLoader with all requested schedules stacked along
        ``SCHEDULE``.

    Raises
    ------
    KeyError
        ``location`` is not a known key of :data:`_LOCATION_ROOTS`.
    FileNotFoundError
        The dataset folder or any requested ``<name>.nc`` slice is
        missing.
    ValueError
        ``optimizers`` is empty.
    """
    if not optimizers:
        raise ValueError("'optimizers' must contain at least one entry.")

    dataset_root = _LOCATION_ROOTS[location] / dataset_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    names = [
        opt.name if isinstance(opt, OptimizationStep) else opt
        for opt in optimizers
    ]

    slices = []
    for name in names:
        nc_path = dataset_root / f"{name}.nc"
        if not nc_path.is_file():
            raise FileNotFoundError(
                f"Schedule '{name}' not found at {nc_path}"
            )
        slices.append(xr.open_dataset(nc_path, engine="netcdf4"))
        logger.info(f"load_databank: opened {nc_path}")

    ds = xr.concat(slices, dim=SCHEDULE, data_vars="minimal")
    logger.info(
        f"load_databank: loaded {len(slices)} schedule(s) from {dataset_root}"
    )
    return DataLoader.from_xarray(ds)
