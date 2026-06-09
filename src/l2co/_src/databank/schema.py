"""Build the empty zarr template for a consolidated databank.

The consolidated store holds one chunk per ``(taskID, schedule)`` cell so
a subset read touches only the selected chunks, and an unwritten cell
(a missing ``(task, optimizer)`` pair) is simply an absent chunk that
reads back as ``NaN`` at zero storage cost.

Two helpers cooperate:

* :func:`big_template` builds the **deferred** float arrays
  (``loss_history`` + ``performance`` variables) with their chunk/shard
  encoding. These are the large, per-pair, region-written arrays.
* the small string / target arrays (``task_labels``,
  ``target_value``, ``optimizationstep``) are written wholesale by the
  writer via ``mode="a"`` and need no template.

All numerical variables are stored as ``float32`` so a missing cell can
be ``NaN`` while keeping disk usage down; float64 is reserved for the
upstream optimization runs that *generate* the data, not for storing the
results. The reader returns them as-is (callers cast to int/bool
downstream if needed).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import dask.array as da
import numpy as np
import xarray as xr

# Local
from l2co.naming import SCHEDULE, TASK_ID

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Storage dtype for every numerical variable (see module docstring):
# float32 on disk to halve storage; float64 is only needed upstream for
# the optimization runs that generate the data, not for storing it.
STORE_DTYPE = np.float32


def canonical_dims(dims: tuple[str, ...]) -> tuple[str, ...]:
    """Reorder ``dims`` so ``taskID`` / ``schedule`` lead, rest in order.

    Region writes index the leading ``(taskID, schedule)`` chunk axes,
    so every variable is stored with those two dims first regardless of
    the order :func:`_build_task_records` happened to produce (e.g.
    ``cost`` arrives as ``(taskID, metric, schedule, ...)``).
    """
    rest = [d for d in dims if d not in (TASK_ID, SCHEDULE)]
    return (TASK_ID, SCHEDULE, *rest)


def big_template(
    sample_loss: xr.Dataset,
    sample_perf: xr.Dataset,
    taskids: list[str],
    schedules: list[str],
) -> tuple[xr.Dataset, dict[str, dict]]:
    """Build the deferred float template for the per-pair arrays.

    Parameters
    ----------
    sample_loss : xr.Dataset
        One pair's **interpolated** loss history (dims include
        ``taskID``/``schedule`` of size 1 plus the interior axes). Its
        variable list, interior dim sizes and interior coords define the
        full-store loss arrays.
    sample_perf : xr.Dataset
        One pair's performance dataset, defining the performance arrays
        the same way.
    taskids : list[str]
        Full, ordered ``taskID`` coordinate of the store.
    schedules : list[str]
        Full, ordered ``schedule`` coordinate of the store.

    Returns
    -------
    template : xr.Dataset
        Dataset of ``dask`` arrays filled with ``NaN`` (one chunk per
        ``(taskID, schedule)``), ready for
        ``to_zarr(compute=False)``.
    encoding : dict[str, dict]
        Per-variable ``{"chunks": ..., "shards": ...}`` encoding (one
        shard per pair, so each cell maps to a single file on disk).
    """
    n_task, n_sched = len(taskids), len(schedules)
    data_vars: dict[str, tuple] = {}
    encoding: dict[str, dict] = {}
    coords: dict[str, object] = {TASK_ID: taskids, SCHEDULE: schedules}

    for sample in (sample_loss, sample_perf):
        for name, var in sample.data_vars.items():
            dims = canonical_dims(var.dims)
            interior = dims[2:]
            interior_shape = tuple(sample.sizes[d] for d in interior)
            shape = (n_task, n_sched, *interior_shape)
            chunks = (1, 1, *interior_shape)
            data_vars[name] = (
                dims,
                da.full(shape, np.nan, dtype=STORE_DTYPE, chunks=chunks),
            )
            encoding[name] = {"chunks": chunks, "shards": chunks}
            # Carry interior coords (realization, output_dim, budget,
            # target, metric) through from whichever sample defines them.
            for d in interior:
                if d not in coords and d in sample.coords:
                    coords[d] = sample.coords[d].values

    return xr.Dataset(data_vars, coords=coords), encoding


def downcast_floats(ds: xr.Dataset) -> xr.Dataset:
    """Cast only the floating data variables to :data:`STORE_DTYPE`.

    Used for the raw ``loss_history`` netCDF sidecar, which keeps its
    original variable-length ``budget`` axis and holds integer counters
    (``fevals`` / ``iterations`` / ``update_step``) alongside the float
    loss variables. A blanket ``.astype(STORE_DTYPE)`` would promote those
    counters to float and break the int-counter contract restored on read
    by :func:`l2co._src.databank.records.loss_preprocess`; this casts the
    float variables only, leaving integer (and any other) dtypes and the
    coordinates untouched.
    """
    return ds.assign(
        {
            name: var.astype(STORE_DTYPE)
            for name, var in ds.data_vars.items()
            if np.issubdtype(var.dtype, np.floating)
        }
    )


def to_region_cell(ds: xr.Dataset) -> xr.Dataset:
    """Prepare one pair's dataset for a ``to_zarr(region=...)`` write.

    Transposes every variable to :func:`canonical_dims` order, casts to
    :data:`STORE_DTYPE`, and drops all coordinates (a region write must
    not carry index coordinates for the dims it targets).
    """
    out = {}
    for name, var in ds.data_vars.items():
        dims = canonical_dims(var.dims)
        out[name] = var.transpose(*dims).astype(STORE_DTYPE)
    cell = xr.Dataset(out)
    return cell.drop_vars(list(cell.coords), errors="ignore")
