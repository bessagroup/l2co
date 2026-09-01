"""Data validation and quality checking utilities for optimization datasets."""

#                                                                       Modules
# =============================================================================

# Standard
import logging
import shutil
from functools import partial
from pathlib import Path

import f3dasm

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Local
from l2co._src.performance import (
    DEFAULT_EASIEST_QUANTILE,
    calculate_ert,
    compute_task_targets,
    shift_to_quality_value,
)
from l2co._src.validation import validation_fn
from l2co.naming import (
    FEATURE,
    GLOBAL_MIN,
    SCHEDULE,
    TASK_ID,
    TASK_LABELS,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


OPTIMIZATION_DATA_SUFFIX = ".odata"

# =============================================================================

logger = logging.getLogger("l2co")

loss_column_name: str = "loss history"

# =============================================================================


def _ert_from_quantile(
    data: xr.Dataset,
    easiest_quantile: float,
    targets: np.ndarray | None = None,
) -> xr.Dataset:
    """Per-task ERT preprocess that derives targets from randomsearch.

    Resolves ``global_min`` from ``data.coords``, shifts into
    quality-value space, computes targets via
    :func:`compute_task_targets` at the given ``easiest_quantile``,
    and forwards them to :func:`calculate_ert`.

    Parameters
    ----------
    data : xr.Dataset
        Per-task dataset concatenated along ``schedule``.
    easiest_quantile : float
        Forwarded to :func:`compute_task_targets` when ``targets`` is
        ``None``.
    targets : np.ndarray or None, optional
        Pre-computed ERT target quality values (in quality-value space).
        When given, :func:`compute_task_targets` is skipped and these
        targets are used directly -- this scores ``data`` against
        targets fixed elsewhere (e.g. the ``target_value`` already
        stored for an existing task) rather than re-deriving them from
        this dataset's randomsearch baseline.
    """
    global_min = data.coords.get(GLOBAL_MIN, np.nan)
    if global_min is None:
        global_min = np.nan
    if targets is None:
        data_qv = shift_to_quality_value(data, global_min)
        targets = compute_task_targets(
            data_qv, easiest_quantile=easiest_quantile
        )
    return calculate_ert(data, targets=targets, global_min=global_min)


def create_combined_odata(
    experiment_data: f3dasm.ExperimentData,
    budget_steps: int,
    tasks_experiment_data: f3dasm.ExperimentData,
    schedules_experiment_data: f3dasm.ExperimentData,
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
):
    """Build a combined optimization dataset from experiment data.

    Loads per-task loss histories, computes ERT performance
    metrics and validation trajectories, merges them with
    task labels, and saves the result to disk.

    Each row of ``experiment_data`` carries integer ``task_idx`` and
    ``optimizer_idx`` parameters that index into the rows of
    ``tasks_experiment_data`` and ``schedules_experiment_data``.

    Parameters
    ----------
    experiment_data : f3dasm.ExperimentData
        Experiment data with finished optimization runs.
    budget_steps : int
        Number of budget interpolation steps for
        trajectory validation.
    tasks_experiment_data : f3dasm.ExperimentData
        Source ExperimentData whose ``task`` output column is indexed
        by ``experiment_data["task_idx"]``.
    schedules_experiment_data : f3dasm.ExperimentData
        Source ExperimentData whose ``optimizer`` input column is
        indexed by ``experiment_data["optimizer_idx"]``.
    easiest_quantile : float, optional
        Fraction of ``randomsearch`` ``output_min`` values that
        should lie above (be worse than) the per-task easiest ERT
        target. Forwarded to :func:`calculate_ert`. Default ``0.75``.

    Notes
    -----
    Per-task intermediate NetCDF files (``placeholder_*.ncs`` and
    ``<task.hash>.ncs``) are written under
    ``experiment_data.project_dir / "combine_tmp"``. The sub-directory
    is wiped at function entry and removed on success, so a stale
    attempt cannot poison a re-run -- previously these landed in
    ``Path.cwd()`` (the SLURM submit dir) and survived crashes,
    which let HDF5 file-lock state from a failed post job hang the
    next attempt.
    """
    # Only select finished experiments and reset the index
    experiment_data = experiment_data.select_with_status(
        "finished"
    ).reset_index()

    # Per-task scratch dir lives alongside the ExperimentData on
    # disk (not in Path.cwd()), so two pipeline runs sharing a
    # submit directory cannot collide on `<task.hash>.ncs`
    # filenames, and a crashed previous attempt cannot leave
    # leftover files with stale HDF5 lock state for this run to
    # block on. Wipe-then-recreate gives a guaranteed clean start.
    tmp_dir = Path(experiment_data.project_dir) / "combine_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    df_input, df_output = experiment_data.to_pandas(keep_references=True)

    df_output = df_output.map(
        lambda row: (
            Path(experiment_data.project_dir)
            / "experiment_data"
            / row.reference
        ),
        na_action="ignore",
    )

    tasks = [es.output_data["task"] for _, es in tasks_experiment_data]
    schedules = [
        es.input_data["optimizer"] for _, es in schedules_experiment_data
    ]

    temp_df = pd.DataFrame(
        {
            TASK_ID: df_input["task_idx"].apply(lambda i: tasks[int(i)]),
            SCHEDULE: df_input["optimizer_idx"].apply(
                lambda i: schedules[int(i)]
            ),
            loss_column_name: df_output[loss_column_name],
        }
    )

    pivot_table = temp_df.pivot(
        index=TASK_ID,
        columns=SCHEDULE,
        values=loss_column_name,
    )

    # Task labels

    df_tasks = pivot_table.index.to_series()
    task_ids = df_tasks.apply(lambda r: r.hash)

    # Opt schedules

    df_schedules = pivot_table.columns.to_series()
    df_schedules.index = df_schedules.apply(lambda r: r.hash)
    df_schedules = df_schedules.apply(lambda r: r.name)

    # Open entire dataset
    xarr_df = pivot_table.copy()
    xarr_df.columns = df_schedules

    # Process per task
    for task, loss_history in tqdm(xarr_df.iterrows(), total=len(xarr_df)):
        placeholder_path = tmp_dir / f"placeholder_{task.hash}.ncs"

        placeholder = xr.open_dataset(
            loss_history.dropna(axis=0, how="all").iloc[0]
        )

        placeholder = xr.full_like(placeholder, fill_value=np.nan)
        placeholder.to_netcdf(placeholder_path)
        placeholder.close()

        loss_history = loss_history.fillna(placeholder_path)

        b = xr.open_mfdataset(
            loss_history.to_numpy().tolist(),
            combine="nested",
            concat_dim=[
                xr.DataArray(
                    xarr_df.columns, coords=[xarr_df.columns], dims=[SCHEDULE]
                ),
            ],
            parallel=False,
            preprocess=None,
            coords="minimal",
            join="outer",
        )

        # add global min
        g_min_val = task.global_min if task.global_min is not None else np.nan
        b = b.assign_coords({GLOBAL_MIN: g_min_val})

        b.to_netcdf(tmp_dir / f"{task.hash}.ncs")
        b.close()

        placeholder_path.unlink()
        # break

    file_list = [tmp_dir / Path(t).with_suffix(".ncs") for t in task_ids]

    # ============= PERFORMANCE ============= #

    logger.info("Starting opening the entire dataset for validation")

    # Materialise performance eagerly and close the file handles
    # before the trajectory pipeline opens the *same* per-task .ncs
    # files. Holding two open generations of one NetCDF file at once
    # is what lets HDF5 deadlock on its internal mutex during the
    # final to_netcdf compute.
    xarr_performance = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim=[
            xr.DataArray(task_ids, coords=[task_ids], dims=[TASK_ID]),
        ],
        parallel=False,
        preprocess=partial(
            _ert_from_quantile, easiest_quantile=easiest_quantile
        ),
        coords="minimal",
        join="outer",
    ).load()
    xarr_performance.close()

    # ============= TRAJECTORY ============= #

    logger.info("Starting trajectory dataset creation")

    xarr_trajectory = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim=[
            xr.DataArray(task_ids, coords=[task_ids], dims=[TASK_ID]),
        ],
        parallel=False,
        preprocess=partial(validation_fn, budget_steps=budget_steps),
        coords="minimal",
        join="outer",
        compat="override",
    ).load()
    xarr_trajectory.close()

    # Create task labels
    labels_data = pd.DataFrame(
        [
            {**task.tag, GLOBAL_MIN: task.global_min}
            for task in pivot_table.index
        ],
    )
    xarr_task_labels = xr.DataArray(
        labels_data.values,
        coords={TASK_ID: task_ids, FEATURE: labels_data.columns},
        dims=[TASK_ID, FEATURE],
    )

    xarr_trajectory[TASK_LABELS] = xarr_task_labels.sel(
        {TASK_ID: xarr_trajectory[TASK_ID]}
    ).astype(str)

    # ============= MERGE AND SAVE ============= #

    logger.info("Merging trajectory and performance datasets")

    entire_dataset = xr.merge(
        [xarr_trajectory, xarr_performance], compat="override"
    )

    entire_dataset = entire_dataset.drop_vars(GLOBAL_MIN)

    logger.info("Saving entire dataset to disk")

    entire_dataset.to_netcdf(
        (Path(experiment_data.project_dir) / "entire_dataset").with_suffix(
            OPTIMIZATION_DATA_SUFFIX
        )
    )

    # ============= CLEAN UP ============= #

    logger.info("Cleaning up temporary files")

    shutil.rmtree(tmp_dir)

    logger.info("End of program.")
