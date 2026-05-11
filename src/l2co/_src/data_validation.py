"""Data validation and quality checking utilities for optimization datasets."""

#                                                                       Modules
# =============================================================================

# Standard
import logging
from functools import partial
from pathlib import Path

import f3dasm

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from l2co_tasks import Task
from tqdm import tqdm

# Local
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.performance import output_cummin
from l2co._src.validation import validation_fn
from l2co.naming import (
    BUDGET,
    COST,
    FEATURE,
    FEVALS,
    GLOBAL_MIN,
    ITERATIONS,
    OUTPUT_MIN,
    SCHEDULE,
    SUCCESS,
    TARGET,
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
optimizer_column_name: str = "optimizer schedule"
task_column_name: str = "task"

# =============================================================================

DEFAULT_TARGETS = np.logspace(-8, 2, 51)


def quality_value(x: xr.DataArray, g_min: float) -> xr.DataArray:
    """Compute scaled quality value relative to global minimum.

    Parameters
    ----------
    x : xr.DataArray
        Cumulative-minimum output values.
    g_min : float
        Global minimum. If NaN, the data minimum is used.

    Returns
    -------
    xr.DataArray
        Scaled quality values where 0 indicates the optimum.
    """
    if np.isnan(g_min):
        g_min = x.min(skipna=True)
    qv = x - g_min

    # # scaling
    # sf = (
    #     1
    #     / qv.isel({BUDGET: 0})
    #     .quantile(q=0.5, skipna=True)
    #     .drop_vars("quantile")
    # ) * DEFAULT_TARGETS[-1]
    return qv  # * sf


def calculate_ert(data: xr.Dataset) -> xr.Dataset:
    """Calculate Expected Running Time (ERT) metrics.

    Parameters
    ----------
    data : xr.Dataset
        Optimization dataset with ``output_min``, ``fevals``,
        and ``iterations`` variables.

    Returns
    -------
    xr.Dataset
        Dataset with ``cost`` and ``success`` variables
        across target quality levels.
    """
    raw_g_min = data.coords.get(GLOBAL_MIN, np.nan)
    raw_g_min = raw_g_min if raw_g_min is not None else np.nan
    g_min = raw_g_min if not np.isnan(raw_g_min) else np.nan

    x = output_cummin(data[OUTPUT_MIN]).ffill(BUDGET)
    qv = quality_value(x, g_min)

    qv = qv.fillna(np.inf)

    targets = xr.DataArray(
        DEFAULT_TARGETS, dims=TARGET, coords={TARGET: DEFAULT_TARGETS}
    )
    solved = qv <= targets
    success = solved.any(BUDGET)

    stopping_budget = (qv.argmin(BUDGET) + 1).clip(max=data.sizes[BUDGET] - 1)

    first_solved = (
        solved.argmax(BUDGET).where(success, other=stopping_budget).compute()
    )

    # Extract the scalar values at the 'first_solved' budget step
    fevals = data[FEVALS].isel({BUDGET: first_solved})
    iterations = data[ITERATIONS].isel({BUDGET: first_solved})

    # Combine metrics into a single DataArray named 'COST'
    # with a new dimension 'metric'
    cost_da = xr.concat(
        [fevals, iterations],
        dim=xr.DataArray([FEVALS, ITERATIONS], dims="metric", name="metric"),
    )

    result = xr.Dataset(
        {
            COST: cost_da,
            SUCCESS: success,
        }
    )
    if GLOBAL_MIN in result.coords:
        result = result.drop_vars(GLOBAL_MIN)
    return result


# =============================================================================


def create_combined_odata(
    experiment_data: f3dasm.ExperimentData, budget_steps: int
):
    """Build a combined optimization dataset from experiment data.

    Loads per-task loss histories, computes ERT performance
    metrics and validation trajectories, merges them with
    task labels, and saves the result to disk.

    Parameters
    ----------
    experiment_data : f3dasm.ExperimentData
        Experiment data with finished optimization runs.
    budget_steps : int
        Number of budget interpolation steps for
        trajectory validation.
    """
    # Only select finished experiments and reset the index
    experiment_data = experiment_data.select_with_status(
        "finished"
    ).reset_index()

    df_input, df_output = experiment_data.to_pandas(keep_references=True)

    # preceed every entry of df_output with experimentdata.project_dir
    df_input = df_input.map(
        lambda row: Path(experiment_data.project_dir)
        / "experiment_data"
        / row.reference,
        na_action="ignore",
    )

    df_output = df_output.map(
        lambda row: Path(experiment_data.project_dir)
        / "experiment_data"
        / row.reference,
        na_action="ignore",
    )

    df = pd.concat(
        [
            df_input[task_column_name],
            df_input[optimizer_column_name],
            df_output[loss_column_name],
        ],
        axis=1,
    )

    temp_df = df.copy()

    temp_df[TASK_ID] = temp_df[task_column_name].apply(lambda x: Task.load(x))
    temp_df[SCHEDULE] = temp_df[optimizer_column_name].apply(
        lambda x: OptimizationStep.load(x)
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
        placeholder_path = Path.cwd() / f"placeholder_{task.hash}.ncs"

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

        b.to_netcdf(f"{task.hash}.ncs")
        b.close()

        placeholder_path.unlink()
        # break

    file_list = [Path.cwd() / Path(t).with_suffix(".ncs") for t in task_ids]

    # ============= PERFORMANCE ============= #

    logger.info("Starting opening the entire dataset for validation")

    xarr_performance = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim=[
            xr.DataArray(task_ids, coords=[task_ids], dims=[TASK_ID]),
        ],
        parallel=False,
        preprocess=calculate_ert,
        coords="minimal",
        join="outer",
    )

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
    )

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

    for f in file_list:
        f.unlink()

    logger.info("End of program.")
