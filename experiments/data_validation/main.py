# Standard
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

# Third-party
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import f3dasm

# Local
from l2co import OptimizationStep, Task, validation_fn
from l2co._src import task
from l2co.naming import (
    FEATURE,
    INPUT_DIM,
    REALIZATION,
    SCHEDULE,
    TASK_ID,
    BUDGET,
    OUTPUT_MIN,
    GLOBAL_MIN,
    TASK_LABELS,
    REALIZATION,
    TASK_ID,
    ITERATIONS,
    OUTPUT_DIM,
    FEVALS,
    TARGET,
    COST,
    SUCCESS,
)
from l2co._src.performance import output_cummin, EPSILON

# =============================================================================

OPTIMIZATION_DATA_SUFFIX = ".odata"

# =============================================================================

logger = logging.getLogger("l2co")

loss_column_name: str = "loss history"
optimizer_column_name: str = "optimizer schedule"
task_column_name: str = "task"

# =============================================================================

# DEFAULT_TARGETS = np.linspace(0.0, 1.0, 51)
DEFAULT_TARGETS = np.logspace(-8, 2, 51)


# def quality_value(x: xr.DataArray, g_min: float) -> xr.DataArray:
#     if np.isnan(g_min):
#         g_min = x.min(skipna=True)
#     p = x - g_min
#     p_positive = p.where(p > 0)
#     eps = p_positive.min(skipna=True) + EPSILON
#     x_log10 = np.log10(p + eps)
#     x_min = x_log10.min(skipna=True)
#     x_max = x_log10.max(skipna=True)
#     qv = (x_log10 - x_min) / (x_max - x_min + EPSILON)
#     return qv


def quality_value(x: xr.DataArray, g_min: float) -> xr.DataArray:
    if np.isnan(g_min):
        g_min = x.min(skipna=True)
    qv = x - g_min

    # scaling
    sf = (
        1 / qv.isel({BUDGET: 0}).quantile(q=0.5, skipna=True)
    ) * DEFAULT_TARGETS[-1]
    return qv * sf


def calculate_ert(data: xr.Dataset) -> xr.Dataset:
    raw_g_min = data.coords.get(GLOBAL_MIN, np.nan)
    raw_g_min = raw_g_min if raw_g_min is not None else np.nan
    g_min = raw_g_min if not np.isnan(raw_g_min) else np.nan

    x = output_cummin(data[OUTPUT_MIN]).ffill(BUDGET)
    qv = quality_value(x, g_min)

    targets = xr.DataArray(
        DEFAULT_TARGETS, dims=TARGET, coords={TARGET: DEFAULT_TARGETS}
    )
    solved = qv <= targets
    success = solved.any(BUDGET)
    first_solved = (
        solved.argmax(BUDGET)
        .where(success, other=solved[BUDGET].argmax())
        .compute()
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

    return xr.Dataset(
        {
            COST: cost_da,
            SUCCESS: success,
        }
    )


# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config):
    path = Path(config.opt_dir) / str(config.path)

    experiment_data = f3dasm.ExperimentData.from_file(path)

    logger.info(
        f"Loaded experiment data from {path}, with {len(experiment_data)} entries."
    )

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
    xarr_df

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

    # xarr_performance = xarr_performance.stack(
    #     problem=(REALIZATION, TASK_ID), create_index=True
    # )

    # ============= TRAJECTORY ============= #

    logger.info("Starting trajectory dataset creation")

    xarr_trajectory = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim=[
            xr.DataArray(task_ids, coords=[task_ids], dims=[TASK_ID]),
        ],
        parallel=False,
        preprocess=partial(validation_fn, budget_steps=config.budget_steps),
        coords="minimal",
        join="outer",
    )

    # xarr_trajectory = xarr_trajectory.stack(
    #     problem=(REALIZATION, TASK_ID), create_index=True
    # )

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

    # entire_dataset = entire_dataset.stack(
    #     problem=(REALIZATION, TASK_ID), create_index=False
    # )

    entire_dataset = entire_dataset.drop_vars(GLOBAL_MIN)

    logging.info("Saving entire dataset to disk")

    entire_dataset.to_netcdf(
        (path / "entire_dataset").with_suffix(OPTIMIZATION_DATA_SUFFIX)
    )

    # ============= CLEAN UP ============= #

    logger.info("Cleaning up temporary files")

    for f in file_list:
        f.unlink()

    logger.info("End of program.")


if __name__ == "__main__":
    main()
