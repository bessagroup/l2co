# import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData

from l2co import logger as l2co_logger
from l2co._src.naming import FEATURE, INPUT_DIM, REALIZATION, SCHEDULE, TASK_ID
from l2co._src.optimizerschedule import OptimizationStep
from l2co._src.task import Task
from l2co._src.validation import validation_fn

OPTIMIZATION_DATA_SUFFIX = '.odata'
OPTIMIZATION_SCHEDULE_SUFFIX = '.osched'


@hydra.main(config_path=".", config_name="config")
def main(config):
    path = Path(config.opt_dir) / str(config.path)

    experimentdata = ExperimentData.from_file(path)
    opt_data, schedule_data, placeholder_path = create_entire_dataset(
        experiment_data=experimentdata,
        budget_steps=config.budget_steps,)

    # Store the files
    opt_data.to_netcdf(
        (path / 'entire_dataset').with_suffix(OPTIMIZATION_DATA_SUFFIX))

    l2co_logger.info(
        f"Entire dataset stored at {(path / 'entire_dataset').with_suffix(OPTIMIZATION_DATA_SUFFIX)}")

    # Remove placeholder
    placeholder_path.unlink()

    if schedule_data is not None:
        schedule_data.to_netcdf(
            (path / 'entire_dataset_schedules').with_suffix(OPTIMIZATION_SCHEDULE_SUFFIX))

        l2co_logger.info((
            f"Entire dataset schedules stored at "
            f"{(path / 'entire_dataset_schedules').with_suffix(OPTIMIZATION_SCHEDULE_SUFFIX)}"))

    l2co_logger.info("End of the program")


def create_entire_dataset(
    experiment_data: ExperimentData,
    budget_steps: int,
    loss_column_name: str = 'loss history',
        seed_column_name: str = 'seed',
        optimizer_column_name: str = 'optimizer schedule',
        task_column_name: str = 'task',
        l2co_schedules_column_name: str = 'l2co schedules',
) -> Tuple[xr.Dataset, xr.Dataset | None, Path]:
    """
    Create an entire dataset from a given experiment data object.

    Parameters
    ----------
    experiment_data : f3dasm.ExperimentData
        The experiment data object.

    Returns
    -------
    xarray.Dataset
        The resulting entire dataset.

    Notes
    -----
    The entire dataset is created by concatenating all the optimization
    trajectories (NetCDF files) of the experiment data object.

    All of the files are checked on:
    - if they are flagged as 'invalid' or 'ERROR'
    - if they contain np.nan values
    """

    # Only select finished experiments and reset the index
    experiment_data = experiment_data.select_with_status(
        'finished').reset_index()

    df_input, df_output = experiment_data.to_pandas(keep_references=True)

    # preceed every entry of df_output with experimentdata.project_dir
    df_input = df_input.map(
        lambda x: experiment_data.project_dir / 'experiment_data' / x,
        na_action='ignore')

    df_output = df_output.map(
        lambda x: experiment_data.project_dir / 'experiment_data' / x,
        na_action='ignore')

    df = pd.concat([
        df_input[task_column_name],
        df_input[optimizer_column_name],
        df_output[loss_column_name]
    ], axis=1)

    # ============= Create the labels =============

    pivot_table, step_dict = create_pivot_table(
        df=df, loss_column_name=loss_column_name,
        optimizer_column_name=optimizer_column_name,
        task_column_name=task_column_name)

    # ============= Create the optimization data =============

    opt_data, placeholder_path = create_optimization_data(
        pivot_table,
        validation_fn=partial(validation_fn, budget_steps=budget_steps))

    df_labels = create_task_labels_df(pivot_table)

    # Task labels
    xarr_task_labels = xr.DataArray(df_labels.values,
                                    coords={TASK_ID: df_labels.index,
                                            FEATURE: df_labels.columns},
                                    dims=[TASK_ID, FEATURE],
                                    )

    opt_data['task_labels'] = xarr_task_labels.sel(
        {TASK_ID: opt_data[TASK_ID]}).astype(str)

    # Schedules
    o = {name: json.dumps(schedule_dict)
         for name, schedule_dict in step_dict.items()}

    xarr_opt_steps = xr.DataArray(
        data=list(o.values()),
        coords={SCHEDULE: list(o.keys())},
        dims=SCHEDULE,
        name='schedules'
    )

    opt_data['opt_steps'] = xarr_opt_steps

    if INPUT_DIM in opt_data.coords:
        opt_data = opt_data.drop(INPUT_DIM)

    if 'l2co schedules' in experiment_data.domain.output_names:
        df_schedules = pd.concat([
            df_input[task_column_name],
            df_input[optimizer_column_name],
            df_output[l2co_schedules_column_name]
        ], axis=1)

        pivot_table_schedules, _ = create_pivot_table(
            df=df_schedules, loss_column_name=l2co_schedules_column_name,
            optimizer_column_name=optimizer_column_name,
            task_column_name=task_column_name)

        schedule_data, placeholder_path = create_optimization_data(
            pivot_table_schedules,
            validation_fn=None
        )

        schedule_data = schedule_data['__xarray_dataarray_variable__'].rename(
            "schedules")

    else:
        schedule_data = None

    return opt_data, schedule_data, placeholder_path


def create_pivot_table(
    df: pd.DataFrame,
        loss_column_name: str = 'loss history',
        optimizer_column_name: str = 'optimizer schedule',
        task_column_name: str = 'task'
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:

    labels = []
    for task_path in df[task_column_name]:
        task = Task.load(task_path)
        labels.append(task.tag_hashable)

        del task

    df_labels = pd.DataFrame(
        labels, columns=[f"task_{i}" for i in range(len(labels[0]))])

    schedules = []
    step_list: List[OptimizationStep] = []
    for schedule_path in df[optimizer_column_name]:
        step = OptimizationStep.load(schedule_path)
        step_list.append(step)
        schedules.append(step.tag_hashable)
        del step

    step_dict: Dict[str, Dict[str, Any]] = {
        s.name: s.to_dict() for s in step_list}

    df_schedules = pd.DataFrame(
        schedules, columns=[f"schedule{i}" for i in range(len(schedules[0]))])

    df_loss = df[loss_column_name].to_frame()

    # # Concatenate and create a MultiIndex
    df_multiindex = pd.concat([df_labels, df_schedules, df_loss], axis=1)

    pivot_table = df_multiindex.pivot_table(
        index=[*df_labels.columns],  # These are the rows
        columns=[*df_schedules.columns],  # This is the column
        values=[*df_loss.columns],  # This is the value you want to aggregate
        aggfunc='first'
    )

    return pivot_table, step_dict


def create_task_labels_df(pivot_table: pd.DataFrame) -> pd.DataFrame:
    df_labels = pivot_table.index.to_frame(index=False)

    df = pd.DataFrame([{v[0]: v[1] for v in row} for i,
                       row in df_labels.iterrows()])

    df = df.reindex(
        sorted(df.columns), axis=1)

    def hash_row_with_columns(row):
        # Include column names
        row_str = "_".join(f"{col}:{row[col]}" for col in row.index)
        # Truncate for readability
        return hashlib.sha256(row_str.encode()).hexdigest()[:16]

    df.index = df.apply(
        hash_row_with_columns, axis=1)

    return df


def create_optimization_data(
        pivot_table: pd.DataFrame,
        validation_fn: Optional[Callable] = None) -> Tuple[xr.Dataset, Path]:

    df = pivot_table.copy(deep=True)
    df_labels = create_task_labels_df(df)

    # Hashing
    df.index = df_labels.index
    df.columns = pd.DataFrame([{v[0]: v[1] for v in col}
                              for col in df.columns.droplevel(0)])['opt_name']

    # Open first file to create placeholder
    placeholder_path = Path.cwd() / 'placeholder.nc'
    placeholder = xr.open_dataset(
        df.dropna(axis=0, how='all').iloc[0].dropna().iloc[0])

    placeholder = xr.full_like(placeholder, np.nan)

    # Store placeholder to disk
    placeholder.to_netcdf(placeholder_path)
    placeholder.close()

    # Fill al nans with the placeholder path
    df: pd.DataFrame = df.fillna(placeholder_path)

    l2co_logger.info("Starting opening the entire dataset")

    xarr_full = xr.open_mfdataset(
        df.to_numpy().tolist(),
        combine='nested',
        concat_dim=[xr.DataArray(df.index, coords=[df.index], dims=[TASK_ID]),
                    xr.DataArray(df.columns, coords=[
                        df.columns], dims=[SCHEDULE]),
                    ],
        parallel=False,
        preprocess=validation_fn,
        coords='minimal',
        join='outer',
    )

    xarr_full = xarr_full.stack(
        problem=(REALIZATION, TASK_ID), create_index=False)

    # Open the whole thing:
    l2co_logger.info("Opening the whole thing")

    return xarr_full, placeholder_path


if __name__ == '__main__':
    main()
