# import argparse
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from f3dasm import ExperimentData

from l2co import logger as l2co_logger
from l2co._src.optimizerschedule import OptimizationSchedule, OptimizationStep
from l2co._src.tasks.benchmarkfunctions.task import Task


@hydra.main(config_path=".", config_name="config")
def main(config):
    path = Path(config.opt_dir) / config.path

    experimentdata = ExperimentData.from_file(path)
    labels, opt_data, placeholder_path, step_dict = create_entire_dataset(
        experiment_data=experimentdata,
        budget_steps=config.budget_steps,)

    # Store the files
    opt_data.to_netcdf(path / 'entire_dataset.nc')
    labels.to_csv(path / 'labels.csv')
    experimentdata.store()

    with open(path / 'steps.json', 'w') as f:
        json.dump(step_dict, f, indent=4)

    # Remove placeholder
    placeholder_path.unlink()


def create_entire_dataset(
    experiment_data: ExperimentData,
    budget_steps: int,
    loss_column_name: str = 'loss history',
        seed_column_name: str = 'seed',
        optimizer_column_name: str = 'optimizer schedule',
        task_column_name: str = 'task'
) -> Tuple[pd.DataFrame, xr.Dataset, Path, Dict[str, OptimizationStep]]:
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

    df, df_output = experiment_data.to_pandas(keep_references=True)

    df[loss_column_name] = df_output[loss_column_name]

    # preceed every entry of df_output with experimentdata.project_dir
    df = df.map(
        lambda x: experiment_data.project_dir / 'experiment_data' / x,
        na_action='ignore')

    # ============= Create the labels =============

    labels = []
    for task_path in df[task_column_name]:
        task = Task.load(task_path)
        labels.append(task.tag_hashable)

        del task

    df_labels = pd.DataFrame(
        labels, columns=[f"task_{i}" for i in range(len(labels[0]))])

    schedules = []
    step_list = []
    for schedule_path in df[optimizer_column_name]:
        schedule = OptimizationSchedule.load(schedule_path)
        step_list.append(schedule)
        schedules.append(schedule.tag_hashable)
        del schedule

    step_dict = {schedule.name: schedule.to_dict() for schedule in step_list}

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

    # ============= Create the optimization data =============

    labels = create_task_labels_df(pivot_table)
    opt_data, placeholder_path = create_optimization_data(
        pivot_table, budget_steps=budget_steps)

    return labels, opt_data, placeholder_path, step_dict


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
        budget_steps: int) -> Tuple[xr.Dataset, Path]:

    df = pivot_table.copy(deep=True)

    # Hashing
    df_labels = create_task_labels_df(df)

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

    def validation_fn(x: xr.Dataset):
        popsizes = x.attrs['popsizes']

        if isinstance(popsizes, Integral):
            popsizes = [popsizes]

        budgets = x.attrs['budgets']

        if isinstance(budgets, Real):
            budgets = [budgets]

        n_iterations = x.attrs['n_iterations']

        bins = np.concatenate(([0], np.cumsum(
            [n_iterations*pop*budget for
             pop, budget in zip(popsizes, budgets)])))

        bins[-1] += 1

        y_bins = x['output'].groupby_bins(
            "budget", bins=bins, include_lowest=True, right=False)

        y_coarsen = [y_group.coarsen(budget=pop)
                     for (_, y_group), pop in zip(y_bins, popsizes)]

        y_mean: xr.DataArray = xr.concat(
            [_y.mean() for _y in y_coarsen], dim='budget')
        y_std = xr.concat([_y.std() for _y in y_coarsen], dim='budget')
        y_min = xr.concat([_y.min() for _y in y_coarsen], dim='budget')

        x = xr.Dataset(
            {'output_mean': y_mean,
             'output_std': y_std,
             'output_min': y_min},
            coords={'budget': np.linspace(0., 1., n_iterations)})

        x = x.interp(budget=np.linspace(
            0., 1., budget_steps), method='nearest')

        return x

    xarr_full = xr.open_mfdataset(
        df.to_numpy().tolist(),
        combine='nested',
        concat_dim=[xr.DataArray(df.index, coords=[df.index], dims=['itemID']),
                    xr.DataArray(df.columns, coords=[
                        df.columns], dims=['optimizer']),
                    ],
        parallel=True,
        preprocess=validation_fn,
        coords='minimal',
        join='outer',
    )

    xarr_full = xarr_full.stack(
        problem=('realization', 'itemID'), create_index=False)

    # Open the whole thing:
    l2co_logger.info("Opening the whole thing")

    if 'input_dim' in xarr_full.coords:
        return xarr_full.drop('input_dim'), placeholder_path

    else:
        return xarr_full, placeholder_path


if __name__ == '__main__':
    main()
