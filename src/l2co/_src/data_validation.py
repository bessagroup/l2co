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
from tqdm import tqdm

# Local
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

# =============================================================================

HARDEST_TARGET_QV = 1e-8
DEFAULT_EASIEST_TARGET_QV = 1e2
N_TARGETS = 51
DEFAULT_EASIEST_QUANTILE = 0.75


def _fallback_targets(n_targets: int) -> np.ndarray:
    return np.logspace(
        np.log10(HARDEST_TARGET_QV),
        np.log10(DEFAULT_EASIEST_TARGET_QV),
        n_targets,
    )


def shift_to_quality_value(data: xr.Dataset, g_min: float) -> xr.Dataset:
    """Shift ``output_min`` into quality-value space (``g_min == 0``).

    Subtracts the task's global minimum from ``output_min`` so the
    resulting dataset lives in quality-value space, where the optimum
    is 0 and the hardest target ``1e-8`` is meaningful regardless of
    the original scale of the objective. If ``g_min`` is NaN, the
    minimum of ``output_min`` over the data is used as a fallback.

    Parameters
    ----------
    data : xr.Dataset
        Per-task optimization dataset.
    g_min : float
        Task's global minimum (NaN allowed; falls back to data min).

    Returns
    -------
    xr.Dataset
        Copy of ``data`` with ``output_min`` shifted by ``-g_min``.
    """
    if np.isnan(g_min):
        g_min = float(data[OUTPUT_MIN].min(skipna=True))
    return data.assign({OUTPUT_MIN: data[OUTPUT_MIN] - g_min})


def compute_task_targets(
    data_qv: xr.Dataset,
    n_targets: int = N_TARGETS,
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
) -> np.ndarray:
    """Build per-task log-spaced ERT targets in quality-value space.

    Expects ``data_qv`` already shifted so that ``global_min == 0``
    (see :func:`shift_to_quality_value`). The hardest target is
    ``1e-8`` and the easiest is set so that ``easiest_quantile`` of
    the ``random_search`` baseline's ``output_min`` values lie above
    (i.e. are worse than) it — equivalently, the
    ``(1 - easiest_quantile)``-quantile of those values. Random_search
    values are flattened across all remaining dims (typically
    ``REALIZATION`` x ``BUDGET`` x any ``random_search`` schedules).
    Intermediate targets are log-spaced between these endpoints.

    Falls back to ``np.logspace(-8, 2, n_targets)`` if any of the
    following holds:

    * no ``random_search`` schedule is present for this task,
    * the easiest quality value is non-finite or not strictly above
      the hardest target.

    Parameters
    ----------
    data_qv : xr.Dataset
        Per-task dataset with ``output_min`` already in quality-value
        space and a ``schedule`` coordinate whose values are
        ``OptimizationStep.name`` strings.
    n_targets : int, optional
        Number of target levels to produce, by default ``51``.
    easiest_quantile : float, optional
        Fraction of ``random_search`` ``output_min`` values that
        should lie above (be worse than) the easiest target.
        Must be in ``(0, 1)``. Default ``0.75``.

    Returns
    -------
    np.ndarray
        1-D array of ``n_targets`` log-spaced target quality values,
        sorted from hardest (``1e-8``) to easiest.
    """
    if not 0.0 < easiest_quantile < 1.0:
        raise ValueError(
            f"easiest_quantile must be in (0, 1), got {easiest_quantile}"
        )

    if SCHEDULE not in data_qv.coords or OUTPUT_MIN not in data_qv.data_vars:
        return _fallback_targets(n_targets)

    schedule_names = np.asarray(data_qv[SCHEDULE].values, dtype=str)
    mask = np.array(
        [name.startswith("random_search") for name in schedule_names]
    )
    if not mask.any():
        return _fallback_targets(n_targets)

    rs = data_qv[OUTPUT_MIN].isel({SCHEDULE: np.where(mask)[0]})
    easiest_qv = float(rs.quantile(1.0 - easiest_quantile, skipna=True))

    if not np.isfinite(easiest_qv) or easiest_qv <= HARDEST_TARGET_QV:
        return _fallback_targets(n_targets)

    return np.logspace(
        np.log10(HARDEST_TARGET_QV), np.log10(easiest_qv), n_targets
    )


def calculate_ert(
    data: xr.Dataset,
    easiest_quantile: float = DEFAULT_EASIEST_QUANTILE,
) -> xr.Dataset:
    """Calculate Expected Running Time (ERT) metrics.

    Shifts ``output_min`` into quality-value space (subtracting
    ``global_min``, falling back to the data minimum when it is NaN)
    so the optimum is 0, then derives targets per task via
    :func:`compute_task_targets`: the hardest target is ``1e-8``
    (i.e. within ``1e-8`` of the global minimum) and the easiest is
    set so that ``easiest_quantile`` of the ``random_search``
    baseline's shifted ``output_min`` values lie above (i.e. are
    worse than) it, with the remaining levels log-spaced between
    them. The actual target quality values are returned as the
    ``target_value`` data variable so they survive concatenation
    along ``TASK_ID``; the ``target`` dim itself is an integer
    index ``0..n_targets-1``.

    Parameters
    ----------
    data : xr.Dataset
        Optimization dataset with ``output_min``, ``fevals``,
        and ``iterations`` variables and a ``schedule`` coordinate.
    easiest_quantile : float, optional
        Fraction of ``random_search`` values that should lie above
        the easiest target. Default ``0.75``.

    Returns
    -------
    xr.Dataset
        Dataset with ``cost``, ``success``, and ``target_value``
        variables across the per-task target quality levels.
    """
    raw_g_min = data.coords.get(GLOBAL_MIN, np.nan)
    raw_g_min = raw_g_min if raw_g_min is not None else np.nan
    g_min = raw_g_min if not np.isnan(raw_g_min) else np.nan

    data_qv = shift_to_quality_value(data, g_min)

    targets_np = compute_task_targets(
        data_qv, easiest_quantile=easiest_quantile
    )
    target_idx = np.arange(len(targets_np))

    qv = output_cummin(data_qv[OUTPUT_MIN]).ffill(BUDGET).fillna(np.inf)

    targets = xr.DataArray(
        targets_np, dims=TARGET, coords={TARGET: target_idx}
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
            "target_value": targets,
        }
    )
    if GLOBAL_MIN in result.coords:
        result = result.drop_vars(GLOBAL_MIN)
    return result


# =============================================================================


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
        Fraction of ``random_search`` ``output_min`` values that
        should lie above (be worse than) the per-task easiest ERT
        target. Forwarded to :func:`calculate_ert`. Default ``0.75``.
    """
    # Only select finished experiments and reset the index
    experiment_data = experiment_data.select_with_status(
        "finished"
    ).reset_index()

    df_input, df_output = experiment_data.to_pandas(keep_references=True)

    df_output = df_output.map(
        lambda row: Path(experiment_data.project_dir)
        / "experiment_data"
        / row.reference,
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
        preprocess=partial(calculate_ert, easiest_quantile=easiest_quantile),
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
