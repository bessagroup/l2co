"""Task continuation utilities for resuming from saved experiment data."""

import pandas as pd
from f3dasm import ExperimentData
from f3dasm.design import Domain


def retrieve_tasks(experiment_data: ExperimentData) -> pd.DataFrame:
    """
    Retrieve unique tasks from the given experiment data.

    Parameters
    ----------
    experiment_data : ExperimentData
        The experiment data containing tasks.

    Returns
    -------
    ExperimentData
        Experiment data containing unique tasks.
    """
    domain = Domain()
    domain.add_output("task")

    tasks = set(es.input_data["task"] for _, es in experiment_data)

    return ExperimentData(
        domain=domain, output_data=[{"task": task} for task in tasks]
    )
