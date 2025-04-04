
# Standard
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Type

# Third-party
import jax.numpy as jnp
import optax
from evosax import CMA_ES, PSO, Sep_CMA_ES, Strategy
from f3dasm_optimize import (EvoSaxUpdate, UpdateStep, evosax_scan, lbfgs_scan,
                             optax_scan)

# Local
from .task import Task

# =============================================================================


def choose_es(task: Task, popsize: Optional[int] = None, **kwargs) -> Strategy:
    """
    Choose an evolutionary strategy based on task dimensionality.

    Parameters
    ----------
    task : Task
        Task to optimize.
    popsize : Optional[int], optional
        Population size for optimization, by default None.
    **kwargs : dict
        Additional arguments for the strategy.

    Returns
    -------
    Strategy
        Selected evolutionary strategy.
    """
    d = task.dimensionality
    if d > 400:
        optimizer = partial(
            EvoSaxUpdate,
            optimizer=Sep_CMA_ES, update_fn=evosax_scan
        )
    else:
        optimizer = partial(
            EvoSaxUpdate,
            optimizer=CMA_ES, update_fn=evosax_scan
        )

    popsize = int(4+3*jnp.log(d))

    return optimizer(**kwargs, popsize=popsize)


# =============================================================================

OPT_MAPPING: Dict[str, Type[UpdateStep]] = {
    'adam': partial(
        UpdateStep,
        optimizer=optax.adam, update_fn=optax_scan,
        popsize=1),
    'cmaes': choose_es,
    'sepcmaes': partial(
        EvoSaxUpdate,
        optimizer=Sep_CMA_ES, update_fn=evosax_scan,
    ),
    'pso': partial(
        EvoSaxUpdate,
        optimizer=PSO, update_fn=evosax_scan,
    ),
    'lbfgsb': partial(
        UpdateStep,
        optimizer=optax.lbfgs, update_fn=lbfgs_scan,
        popsize=1),
}
