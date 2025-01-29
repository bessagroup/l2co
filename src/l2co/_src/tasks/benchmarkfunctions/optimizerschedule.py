from itertools import product
from typing import Any, Dict, Iterable, List, NamedTuple

# =============================================================================


class OptimizerScheduler(NamedTuple):
    optimizer: str
    n_evals: int
    x0_strategy: str
    hyperparameters: Dict[str, Any] = {}


def get_scheduler(
        optimizer_1: Dict[str, Any], optimizer_2: Dict[str, Any],
        switch: float, total_evals: int,
        first_x0: str = 'new') -> List[OptimizerScheduler]:
    """
    Get the scheduler for two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : _type_
        name of the second optimizer
    iterations : int
        total number of iterations across both optimizers
    switch : float, optional
        fraction of the iterations to use for the first optimizer
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'

    Returns
    -------
    List[Tuple[str, int, str]]
        List of schedulers for the two optimizers
    """

    i_0 = int(total_evals // (1/switch))
    i_1 = total_evals - i_0

    if optimizer_1['optimizer'] == optimizer_2['optimizer']:
        return [OptimizerScheduler(
            n_evals=total_evals,
            x0_strategy=first_x0,
            **optimizer_1
        ),
        ]

    return [OptimizerScheduler(
        n_evals=i_0,
        x0_strategy=first_x0,
        **optimizer_1
    ),
        OptimizerScheduler(
        n_evals=i_1,
        x0_strategy=get_handshake(
            optimizer_1['optimizer'], optimizer_2['optimizer']),
        **optimizer_2
    )
    ]


def get_handshake(optimizer_1: str, optimizer_2: str) -> str:
    """
    Return the handshake method to use between two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : str
        name of the optimizer to switch from

    Returns
    -------
    str
        method to determine the starting population of the second optimizer

    Notes
    -----
    The handshake method is either 'best', 'last', 'new' or 'random'
    """
    if optimizer_1.lower() == optimizer_2.lower():
        return 'last'

    if optimizer_2.lower() in ['lbfgsb', 'lbfgs', 'adam', 'randomsearch']:
        return 'best'

    if optimizer_1.lower() == 'randomsearch':
        return 'best'

    if optimizer_1.lower() == 'cmaes' and optimizer_2.lower() == 'pso':
        return 'last'

    if optimizer_1.lower() == 'pso' and optimizer_2.lower() == 'cmaes':
        return 'last'

    return 'new'


def get_all_combinations(
        optimizers: List[Dict[str, Any]], total_evals: int,
        first_x0: str = 'new',
        switch: float = 0.5) -> List[List[OptimizerScheduler]]:
    """
    Get all combination of two optimizer names

    Parameters
    ----------
    optimizers : List[str]
        list of optimizer names
    iterations : int
        total number of iterations across both optimizers
    first_x0 : str, optional
        x0 selection strategy for the first optimizer, by default 'new'
    switch : float, optional
        fraction of the iterations to use for the first
        optimizer, by default 0.5

    Returns
    -------
    List[OptimizerScheduler]
        List of optimizer schedulers
    """
    return [get_scheduler(
        optimizer_1=opt_1, optimizer_2=opt_2,
        total_evals=total_evals, first_x0=first_x0,
        switch=switch)
        for opt_1, opt_2 in product(optimizers, repeat=2)]


def name_from_schedule(schedules: Iterable[OptimizerScheduler]) -> str:
    return '+'.join([s.optimizer for s in schedules])
