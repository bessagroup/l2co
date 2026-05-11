"""Mapping from optimizer names to their UpdateClass implementations."""

from l2co._src.update_class import UpdateClass

from .evosax_implementations import evosax_mapping
from .optax_implementations import optax_mapping
from .random_search import random_search_mapping

optimizers = evosax_mapping | optax_mapping | random_search_mapping


def optimizer_mapping(name: str) -> UpdateClass:
    """
    Maps optimizer name to corresponding UpdateClass or function.

    Parameters
    ----------
    name : str
        Name of the optimizer.

    Returns
    -------
    UpdateClass
        Corresponding UpdateClass or function for the optimizer.

    Raises
    ------
    ValueError
        If optimizer name is not recognized.
    """
    try:
        return optimizers[name.lower()]
    except KeyError:
        raise ValueError(f"Optimizer '{name}' not recognized.") from None
