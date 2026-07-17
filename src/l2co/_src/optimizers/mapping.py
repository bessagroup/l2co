"""Mapping from optimizer names to their UpdateClass implementations."""

from collections.abc import Callable

from .evosax_implementations import evosax_mapping
from .lbfgs_per_eval_key import lbfgs_per_eval_key_mapping
from .optax_implementations import optax_mapping
from .random_search import random_search_mapping
from .shade import shade_mapping
from .update_class import UpdateClass

# Registry of optimizer-name -> factory callables. Each value is a callable
# that, when called with the keyword contract used by ``RunState.init``
# (``**hyperparameters, task, opt_hash, bounded, stop_fn``), returns an
# ``UpdateClass`` instance. Built-in entries come from the optax/evosax/
# random-search/shade modules, plus the ``"l2co"`` meta-optimizer (the
# two-stage L2CO strategy as a strategy wrapper — see ``docs/adr/0002``);
# sibling packages inject their own via ``register_optimizer`` (see below)
# so l2co stays standalone.
optimizers: dict[str, Callable[..., UpdateClass]] = (
    evosax_mapping
    | optax_mapping
    | random_search_mapping
    | shade_mapping
    | lbfgs_per_eval_key_mapping
    | {"l2co": UpdateClass.l2co_update}
)


def register_optimizer(name: str, factory: Callable[..., UpdateClass]) -> None:
    """Register an external optimizer factory under ``name``.

    This is the extension point that lets sibling packages (e.g. ``rl2co``)
    plug their own ``UpdateClass`` into l2co's optimizer registry *without*
    l2co taking a dependency on them. The registering package calls this at
    import time; ``RunState.init`` then resolves the name through
    ``optimizer_mapping`` like any built-in optimizer.

    Parameters
    ----------
    name : str
        Optimizer name to register under. Stored lowercased to match the
        lookup performed by :func:`optimizer_mapping`. Re-registering an
        existing name overwrites the previous factory.
    factory : Callable[..., UpdateClass]
        Factory callable that returns an ``UpdateClass``. It must accept the
        same keyword contract that ``RunState.init`` calls it with:
        ``**hyperparameters, task, opt_hash, bounded, stop_fn``.
    """
    optimizers[name.lower()] = factory


def optimizer_mapping(name: str) -> Callable[..., UpdateClass]:
    """
    Maps optimizer name to corresponding UpdateClass or function.

    Parameters
    ----------
    name : str
        Name of the optimizer.

    Returns
    -------
    Callable[..., UpdateClass]
        Factory callable that constructs the corresponding ``UpdateClass``
        when called with the ``RunState.init`` keyword contract.

    Raises
    ------
    ValueError
        If optimizer name is not recognized.
    """
    try:
        return optimizers[name.lower()]
    except KeyError:
        raise ValueError(f"Optimizer '{name}' not recognized.") from None
