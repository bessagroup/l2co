"""
Where a ``Task`` meets an ``OptimizationStep`` (ADR 0018).

``l2co-optimizers`` has no notion of a task: its factories take
``model``, ``loss_fn`` and ``pass_rng`` as keywords, and
:meth:`~l2co_optimizers.RunState.init` takes an already-built
``UpdateClass``. ``l2co-tasks`` has no notion of an optimizer. This
module is the one place the two meet:

* :data:`meta_optimizers` / :func:`register_optimizer` -- the registry
  for meta-optimizers (``"l2co"``, ``"rl2co"``), whose factories need
  the whole task (its budget, tag, ``global_min``), not just the three
  keywords a bare optimizer reads.
* :func:`optimizer_mapping` -- the Task-level lookup. A meta-optimizer
  name resolves to its factory; any other name falls through to the
  bare registry, wrapped so it too is called with ``task=``.
* :func:`init_run_state` -- resolve an ``OptimizationStep`` against a
  task and build the :class:`~l2co_optimizers.RunState` for it.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from functools import cache

# Third-party
from jaxtyping import PRNGKeyArray
from l2co_optimizers import (
    OptimizationStep,
    RunState,
    UpdateClass,
    normalize_key,
)
from l2co_optimizers import optimizer_mapping as bare_optimizer_mapping
from l2co_optimizers import optimizers as bare_optimizers
from l2co_tasks import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Normalized meta-optimizer name -> factory. A factory is called with
#: ``task=``, ``opt_hash=``, ``bounded=``, ``stop_fn=`` and the step's
#: ``**hyperparameters``, and returns an ``UpdateClass``.
meta_optimizers: dict[str, Callable[..., UpdateClass]] = {}


def register_optimizer(name: str, factory: Callable[..., UpdateClass]) -> None:
    """Register a meta-optimizer under ``name``.

    The extension point for optimizers that need a whole task -- l2co's
    own ``"l2co"`` and rl2co's ``"rl2co"`` register here as an import
    side effect. A bare optimizer, which needs only ``model`` /
    ``loss_fn`` / ``pass_rng``, registers with
    ``l2co_optimizers.register_optimizer`` instead.

    Parameters
    ----------
    name : str
        Name to register under, normalized like the bare registry's keys
        (non-alphanumerics stripped, lowercased). Re-registering a
        meta-optimizer overwrites it.
    factory : Callable[..., UpdateClass]
        ``factory(*, task, opt_hash, bounded, stop_fn, **hyperparameters)
        -> UpdateClass``.

    Raises
    ------
    ValueError
        If ``name`` is already a bare optimizer: one name, one
        optimizer.
    """
    key = normalize_key(name)
    if key in bare_optimizers:
        raise ValueError(
            f"{name!r} is already a bare optimizer in l2co_optimizers; a "
            "meta-optimizer may not shadow it."
        )
    meta_optimizers[key] = factory


def optimizer_mapping(name: str) -> Callable[..., UpdateClass]:
    """Resolve an optimizer name to a factory that takes a task.

    Parameters
    ----------
    name : str
        Optimizer name, in any spelling that normalizes to a registered
        key.

    Returns
    -------
    Callable[..., UpdateClass]
        ``factory(*, task, opt_hash, bounded, stop_fn, **hyperparameters)
        -> UpdateClass``. For a meta-optimizer it is the registered
        factory itself; for a bare optimizer it unpacks ``task.model``,
        ``task.loss_fn`` and ``task.pass_rng`` into the bare factory. The
        same object is returned for every spelling of a name.

    Raises
    ------
    ValueError
        If ``name`` is neither a meta-optimizer nor a bare optimizer.
    """
    key = normalize_key(name)
    if key in meta_optimizers:
        return meta_optimizers[key]
    # Raises the bare registry's "not recognized" error for unknown names.
    bare_optimizer_mapping(key)
    return _bare_from_task(key)


@cache
def _bare_from_task(key: str) -> Callable[..., UpdateClass]:
    """The Task-taking wrapper of the bare optimizer ``key``.

    Cached so a name always resolves to the same object. The bare
    factory is looked up per call, not captured, so re-registering a
    bare optimizer takes effect here too.
    """

    def from_task(*, task: Task, **kwargs) -> UpdateClass:
        return bare_optimizer_mapping(key)(
            model=task.model,
            loss_fn=task.loss_fn,
            pass_rng=task.pass_rng,
            **kwargs,
        )

    from_task.__name__ = f"{key}_from_task"
    from_task.__qualname__ = from_task.__name__
    return from_task


def init_run_state(
    optimizer: OptimizationStep,
    task: Task,
    bounded: tuple[float | None, float | None],
    key: PRNGKeyArray,
) -> RunState:
    """Build the :class:`~l2co_optimizers.RunState` for one optimizer on
    one task.

    Resolves ``optimizer`` through :func:`optimizer_mapping` (so
    meta-optimizers work too), then hands the result to
    :meth:`RunState.init` with the task's model and data.

    Parameters
    ----------
    optimizer : OptimizationStep
        Optimizer configuration.
    task : Task
        Task to optimize.
    bounded : tuple[float | None, float | None]
        Box bounds (lower, upper) the optimizer clips to.
    key : PRNGKeyArray
        PRNG key for the batch state and optimizer initialization.

    Returns
    -------
    RunState
        Initialized run state.
    """
    update_class = optimizer_mapping(optimizer.optimizer)(
        **optimizer.hyperparameters,
        task=task,
        opt_hash=optimizer.hash,
        bounded=bounded,
        stop_fn=optimizer.stopping_fn,
    )
    return RunState.init(
        update_class,
        model=task.model,
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=key,
    )
