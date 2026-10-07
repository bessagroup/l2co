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
* :func:`task_bounds` / :func:`refuse_constraints` -- the task is the
  only source of bounds (ADR 0020). ``task_bounds`` turns a task's box
  into the ``(lo, hi)`` pair optimizers clip to; a path with no way to
  enforce constraints calls ``refuse_constraints`` instead.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

# Third-party
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree
from l2co_optimizers import (
    OptimizationStep,
    RunState,
    UpdateClass,
    normalize_key,
)
from l2co_optimizers import optimizer_mapping as bare_optimizer_mapping
from l2co_optimizers import optimizers as bare_optimizers
from l2co_tasks import Box, Task

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


def task_bounds(task: Task) -> tuple[float | None, float | None]:
    """The box a task declares, as the ``(lo, hi)`` pair optimizers clip to.

    The only way bounds leave a task (ADR 0020). Optimizers take one
    pair for every parameter, and "best so far" ignores feasibility, so
    only a box with the same bounds on every parameter can be enforced.

    Parameters
    ----------
    task : Task
        The task whose constraints to read.

    Returns
    -------
    tuple[float | None, float | None]
        ``(None, None)`` for a task without constraints. For a task with
        a :class:`~l2co_tasks.Box`, its lower and upper bound, with an
        infinite side as ``None``.

    Raises
    ------
    ValueError
        If the task has an :class:`~l2co_tasks.Inequality` or
        :class:`~l2co_tasks.Equality`, or a ``Box`` whose bounds differ
        between parameters.
    """
    unsupported = [c for c in task.constraints if not isinstance(c, Box)]
    if unsupported:
        names = ", ".join(f"{c.kind} {c.name!r}" for c in unsupported)
        raise ValueError(
            f"task {task.name!r} has constraints nothing in l2co can "
            f"enforce yet ({names}); only a Box is supported (l2co ADR "
            "0020)"
        )
    if not task.constraints:
        return (None, None)
    (box,) = task.constraints
    return (
        _uniform_bound(task, box.lower, "lower"),
        _uniform_bound(task, box.upper, "upper"),
    )


def _uniform_bound(task: Task, bound: PyTree, side: str) -> float | None:
    """One side of a cast box as a single number, ``None`` if infinite."""
    values = np.concatenate(
        [np.ravel(np.asarray(leaf)) for leaf in jtu.tree_leaves(bound)]
        or [np.zeros(0)]
    )
    if values.size == 0:
        return None
    if not (values == values[0]).all():
        raise ValueError(
            f"task {task.name!r} has a Box whose {side} bound differs "
            "between parameters; optimizers take one (lo, hi) pair for "
            "all parameters (l2co ADR 0020)"
        )
    value = float(values[0])
    return None if math.isinf(value) else value


def refuse_constraints(task: Task, where: str) -> None:
    """Raise if ``task`` has constraints, for a path that can't enforce
    any (ADR 0020).

    Parameters
    ----------
    task : Task
        The task about to be run.
    where : str
        Name of the calling path, for the error message.

    Raises
    ------
    ValueError
        If ``task`` has any constraint, a ``Box`` included.
    """
    if task.constraints:
        kinds = ", ".join(sorted({c.kind for c in task.constraints}))
        raise ValueError(
            f"{where} has no way to enforce a task's constraints, and "
            f"task {task.name!r} has some ({kinds}); see l2co ADR 0020"
        )


def _bounds_from_task(
    task: Task,
    bounded: tuple[float | None, float | None] | None,
    where: str,
) -> tuple[float | None, float | None]:
    """The bounds to clip ``task`` to, refusing caller-supplied ones.

    Transitional (ADR 0020): entry points still accept a ``bounded``
    argument so existing callers keep working, but only as "no bounds".
    Real bounds must be declared as a :class:`~l2co_tasks.Box` on the
    task.
    """
    if bounded is not None and tuple(bounded) != (None, None):
        raise ValueError(
            f"{where}: bounded={tuple(bounded)!r} -- bounds now come from "
            "the task (l2co ADR 0020). Declare them as an l2co_tasks.Box "
            "in the task's constraints instead."
        )
    return task_bounds(task)


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
    bounded : tuple[float | None, float | None] or None
        Transitional (ADR 0020): must be ``(None, None)`` or ``None``.
        The optimizer clips to the task's own box (see
        :func:`task_bounds`). To be removed once no caller passes it.
    key : PRNGKeyArray
        PRNG key for the batch state and optimizer initialization.

    Returns
    -------
    RunState
        Initialized run state.

    Raises
    ------
    ValueError
        If ``bounded`` is not "no bounds", or the task has constraints
        :func:`task_bounds` can't turn into a pair.
    """
    update_class = optimizer_mapping(optimizer.optimizer)(
        **optimizer.hyperparameters,
        task=task,
        opt_hash=optimizer.hash,
        bounded=_bounds_from_task(task, bounded, "init_run_state"),
        stop_fn=optimizer.stopping_fn,
    )
    return RunState.init(
        update_class,
        model=task.model,
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=key,
    )
