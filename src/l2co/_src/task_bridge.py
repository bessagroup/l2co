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
* :func:`noise_free_loss` -- a noisy task's noise-free loss, as the
  ``true_loss_fn`` an ``UpdateClass`` records the noise-free series
  with (ADR 0024).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

# Third-party
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PRNGKeyArray, PyTree
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


def noise_free_loss(task: Task) -> Callable[[PyTree], Array] | None:
    """The noise-free loss of one population member of ``task``.

    The ``true_loss_fn`` an ``UpdateClass`` takes
    (:meth:`~l2co_optimizers.UpdateClass.with_true_loss_fn`): called on
    one member's unbatched parameters -- the model's inexact-array
    leaves, as the run evaluates them -- it combines them with the
    rest of the model and returns ``task.loss_fn_true`` of the result.
    The run then records, per step, the lowest noise-free loss of the
    evaluated population in ``HistoryState.output_min_true``, and noisy
    tasks are scored on that (ADR 0024). The optimizer never sees it.
    A member with a ``NaN`` parameter gets ``NaN``, so it never counts,
    even under a ``loss_fn_true`` that would map it to a finite value.

    Parameters
    ----------
    task : Task
        The task about to be run.

    Returns
    -------
    Callable[[PyTree], Array] or None
        ``None`` for a noise-free task (no ``loss_fn_true``): there is
        nothing to attach, and the observed loss already is the
        noise-free one.

    Raises
    ------
    NotImplementedError
        If the task has a dataset. ``true_loss_fn`` is called without a
        data batch, so a noise-free loss that takes one
        (``loss_fn_true(model, **sample)``) cannot be recorded yet. No
        such task exists today.
    """
    if task.loss_fn_true is None:
        return None
    # The same test ``Task.loaded_dataset`` makes for "has data".
    if task.dataset is not None and (
        task.dataset.get("dataset_path") is not None
    ):
        raise NotImplementedError(
            f"task {task.name!r} has a dataset and a noise-free loss; "
            "the run calls the noise-free loss without a data batch, so "
            "recording it for a task with data is not supported yet "
            "(l2co ADR 0024)"
        )
    _, static = eqx.partition(task.model, eqx.is_inexact_array)
    loss_fn_true = task.loss_fn_true

    # Captures the static part and the function, not the task, whose
    # model arrays would otherwise ride along in this static field.
    def true_loss(params: PyTree) -> Array:
        value = loss_fn_true(eqx.combine(params, static))
        # A member with a NaN parameter is no evaluated point (the
        # meta-optimizer pads unused population rows with NaN): NaN,
        # whatever ``loss_fn_true`` makes of it -- the rule
        # ``row_has_nan`` applies to the observed loss in the menu
        # evaluator.
        has_nan = jnp.zeros((), dtype=bool)
        for leaf in jtu.tree_leaves(params):
            has_nan = has_nan | jnp.isnan(leaf).any()
        return jnp.where(has_nan, jnp.nan, value)

    return true_loss


def init_run_state(
    optimizer: OptimizationStep,
    task: Task,
    key: PRNGKeyArray,
    *,
    record_noise_free: bool = False,
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
    key : PRNGKeyArray
        PRNG key for the batch state and optimizer initialization.
    record_noise_free : bool, optional
        Attach the task's :func:`noise_free_loss` to the optimizer, so a
        run on a noisy task also records its noise-free series
        (``HistoryState.output_min_true``; ADR 0024). A noise-free task
        is unaffected either way. Off by default: a caller that appends
        run histories into one buffer, as rl2co's environment does
        through ``HistoryState.add``, would see the buffer change
        structure once a part carries the series.
        :class:`~l2co.optimization.RolloutWrapper`, which dataset
        generation runs through, turns it on.

    Returns
    -------
    RunState
        Initialized run state.

    Raises
    ------
    ValueError
        If the task has constraints :func:`task_bounds` can't turn into
        a pair.
    NotImplementedError
        If ``record_noise_free`` is set for a noisy task with a dataset
        (see :func:`noise_free_loss`).

    Notes
    -----
    What attaching covers (ADR 0024). Every bare optimizer records the
    series through its own run loop. A meta-optimizer built here
    (``"l2co"``, rl2co's ``"rl2co"``) records it too: its run loop is
    the inherited one and its step reports the population it asked for,
    so the series is the noise-free minimum over the rows its active
    sub-optimizer evaluated. The sub-optimizers inside it are not given
    the noise-free loss and need none. The whole-run scipy entries
    (``"slsqp"``, ``"trustconstr"``) cannot record the series and raise
    ``NotImplementedError`` when run with it; noisy suites do not
    schedule them.
    """
    update_class = optimizer_mapping(optimizer.optimizer)(
        **optimizer.hyperparameters,
        task=task,
        opt_hash=optimizer.hash,
        bounded=task_bounds(task),
        stop_fn=optimizer.stopping_fn,
    )
    if record_noise_free:
        true_loss = noise_free_loss(task)
        if true_loss is not None:
            update_class = update_class.with_true_loss_fn(true_loss)
    return RunState.init(
        update_class,
        model=task.model,
        dataset=task.loaded_dataset,
        batch_size=task.batch_size,
        key=key,
    )
