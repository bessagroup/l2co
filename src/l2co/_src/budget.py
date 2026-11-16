"""
Module for determining the budget for optimization tasks.

A budget function returns the number of iterations a run takes. Dataset
generation (:class:`~l2co.optimization.RunOptimization`) calls it as
``budget_fn(task=task, optimizer=update_class)``, once the optimizer is
built; the meta-optimizers call it as ``budget_fn(task=task)``. The
iteration budgets (:func:`fixed_budget`, :func:`based_on_dimensionality`)
accept and ignore ``optimizer``. The evaluation budget for one optimizer
(:func:`evaluations_based_on_dimensionality`) needs it, and so works in
dataset generation only (``docs/adr/0026``). :func:`evaluation_budget`
returns that rule's number of evaluations, B, and accepts and ignores
``optimizer``: it is the budget function of a policy schedule (rl2co,
the l2co meta-optimizer), which gets B as its iteration count and stops
itself once its runs have spent B evaluations (``docs/adr/0026``,
amendment 2026-11-16). The runner picks the function per row.
"""

#                                                                       Modules
# =============================================================================

# Third-party
from l2co_optimizers import UpdateClass
from l2co_tasks import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# =============================================================================


def fixed_budget(budget: int, **kwargs) -> int:
    """
    Return a fixed budget for optimization tasks.

    Parameters
    ----------
    budget : int
        Fixed budget value.
    **kwargs : dict
        Additional arguments (ignored), such as ``task`` and
        ``optimizer``.

    Returns
    -------
    int
        The fixed budget value.
    """
    return budget


def based_on_dimensionality(
    task: Task,
    budget_per_dimension: int,
    max_iterations: int | None = None,
    *,
    optimizer: UpdateClass | None = None,
) -> int:
    """
    Calculate the budget based on the dimensionality of the task.

    An iteration budget: every optimizer gets the same number of
    iterations, whatever one iteration costs. For the same number of
    function evaluations, use
    :func:`evaluations_based_on_dimensionality`.

    Parameters
    ----------
    task : Task
        The optimization task.
    budget_per_dimension : int
        Budget allocated per dimension of the task.
    max_iterations : Optional[int], optional
        Maximum number of iterations allowed, by default None.
    optimizer : UpdateClass or None, optional
        Ignored. Accepted because dataset generation passes the built
        optimizer to every budget function.

    Returns
    -------
    int
        The calculated budget, capped at `max_iterations` if provided.
    """

    budget = task.dimensionality * budget_per_dimension

    if max_iterations is not None:
        return min(budget, max_iterations)

    else:
        return budget


def evaluation_budget(
    task: Task,
    evaluations_per_dimension: int = 1000,
    max_evaluations: int = 100_000,
    *,
    optimizer: object = None,
) -> int:
    """
    Return the number of function evaluations a run on ``task`` gets.

    ``B = min(evaluations_per_dimension * d, max_evaluations)``, where
    ``d`` is ``task.dimensionality``: the number of parameters, which for
    an embedded task is its ambient dimension. B is the same for every
    optimizer, so it needs none; :func:`evaluations_based_on_dimensionality`
    turns it into iterations for one optimizer (``docs/adr/0026``).

    This is also the budget function of a policy schedule (rl2co, the
    l2co meta-optimizer): it returns B as the iteration count, the most
    iterations B could buy at one evaluation per iteration. That only
    sizes the run's scan and history buffer; the schedule stops itself
    through ``UpdateClass.stop_fn`` once it has spent B evaluations
    (``docs/adr/0026``, amendment 2026-11-16). ``optimizer`` is accepted
    and ignored, so dataset generation's
    ``budget_fn(task=task, optimizer=update_class)`` and the
    meta-optimizers' ``budget_fn(task=task)`` both work.

    Parameters
    ----------
    task : Task
        The optimization task.
    evaluations_per_dimension : int, optional
        Function evaluations per dimension of the task, by default 1000.
    max_evaluations : int, optional
        Most function evaluations in one run, by default 100 000.
    optimizer : object, optional
        Ignored; accepted so that a schedule's runner can pass it.

    Returns
    -------
    int
        The evaluation budget B.
    """
    return int(
        min(evaluations_per_dimension * task.dimensionality, max_evaluations)
    )


def evaluations_based_on_dimensionality(
    task: Task,
    optimizer: UpdateClass | None = None,
    evaluations_per_dimension: int = 1000,
    max_evaluations: int = 100_000,
) -> int:
    """
    Give every run the same number of function evaluations, as iterations.

    The evaluation budget is
    ``B = min(evaluations_per_dimension * d, max_evaluations)``, where
    ``d`` is ``task.dimensionality``: the number of parameters, which for
    an embedded task is its ambient dimension; :func:`evaluation_budget`
    returns B itself, for callers without an optimizer. The run loop
    takes a number of iterations, so this returns ``floor(B / e)``, where
    ``e`` is what one iteration of ``optimizer`` costs:

    - ``e = optimizer.popsize`` for an optimizer with a fixed cost per
      iteration (``max_fevals_per_step`` unset). Such a run spends at
      most ``B`` evaluations.
    - ``e = 1`` for an optimizer whose cost per iteration varies, which
      it declares by setting ``max_fevals_per_step``: L-BFGS (1 to 21
      evaluations per step) and Nelder-Mead (2 on an ordinary step,
      ``n + 1`` on the first, up to ``n + 3``). Such a run can spend
      more than ``B``, and a Nelder-Mead run always does; the
      evaluations past ``B`` are meant to be ignored in
      post-processing.

    An optimizer whose population shrinks during a run (JDE21, MadDE,
    NL-SHADE-RSP) declares its own number of steps instead: when
    ``optimizer`` has a ``generations_for_budget(B, d)`` method, this
    returns its value, called with ``B`` and ``d`` as Python ints. The
    method wins over both rules for ``e`` (l2co#121).

    Dataset generation calls this with the optimizer it built
    (``docs/adr/0026``; the rationale is Section 3.2 of the dataset
    paper's protocol revision). It is for single optimizers only. A
    built ``rl2co`` or ``l2co`` bridge is an ``UpdateClass`` too, and
    this returns ``floor(B / e)`` for it without guessing that it is a
    schedule; a schedule's budget function is :func:`evaluation_budget`,
    which returns B. The runner picks the function per row, so a
    schedule row given this function is a configuration mistake
    (``docs/adr/0026``, amendment 2026-11-16).

    Parameters
    ----------
    task : Task
        The optimization task.
    optimizer : UpdateClass
        The optimizer the run uses, as built for ``task``: in dataset
        generation, the run state's ``update_class``. Required; ``None``
        is the default only so that leaving it out raises the error
        below rather than a bare signature error.
    evaluations_per_dimension : int, optional
        Function evaluations per dimension of the task, by default 1000.
    max_evaluations : int, optional
        Most function evaluations in one run, by default 100 000.

    Returns
    -------
    int
        The number of iterations, ``floor(B / e)``, or the value of
        ``optimizer.generations_for_budget(B, d)`` when it has one.

    Raises
    ------
    TypeError
        If ``optimizer`` is missing or ``None``, or is not an
        ``UpdateClass`` (an ``OptimizationStep`` is the schedule, not
        the built optimizer). Without it, ``e`` is unknown, and
        assuming ``e = 1`` would give every population method ``lambda``
        times its budget.
    ValueError
        If ``B < e``: the budget does not pay for one iteration, so the
        run would get 0 iterations. Only tiny custom settings reach
        this; the decided ones give ``B >= 2000`` at ``d >= 2``. For an
        optimizer with ``generations_for_budget``, the check is instead
        that the hook returns at least 1, because its ``popsize`` is only
        the starting size.
    """
    if optimizer is None:
        raise TypeError(
            "evaluations_based_on_dimensionality needs the optimizer: "
            "pass optimizer=<the run's UpdateClass>. It turns an "
            "evaluation budget into iterations, so without the optimizer "
            "the cost of one iteration is unknown. A policy schedule "
            "(rl2co, the l2co meta-optimizer) takes evaluation_budget "
            "instead, which returns B and needs no optimizer (l2co ADR "
            "0026)."
        )
    if not isinstance(optimizer, UpdateClass):
        raise TypeError(
            "evaluations_based_on_dimensionality needs the built "
            f"optimizer, an UpdateClass; got {type(optimizer).__name__}. "
            "Dataset generation passes the run state's update_class. A "
            "policy schedule's budget function is evaluation_budget, "
            "not this one (l2co ADR 0026)."
        )

    budget = evaluation_budget(
        task,
        evaluations_per_dimension=evaluations_per_dimension,
        max_evaluations=max_evaluations,
    )
    generations_for_budget = getattr(optimizer, "generations_for_budget", None)
    if generations_for_budget is not None:
        generations = generations_for_budget(
            int(budget), int(task.dimensionality)
        )
        if generations < 1:
            raise ValueError(
                f"an evaluation budget of {budget} is less than one "
                "iteration of this optimizer's shrink schedule: its "
                f"generations_for_budget returned {generations}, so the "
                "run would get no iterations. Raise "
                "evaluations_per_dimension or max_evaluations."
            )
        return generations
    per_iteration = _evaluations_per_iteration(optimizer)
    if budget < per_iteration:
        raise ValueError(
            f"an evaluation budget of {budget} is less than one iteration "
            f"of this optimizer, which costs {per_iteration} evaluations, "
            "so the run would get 0 iterations. Raise "
            "evaluations_per_dimension or max_evaluations."
        )
    return budget // per_iteration


def _evaluations_per_iteration(optimizer: UpdateClass) -> int:
    """What one iteration of ``optimizer`` costs, for the budget.

    ``popsize``, unless the optimizer sets ``max_fevals_per_step``: its
    cost per step then varies, and it is budgeted at one evaluation per
    iteration.
    """
    if optimizer.max_fevals_per_step is not None:
        return 1
    return optimizer.popsize
