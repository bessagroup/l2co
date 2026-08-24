"""What one optimizer can hand the next when control passes to it.

The handshake in :mod:`~l2co._src.optimizers.handshake_policy` answers
*which population* the incoming optimizer sees and *whether* its internal
state survives. It carries no information **out of** the outgoing
optimizer: the incoming one is handed a point and, at best, told to keep
or discard its own accumulated state. Everything the outgoing optimizer
learned about the landscape is discarded at every switch.

This module is the channel for that information. It is deliberately
**not** a pairwise ``(outgoing, incoming)`` table. Each optimizer gets
one *reader* that projects its own state onto a small canonical bundle,
and one *writer* that folds a bundle into its own state -- ``2N``
adapters rather than ``N²``. The justification is maintainability
(``2N`` hand-written adapters against ``N²`` for a registry of 59),
compile time and code size (a pairwise dispatch would trace ``N²``
branches), and redundancy (most of those ``N²`` cells would be derived
from the same handful of statistics anyway). It is explicitly **not** a
per-tick runtime argument: rl2co's ``map_unroll`` and l2co's
:func:`~l2co._src.update_class.batch_run_sequential_` both execute only
the branch actually selected, so per-tick cost does not scale with the
branch count.

What is carried
---------------
Only **scale** -- a single positive scalar ``sigma`` -- plus the location
it belongs to. That is not a placeholder for a richer bundle so much as
the slot that carries almost all of the available signal: over the
registry, 27 optimizers hold an explicit ``std`` and 20 hold a diagonal
second moment, and on a menu mixing gradient-based and
distribution-based optimizers the *initial* scales can differ by three
orders of magnitude (``rprop`` at ``learning_rate=1e-3`` against an ES at
``std_init=1.0``). Shape and direction are deliberately absent; see
``docs/adr/0013``.

The governing principle for writes is that **scale goes stale and shape
does not**. An optimizer that has not run for many generations holds a
``std`` that is wrong about the current search, but its *relative*
per-coordinate structure is still informative. So a write never
overwrites a per-coordinate quantity with a flat scalar -- it **rescales**
it so its RMS matches the incoming ``sigma`` while the relative structure
survives. Writing ``sigma`` flat into ``rprop``'s ``step_sizes`` would
destroy exactly the per-coordinate ratchet that makes ``rprop`` work.

Who calls this
--------------
The population half of a switch stays with the caller, keyed on
:class:`TransferSpec.family`:

* ``distribution`` -- the caller writes location and scale through
  :func:`build_transfer_fns`'s writer, then asks the incoming optimizer
  for its **own** population. That is what keeps the ask/tell-coupled
  algorithms correct: CR-FM-NES and seven relatives cache their draws in
  state and consume them on the next ``tell``, so substituting a foreign
  population silently credits each candidate's fitness to a direction
  that produced a different point (see ``docs/adr/0012``). Regenerating
  the population through the optimizer's own ``ask`` makes the cached
  draws consistent by construction.
* ``population`` -- the stored population *is* the state; the caller
  keeps the existing ``best_one`` layout.
* ``gradient`` -- unit population; the caller keeps ``best``.

Reader and writer are built at optimizer-construction time rather than
looked up per call, because both need hyperparameters that are not
recoverable from the state: ``rprop``'s ``min_step_size`` /
``max_step_size`` clip bounds, the learning rate of every optax
optimizer (``scale_by_learning_rate`` carries an ``EmptyState``), and the
ravel function an evosax algorithm uses to flatten a solution pytree.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

# =============================================================================

__all__ = [
    "CONF_ABSENT",
    "CONF_ESTIMATED",
    "CONF_EXACT",
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_GRADIENT",
    "FAMILY_POPULATION",
    "TRANSFER_OVERRIDES",
    "TransferBundle",
    "TransferSpec",
    "build_transfer_fns",
    "transfer_spec_for",
]


#                                                            Confidence levels
# =============================================================================

CONF_ABSENT = 0.0
"""The outgoing optimizer holds no scale information. Writers skip."""

CONF_ESTIMATED = 0.5
"""Scale inferred from samples or from an unrecognised shape field.

Written the same way an exact value is -- the distinction is carried so a
later blend rule can weigh it, and so ``CONF_ABSENT`` stays the only
level that changes behaviour.
"""

CONF_EXACT = 1.0
"""Scale read directly off a field that *is* the optimizer's scale."""


#                                                          Optimizer families
# =============================================================================

FAMILY_DISTRIBUTION = "distribution"
"""Holds an explicit search distribution; regenerates its own population."""

FAMILY_POPULATION = "population"
"""The stored population is the state."""

FAMILY_GRADIENT = "gradient"
"""Unit population, iterate plus a preconditioner."""

FAMILIES = (FAMILY_DISTRIBUTION, FAMILY_POPULATION, FAMILY_GRADIENT)


#                                                              The bundle
# =============================================================================


class TransferBundle(eqx.Module):
    """What crosses a switch.

    Every reader must produce this same structure: it is the output of a
    ``lax.switch`` over the outgoing optimizer's index, so all branches
    must agree on layout leaf for leaf.

    Attributes
    ----------
    x : PyTree
        Best parameters seen so far, shaped as one solution (no leading
        population axis).
    f : Float[Array, ""]
        Loss at ``x``.
    sigma : Float[Array, ""]
        Characteristic scale of the outgoing optimizer's search: the RMS
        per-coordinate standard deviation for a sampler, the RMS step
        length for a stepper. Meaningless unless ``conf_sigma`` is above
        :data:`CONF_ABSENT`.
    conf_sigma : Float[Array, ""]
        One of :data:`CONF_ABSENT`, :data:`CONF_ESTIMATED`,
        :data:`CONF_EXACT`.
    """

    x: PyTree
    f: Float[Array, ""]
    sigma: Float[Array, ""]
    conf_sigma: Float[Array, ""]

    @classmethod
    def absent(cls, x: PyTree, f: Float[Array, ""]) -> TransferBundle:
        """Build a bundle carrying location only, no scale.

        Parameters
        ----------
        x : PyTree
            Best parameters seen so far.
        f : Float[Array, ""]
            Loss at ``x``.

        Returns
        -------
        TransferBundle
            Bundle whose ``conf_sigma`` is :data:`CONF_ABSENT`.
        """
        return cls(
            x=x,
            f=jnp.asarray(f, dtype=float),
            sigma=jnp.asarray(0.0, dtype=float),
            conf_sigma=jnp.asarray(CONF_ABSENT, dtype=float),
        )


#                                                              Sigma readers
# =============================================================================


def _rms(x: Array) -> Array:
    """Root-mean-square of ``x``, NaN-safe over an all-NaN input."""
    return jnp.sqrt(jnp.nanmean(jnp.square(jnp.asarray(x, dtype=float))))


def _shape_factor_sq(opt_state: PyTree) -> tuple[Array, float]:
    """Per-coordinate variance factor implied by an ES's shape fields.

    An evosax distribution-based algorithm samples
    ``x = mean + std * T(z)`` for some shape transform ``T`` held in the
    state. The RMS per-coordinate standard deviation is therefore
    ``std * sqrt(mean(diag(T Tᵀ)))``, and this returns that
    ``mean(diag(T Tᵀ))`` together with the confidence it deserves.

    Recognised, each verified against the algorithm's ``_ask``:

    * ``D`` and ``v`` -- CR-FM-NES, ``Cov = std² D (I + v vᵀ) D``, so the
      diagonal is ``D² (1 + v²)`` (``cr_fm_nes.py:176-184``).
    * ``C`` one-dimensional -- Sep-CMA-ES, ``x = mean + std * (sqrt(C) *
      z)`` (``sep_cma_es.py:66-78``).
    * ``C`` two-dimensional -- CMA-ES, ``Cov = std² C``
      (``cma_es.py:162-176``).
    * neither -- an identity transform, e.g. SNES or OpenES, where the
      per-coordinate scale lives entirely in ``std``
      (``snes.py:76-84``).

    Anything else (``maes``/``lmmaes``'s ``M``, ``xnes``'s ``B``) falls
    back to ``1.0`` and downgrades the confidence rather than guessing a
    relation.

    Parameters
    ----------
    opt_state : PyTree
        The algorithm's own state.

    Returns
    -------
    tuple[Array, float]
        ``(factor_sq, confidence)``.
    """
    D = getattr(opt_state, "D", None)
    v = getattr(opt_state, "v", None)
    if D is not None and v is not None:
        D = jnp.asarray(D, dtype=float)
        v = jnp.asarray(v, dtype=float)
        return jnp.nanmean(jnp.square(D) * (1.0 + jnp.square(v))), CONF_EXACT

    C = getattr(opt_state, "C", None)
    if C is not None:
        C = jnp.asarray(C, dtype=float)
        if C.ndim == 1:
            return jnp.nanmean(C), CONF_EXACT
        if C.ndim == 2:
            return jnp.nanmean(jnp.diagonal(C)), CONF_EXACT

    # No shape field at all: the transform is the identity and ``std``
    # already carries the whole per-coordinate scale.
    if not any(
        getattr(opt_state, name, None) is not None
        for name in ("M", "B", "D", "v", "C")
    ):
        return jnp.asarray(1.0), CONF_EXACT

    return jnp.asarray(1.0), CONF_ESTIMATED


def read_sigma_distribution(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the sampling scale off a distribution-based ES state.

    ``std`` is a scalar for most algorithms and a per-coordinate vector
    for a few (SNES); ``_rms`` handles both, and the shape fields supply
    the remaining factor.

    Parameters
    ----------
    opt_state : PyTree
        The algorithm's own state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    std = getattr(opt_state, "std", None)
    if std is None:
        return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

    factor_sq, conf = _shape_factor_sq(opt_state)
    sigma = _rms(std) * jnp.sqrt(factor_sq)
    return _finite_or_absent(sigma, conf)


def read_sigma_population(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the search scale off a population-based state.

    Three sources, in order of how directly they mean "scale":
    an explicit mutation ``std`` (the GA family), the RMS ``velocity``
    (PSO -- its per-iteration step), and otherwise the RMS
    per-coordinate spread of the stored population (DE, which holds
    neither).

    Parameters
    ----------
    opt_state : PyTree
        The algorithm's own state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    std = getattr(opt_state, "std", None)
    if std is not None:
        return _finite_or_absent(_rms(std), CONF_EXACT)

    velocity = getattr(opt_state, "velocity", None)
    if velocity is not None:
        # Zero at init, before any ``ask`` has moved a particle -- that
        # is absence of information, not a scale of zero.
        return _finite_or_absent(_rms(velocity), CONF_ESTIMATED)

    population = getattr(opt_state, "population", None)
    if population is not None:
        population = jnp.asarray(population, dtype=float)
        spread = jnp.sqrt(jnp.nanmean(jnp.nanvar(population, axis=0)))
        return _finite_or_absent(spread, CONF_ESTIMATED)

    return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)


#: Field names optax uses for the first moment, most specific first.
_MOMENTUM_NAMES = ("mu", "m", "trace")

#: Field names optax uses for the second moment that *divides* the step,
#: most specific first. ``adan``'s denominator is ``n`` -- its ``v`` is a
#: moment of gradient *differences* and appears in the numerator
#: (``transform.py:653-680``) -- so ``n`` must be tried before ``v``,
#: which is ``adafactor``'s unfactored second moment.
_SECOND_MOMENT_NAMES = ("nu", "n", "v", "sum_of_squares")

#: Field names optax uses for the update counter.
_COUNT_NAMES = ("count", "t")


def _get_first(opt_state: PyTree, names: tuple[str, ...]):
    """Return the first present field among ``names``, with its name.

    optax spells the same quantity differently across transforms
    (``mu``/``m``/``trace``, ``nu``/``n``/``v``), so the readers and
    writers resolve by alias rather than assuming Adam's naming.

    Parameters
    ----------
    opt_state : PyTree
        An optax optimizer state, possibly a chain.
    names : tuple[str, ...]
        Candidate field names, most specific first.

    Returns
    -------
    tuple[Array | None, str | None]
        ``(value, name)``, or ``(None, None)`` if none are present.
    """
    for name in names:
        value = optax.tree.get(opt_state, name)
        if value is not None:
            return value, name
    return None, None


def make_read_sigma_gradient(learning_rate: float) -> Callable:
    """Build a reader for an optax second-moment optimizer.

    The transferable scale is the RMS of the *step* the optimizer would
    take, ``learning_rate * mu / (sqrt(nu) + eps)``. Neither the learning
    rate nor the bias-correction coefficients are recoverable from the
    state -- ``scale_by_learning_rate`` carries an ``EmptyState`` -- so
    the rate is closed over here and bias correction is skipped, which is
    why the result is only :data:`CONF_ESTIMATED`.

    Parameters
    ----------
    learning_rate : float
        The rate the optimizer was constructed with.

    Returns
    -------
    Callable
        ``reader(opt_state) -> (sigma, conf_sigma)``.
    """

    def read(opt_state: PyTree) -> tuple[Array, Array]:
        momentum, _ = _get_first(opt_state, _MOMENTUM_NAMES)
        if momentum is None:
            # A preconditioner without a stored step (RMSProp, Adagrad,
            # Adafactor) cannot report a scale: the step it would take
            # depends on a gradient magnitude nothing retained. Such
            # optimizers are receivers only -- see
            # :func:`make_write_gradient`.
            return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

        step = jnp.asarray(momentum, dtype=float)
        second, _ = _get_first(opt_state, _SECOND_MOMENT_NAMES)
        if second is not None:
            step = step / (jnp.sqrt(jnp.asarray(second, dtype=float)) + 1e-8)

        sigma = float(learning_rate) * _rms(step)
        sigma, conf = _finite_or_absent(sigma, CONF_ESTIMATED)

        count, _ = _get_first(opt_state, _COUNT_NAMES)
        if count is not None:
            # Before the first update the moments are zeros, so the step
            # RMS is zero and means nothing.
            conf = jnp.where(jnp.asarray(count) > 0, conf, CONF_ABSENT)
        return sigma, conf

    return read


def read_sigma_rprop(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the scale off Rprop's per-coordinate step sizes.

    ``step_sizes`` *is* a length, which makes Rprop the one optimizer
    whose scale field needs no unit conversion. Its provenance is a
    success-history ratchet rather than a curvature estimate, but the
    quantity transferred here is scale, and for scale it is exact.

    Parameters
    ----------
    opt_state : PyTree
        Rprop's optax state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    step_sizes = optax.tree.get(opt_state, "step_sizes")
    if step_sizes is None:
        return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)
    return _finite_or_absent(_rms(step_sizes), CONF_EXACT)


def read_sigma_lbfgs(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the scale off L-BFGS's most recent accepted step.

    ``diff_params_memory`` holds the last ``memory_size`` parameter
    differences, so the previous step is literally in the state.

    The ring index is ``(count - 2) % memory_size``, which is **not**
    optax's own expression even though it names the same slot. Optax
    writes to ``prev_memory_idx = (state.count - 1) % memory_size``
    (``transform.py:1734``) and *then* returns ``count + 1``
    (``:1798``), so its ``state.count`` is the pre-increment value while
    a reader holding the returned state sees the post-increment one --
    one further along. Copying the expression across that frame shift
    reads the next, not-yet-written slot: zeros until the ring fills, and
    a step one generation stale afterwards. Verified empirically -- after
    ``n`` steps ``count == n`` and rows ``0 .. n-2`` are populated.

    Two steps are needed before there is any difference to read at all,
    hence the ``count > 1`` gate rather than ``count > 0``.

    Parameters
    ----------
    opt_state : PyTree
        L-BFGS's optax state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    memory = optax.tree.get(opt_state, "diff_params_memory")
    count = optax.tree.get(opt_state, "count")
    if memory is None or count is None:
        return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

    memory = jnp.asarray(memory, dtype=float)
    memory_size = memory.shape[0]
    prev = (jnp.asarray(count) - 2) % memory_size
    last_step = jnp.take(memory, prev, axis=0)

    sigma, conf = _finite_or_absent(_rms(last_step), CONF_EXACT)
    conf = jnp.where(jnp.asarray(count) > 1, conf, CONF_ABSENT)
    return sigma, conf


def read_sigma_absent(opt_state: PyTree) -> tuple[Array, Array]:
    """Report no scale information, for optimizers that hold none.

    Parameters
    ----------
    opt_state : PyTree
        Ignored.

    Returns
    -------
    tuple[Array, Array]
        ``(0.0, CONF_ABSENT)``.
    """
    del opt_state
    return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)


def _finite_or_absent(sigma, conf) -> tuple[Array, Array]:
    """Downgrade a non-finite or non-positive scale to absence.

    A zero scale is what an uninitialised velocity or second moment
    looks like, and writing it would freeze the incoming optimizer; a
    NaN would poison it. Both are absence of information.

    Parameters
    ----------
    sigma : Array or float
        Candidate scale.
    conf : Array or float
        Confidence the reader would otherwise claim.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf)`` with ``conf`` forced to :data:`CONF_ABSENT`
        where ``sigma`` is not finite and positive.
    """
    sigma = jnp.asarray(sigma, dtype=float)
    usable = jnp.isfinite(sigma) & (sigma > 0.0)
    return (
        jnp.where(usable, sigma, 0.0),
        jnp.where(usable, jnp.asarray(conf, dtype=float), CONF_ABSENT),
    )


def _all_finite(tree: PyTree) -> Array:
    """Whether every float leaf of ``tree`` is finite.

    Parameters
    ----------
    tree : PyTree
        Tree to check; non-float leaves are ignored.

    Returns
    -------
    Array
        Scalar boolean. A tree with no float leaves reports ``False`` --
        nothing usable is present.
    """
    leaves = [
        jnp.asarray(leaf)
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x.astype(float))) for x in leaves])
    )


def _usable_sigma(bundle: TransferBundle) -> Array:
    """Whether ``bundle.sigma`` may be written.

    Re-checks finiteness rather than trusting ``conf_sigma`` alone. Every
    reader here routes through :func:`_finite_or_absent`, so a
    non-finite ``sigma`` paired with a confident ``conf_sigma`` cannot
    arise from this module -- but a hand-built bundle or a future reader
    that forgets the helper would poison the receiver silently, and the
    check costs one comparison.

    Parameters
    ----------
    bundle : TransferBundle
        The bundle being written.

    Returns
    -------
    Array
        Scalar boolean gate.
    """
    return (
        (bundle.conf_sigma > CONF_ABSENT)
        & jnp.isfinite(bundle.sigma)
        & (bundle.sigma > 0.0)
    )


#                                                              State writers
# =============================================================================


def _blend(use: Array, new: Array, old: Array) -> Array:
    """Select ``new`` where ``use`` holds, elementwise, keeping dtype."""
    return jnp.where(use, new, old).astype(jnp.asarray(old).dtype)


def _rescale_to(values: Array, sigma: Array, use: Array) -> Array:
    """Scale ``values`` so their RMS becomes ``sigma``.

    The relative structure across coordinates is preserved -- this is the
    "scale goes stale, shape does not" rule. A degenerate input (all
    zeros, so no relative structure to preserve) is left alone rather
    than divided by zero.

    Parameters
    ----------
    values : Array
        Per-coordinate quantity to rescale.
    sigma : Array
        Target RMS.
    use : Array
        Boolean gate; where false, ``values`` passes through.

    Returns
    -------
    Array
        Rescaled values.
    """
    values = jnp.asarray(values, dtype=float)
    current = _rms(values)
    safe = jnp.isfinite(current) & (current > 0.0)
    factor = jnp.where(safe, sigma / jnp.where(safe, current, 1.0), 1.0)
    return _blend(use & safe, values * factor, values)


def make_write_distribution(ravel_fn: Callable) -> Callable:
    """Build the writer for a distribution-based ES.

    Writes the handed-over location into ``mean``, sets the scale, and
    zeros the cumulation paths. Shape fields (``C``, ``D``, ``v``, ``B``,
    ``M``) are deliberately **kept**: they describe the geometry of the
    landscape, which does not go stale the way scale does.

    Zeroing ``p_std`` / ``p_c`` is the same reasoning restart schemes
    (IPOP/BIPOP) use. A cumulation path is calibrated against the
    expected step length under the algorithm's *own* distribution, so
    carrying one across a jump to a foreign iterate corrupts step-size
    control; keeping the covariance while resetting the paths is the
    standard restart.

    ``mean`` is stored **raveled** -- evosax flattens solutions
    internally (``distribution_based/base.py:60-69``) -- so the bundle's
    pytree ``x`` has to go through the algorithm's own ravel function.

    Parameters
    ----------
    ravel_fn : Callable
        The algorithm's ``_ravel_solution``.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        updates: dict[str, Array] = {}

        # A non-finite location is the one input that cannot be
        # rescaled, clipped or downgraded into something usable: it
        # would set ``mean`` to NaN, every ``ask`` from that
        # distribution returns NaN, every loss is NaN, and no later
        # ``tell`` can recover it -- the optimizer is dead for the rest
        # of the episode, including after switching away and back, since
        # the ``RunState`` persists. So the fallback is to keep the
        # optimizer's own mean and let it carry on from where it was.
        use_x = _all_finite(bundle.x)
        mean = getattr(opt_state, "mean", None)
        if mean is not None:
            updates["mean"] = _blend(
                use_x,
                jnp.asarray(ravel_fn(bundle.x), dtype=float),
                jnp.asarray(mean, dtype=float),
            )

        use = _usable_sigma(bundle)
        std = getattr(opt_state, "std", None)
        if std is not None:
            std = jnp.asarray(std)
            # ``sigma`` is an RMS per-coordinate standard deviation, but
            # ``std`` is only one factor of that: the target keeps its own
            # shape fields, which contribute ``sqrt(factor_sq)``. Divide
            # that out so the *resulting* sampling scale is ``sigma``,
            # making the writer the exact inverse of the reader.
            factor_sq, _ = _shape_factor_sq(opt_state)
            safe = jnp.isfinite(factor_sq) & (factor_sq > 0.0)
            target = bundle.sigma / jnp.sqrt(jnp.where(safe, factor_sq, 1.0))
            if std.ndim == 0:
                updates["std"] = _blend(use, target, std)
            else:
                # Per-coordinate std (SNES): rescale, never flatten.
                updates["std"] = _rescale_to(std, target, use)

        for path in ("p_std", "p_c"):
            value = getattr(opt_state, path, None)
            if value is not None:
                updates[path] = jnp.zeros_like(value)

        return opt_state.replace(**updates) if updates else opt_state

    return write


def make_write_population(repair_baseline: bool) -> Callable:
    """Build the writer for a population-based algorithm.

    Scale lands in whichever field carries it: an explicit mutation
    ``std`` (GA family) or the ``velocity`` (PSO, rescaled so its RMS
    matches while each particle's direction survives). The stored
    population is left to the caller's ``best_one`` handshake.

    ``repair_baseline`` additionally overwrites the stored
    ``population`` / ``fitness`` with the handed-over best. That is
    needed where the algorithm adapts its scale by comparing the incoming
    generation against its *own stored* elite: MR15-GA's 1/5 rule is
    ``mean(fitness < state.fitness)`` (``mr15_ga.py:96``), so a stale
    elite from an early generation makes a population drawn around a much
    better point look like a near-total success and **doubles** the
    mutation scale exactly when it should shrink -- undoing the scale
    write one generation later. Rebasing on the handed-over best makes
    the rule ask the honest question. It also keeps the algorithm's
    ``(µ+λ)`` selection over ``concat(new, old)`` coherent, which
    overwriting ``fitness`` alone would not: stale rows would keep their
    positions while claiming the new fitness.

    Parameters
    ----------
    repair_baseline : bool
        Whether to rebase ``population`` / ``fitness`` on the bundle.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        updates: dict[str, Array] = {}
        use = _usable_sigma(bundle)

        std = getattr(opt_state, "std", None)
        if std is not None:
            std = jnp.asarray(std)
            if std.ndim == 0:
                updates["std"] = _blend(use, bundle.sigma, std)
            else:
                updates["std"] = _rescale_to(std, bundle.sigma, use)

        velocity = getattr(opt_state, "velocity", None)
        if velocity is not None:
            updates["velocity"] = _rescale_to(velocity, bundle.sigma, use)

        if repair_baseline:
            population = getattr(opt_state, "population", None)
            fitness = getattr(opt_state, "fitness", None)
            if population is not None and fitness is not None:
                # Both halves move together or neither does: rebasing
                # the points while keeping the old losses (or the
                # reverse) leaves the ``(mu + lambda)`` selection
                # incoherent, rows claiming a fitness they did not earn.
                #
                # ``f`` is gated as tightly as ``x``, and ``+inf`` is
                # the case that matters rather than NaN. ``best_loss``
                # starts at ``+inf`` and stays there while nothing has
                # improved -- a first generation that is entirely NaN,
                # which this ecosystem sees at high ambient dimension.
                # Writing that in makes *every* candidate beat the
                # stored elite, so the 1/5 rule reads a 100% success
                # rate and **doubles** the width immediately after the
                # scale write set it: measured success 1.00 against 0.00
                # for a finite baseline. That is precisely the failure
                # this repair exists to prevent, so an unusable ``f``
                # must leave the optimizer's own elite alone.
                repair_ok = _all_finite(bundle.x) & jnp.isfinite(bundle.f)
                flat = jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(leaf, dtype=float))
                        for leaf in jax.tree.leaves(bundle.x)
                    ]
                )
                updates["population"] = _blend(
                    repair_ok,
                    jnp.broadcast_to(flat, jnp.shape(population)),
                    jnp.asarray(population),
                )
                updates["fitness"] = _blend(
                    repair_ok,
                    jnp.full_like(jnp.asarray(fitness), bundle.f),
                    jnp.asarray(fitness),
                )

        return opt_state.replace(**updates) if updates else opt_state

    return write


def make_write_gradient(learning_rate: float) -> Callable:
    """Build the writer for an optax second-moment optimizer.

    The per-coordinate step is ``learning_rate * mu / (sqrt(nu) + eps)``,
    so scaling ``nu`` by ``k²`` scales the step by ``1/k``. To move the
    step RMS from its current value to ``sigma`` the second moment is
    multiplied by ``(rms_own / sigma)²``, which leaves the *relative*
    per-coordinate preconditioner untouched -- writing a flat value into
    ``nu`` would erase it.

    Parameters
    ----------
    learning_rate : float
        The rate the optimizer was constructed with.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        momentum, momentum_name = _get_first(opt_state, _MOMENTUM_NAMES)
        second, second_name = _get_first(opt_state, _SECOND_MOMENT_NAMES)
        if momentum is None and second is None:
            return opt_state
        if second is not None:
            second = jnp.asarray(second, dtype=float)

        if momentum is not None:
            momentum = jnp.asarray(momentum, dtype=float)
            step = momentum
            if second is not None:
                step = step / (jnp.sqrt(second) + 1e-8)
        else:
            # Preconditioner but no stored step (RMSProp, Adagrad,
            # Adafactor): the reference is the step the optimizer would
            # take for a unit gradient. That makes these optimizers
            # receivers even though the reader cannot make them
            # providers -- you cannot measure a step that was never
            # stored, but you can still rescale the preconditioner.
            step = 1.0 / (jnp.sqrt(second) + 1e-8)
        rms_own = float(learning_rate) * _rms(step)

        use = _usable_sigma(bundle)
        use = use & jnp.isfinite(rms_own) & (rms_own > 0.0)
        count, _ = _get_first(opt_state, _COUNT_NAMES)
        if count is not None:
            use = use & (jnp.asarray(count) > 0)

        ratio = jnp.where(
            use, rms_own / jnp.where(use, bundle.sigma, 1.0), 1.0
        )
        if second is not None:
            # Scaling the second moment by k² scales the step by 1/k,
            # leaving the relative per-coordinate preconditioner intact.
            return optax.tree.set(
                opt_state,
                **{
                    second_name: _blend(
                        use, second * jnp.square(ratio), second
                    )
                },
            )
        # Momentum-only (SGD with momentum, LARS): no preconditioner to
        # rescale, so the step scale lives in the trace itself.
        return optax.tree.set(
            opt_state,
            **{momentum_name: _blend(use, momentum / ratio, momentum)},
        )

    return write


def make_write_rprop(min_step_size: float, max_step_size: float) -> Callable:
    """Build the writer for Rprop.

    ``step_sizes`` is rescaled so its RMS matches ``sigma``, preserving
    the per-coordinate ratchet, then clipped back into the bounds Rprop
    was constructed with -- ``update_fn`` clips on every step, so a write
    that ignored them would be silently saturated on the next one.

    ``prev_updates`` is zeroed. It holds the previous signed step, and
    Rprop's ratchet keys on ``sign(g * prev_updates)``; across an iterate
    jump those signs describe a different neighbourhood, so keeping them
    fires a spurious increase or decrease on every coordinate whose
    gradient sign happened to flip. With ``prev_updates`` zero the
    product is zero, which is the branch that leaves ``step_sizes``
    untouched and takes one clean step (``transform.py:928-940``).

    Parameters
    ----------
    min_step_size : float
        Lower clip bound the optimizer was constructed with.
    max_step_size : float
        Upper clip bound the optimizer was constructed with.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        step_sizes = optax.tree.get(opt_state, "step_sizes")
        if step_sizes is None:
            return opt_state

        use = _usable_sigma(bundle)
        rescaled = _rescale_to(step_sizes, bundle.sigma, use)
        rescaled = jnp.clip(rescaled, min_step_size, max_step_size)

        opt_state = optax.tree.set(
            opt_state, step_sizes=_blend(use, rescaled, step_sizes)
        )
        prev = optax.tree.get(opt_state, "prev_updates")
        if prev is not None:
            opt_state = optax.tree.set(
                opt_state,
                prev_updates=_blend(use, jnp.zeros_like(prev), prev),
            )
        return opt_state

    return write


def write_none(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
    """Accept nothing, for optimizers with no slot a scale survives in.

    L-BFGS is the case: after ``init`` its state is all zeros with
    ``learning_rate = 1.0`` and ``value = inf``, and the first step
    length comes from ``identity_scale = min(1, 1/‖g‖)`` before the zoom
    line search overwrites it. There is nowhere for a scale to land.

    Parameters
    ----------
    bundle : TransferBundle
        Ignored.
    opt_state : PyTree
        Returned unchanged.

    Returns
    -------
    PyTree
        ``opt_state``.
    """
    del bundle
    return opt_state


#                                                              The policy
# =============================================================================


class TransferSpec(eqx.Module):
    """Which reader and writer one optimizer needs.

    Attributes
    ----------
    family : str
        One of :data:`FAMILIES`. Read by the *caller* to decide the
        population half of a switch -- see the module docstring.
    reader : str
        Key into the reader implementations.
    writer : str
        Key into the writer implementations.
    """

    family: str = eqx.field(static=True)
    reader: str = eqx.field(static=True)
    writer: str = eqx.field(static=True)


_FAMILY_DEFAULTS: dict[str, TransferSpec] = {
    FAMILY_DISTRIBUTION: TransferSpec(
        FAMILY_DISTRIBUTION, "distribution", "distribution"
    ),
    FAMILY_POPULATION: TransferSpec(
        FAMILY_POPULATION, "population", "population"
    ),
    FAMILY_GRADIENT: TransferSpec(FAMILY_GRADIENT, "gradient", "gradient"),
}

TRANSFER_OVERRIDES: dict[str, TransferSpec] = {
    # Reads its own step length exactly; receives one exactly. The only
    # optimizer whose scale field is natively a length.
    "rprop": TransferSpec(FAMILY_GRADIENT, "rprop", "rprop"),
    # Its last accepted step is in the state; nothing survives a write.
    "lbfgs": TransferSpec(FAMILY_GRADIENT, "lbfgs", "none"),
    # Adapts scale against its own stored elite -- see
    # ``make_write_population``.
    "mr15ga": TransferSpec(FAMILY_POPULATION, "population", "mr15ga"),
    # Holds no persistent search state at all.
    "randomsearch": TransferSpec(FAMILY_POPULATION, "none", "none"),
}
"""Per-optimizer departures from the family default, keyed as in the registry.

Deliberately short. The family default is right for the large majority,
and every entry here exists because a specific field was read and found
to behave differently -- not because the optimizer is unusual in general.
"""


def transfer_spec_for(name: str, family: str) -> TransferSpec:
    """Resolve the transfer spec for one optimizer.

    Parameters
    ----------
    name : str
        Registry name, normalised as in
        :func:`~l2co._src.optimizers.utils.normalize_key`.
    family : str
        One of :data:`FAMILIES`, supplied by the constructing factory,
        which knows which family it is building. Not inferred from
        registry membership: ``FEEDBACK_evosax.md`` records that
        membership-based family dispatch is already a source of friction.

    Returns
    -------
    TransferSpec
        The override for ``name`` if one exists, else the family default.

    Raises
    ------
    ValueError
        If ``family`` is not one of :data:`FAMILIES`.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}; got {family!r}")
    return TRANSFER_OVERRIDES.get(name, _FAMILY_DEFAULTS[family])


def build_transfer_fns(
    name: str,
    family: str,
    hyperparameters: dict | None = None,
    ravel_fn: Callable | None = None,
) -> tuple[Callable, Callable]:
    """Bind one optimizer's reader and writer at construction time.

    Both halves need values that are not recoverable from the state, so
    they are closed over here rather than looked up per call: the
    learning rate (``scale_by_learning_rate`` holds an ``EmptyState``),
    Rprop's clip bounds, and the ravel function evosax uses to flatten a
    solution pytree.

    Parameters
    ----------
    name : str
        Registry name.
    family : str
        One of :data:`FAMILIES`.
    hyperparameters : dict or None, optional
        The hyperparameters the optimizer was constructed with. Missing
        keys fall back to the underlying library's documented defaults.
    ravel_fn : Callable or None, optional
        The algorithm's ``_ravel_solution``. Required for
        ``distribution`` -- evosax stores ``mean`` raveled.

    Returns
    -------
    tuple[Callable, Callable]
        ``(reader, writer)``. ``reader(opt_state) -> (sigma, conf)``;
        ``writer(bundle, opt_state) -> opt_state``.

    Raises
    ------
    ValueError
        If a ``distribution`` writer is requested without ``ravel_fn``.
    """
    hyperparameters = dict(hyperparameters or {})
    spec = transfer_spec_for(name, family)

    learning_rate = hyperparameters.get("learning_rate", 1.0)

    readers: dict[str, Callable] = {
        "distribution": read_sigma_distribution,
        "population": read_sigma_population,
        "gradient": make_read_sigma_gradient(learning_rate),
        "rprop": read_sigma_rprop,
        "lbfgs": read_sigma_lbfgs,
        "none": read_sigma_absent,
    }

    if spec.writer == "distribution" and ravel_fn is None:
        raise ValueError(
            "a 'distribution' writer needs ravel_fn: evosax stores "
            "state.mean raveled, so the bundle's pytree x must be "
            "flattened with the algorithm's own _ravel_solution"
        )

    writers: dict[str, Callable] = {
        "distribution": (
            make_write_distribution(ravel_fn)
            if ravel_fn is not None
            else write_none
        ),
        "population": make_write_population(repair_baseline=False),
        "mr15ga": make_write_population(repair_baseline=True),
        "gradient": make_write_gradient(learning_rate),
        "rprop": make_write_rprop(
            hyperparameters.get("min_step_size", 1e-6),
            hyperparameters.get("max_step_size", 50.0),
        ),
        "none": write_none,
    }

    return readers[spec.reader], writers[spec.writer]


# =============================================================================
