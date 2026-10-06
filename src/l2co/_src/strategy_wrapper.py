"""The L2CO strategy as a duck-typed evosax population strategy.

Exposes a trained :class:`~l2co._src.models.l2co_model.L2COModel`
(the stage-0 + stage-1 pair) as an evosax-style **population-based
algorithm**: ``init`` / ``ask`` / ``tell`` plus ``default_params``.
From the outer loop's perspective the call pattern follows the
classical ask-eval-tell rhythm::

    es = l2co_population_strategy(model, task, sampler, t, budget)
    params = es.default_params

    population = sampler(key, initial_params, es.population_size)
    state = es.init(key, population, fitness, params)

    for _ in range(budget):
        key, key_ask, key_tell = jr.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        fitness, grads = jax.vmap(jax.value_and_grad(loss_fn))(population)
        state, metrics = es.tell(
            key_tell, population, fitness, state, params, grads=grads,
        )

**Single-fire semantics** (see ``docs/adr/0002``). The L2CO strategy
fires exactly twice, and only the second fire happens inside the
ask/tell loop:

* The **stage-0 fire** happens at *construction time* inside
  :func:`l2co_population_strategy` — its inputs (task features) exist
  before the run, and firing eagerly keeps ``ScikitStage0Model``
  usable, since nothing sklearn ever enters a jitted region. The
  chosen optimizer starts as the active sub-optimizer.
* The **stage-1 fire** happens in-graph, in the first ``ask`` after
  ``switch_generation`` tells have completed
  (``switch_generation = round(t × max_iterations)``). It feeds the
  recorded per-generation ``(mean, min)`` loss trajectory through
  :func:`~l2co._src.tree_utils.traj_and_stair` and the stage-1
  network, and argmaxes the logits into the menu. The fire happens at
  most once (``state.fired``).

**Handshake** (``reset_opt_state`` + ``best_one``, reproducing the
former ``reset_from_stage0`` path). When the stage-1 fire selects a
*different* optimizer, the incoming sub-optimizer's internal state is
re-initialised around ``state.best_solution`` and the emitted
population for that one generation is ``[best_solution,
*sampler_draws, NaN, …]`` instead of the sub-optimizer's own ``ask``.
When it re-selects the stage-0 optimizer, nothing is reset and the
run simply continues.

The unconditional reset is the **default** (``opt_state_handshake=
"legacy_reset"``) because it is what the published L2CO results and
``docs/adr/0004``'s feval accounting were produced with. Passing
``opt_state_handshake="per_optimizer"`` instead honours each
optimizer's shared ``HANDSHAKE_POLICY`` entry, so only ``lbfgs`` is
reset and adaptive ES state survives the switch -- the same behaviour
the rl2co environment and (since ``docs/adr/0009``) rl2co's
``rl2co_population_strategy`` apply.

**Population size.** ``population_size = max(sub.popsize)`` over the
menu. When the active sub-optimizer has fewer candidates, slots
``[sub.popsize:]`` are ``NaN`` (population and fitness alike);
callers are responsible for masking them (or relying on NaN-robust
losses) — see the identical convention in rl2co's evosax wrapper.

**Deviations from strict evosax** (shared with rl2co's wrapper):

* ``tell`` accepts ``grads=`` — required when any gradient-based
  sub-optimizer is in the menu, ignored for population-based ones.
  The grad branch reads only ``grads[0]`` (the gradient at the
  current iterate, ``state.population[0]``).
* ``tell`` accepts ``value_fn=`` — a callable ``params -> scalar
  loss`` that line-search transforms (``optax.lbfgs``) probe at trial
  points; plain transforms ignore it.
* ``init`` accepts an all-``NaN`` ``fitness``: the bridge in
  :mod:`l2co._src.optimizers.update_class` does not evaluate the
  initial population (matching the stock l2co machinery, where the
  initial samples are only evaluated by ``reset``'s hidden
  bookkeeping pass), so best-tracking seeds to ``+inf`` and the
  sub-optimizers initialise around ``population[0]`` — exactly the
  point the old path initialised from (``samples[0]``). Standalone
  users may pass real fitness, in which case the best candidate
  seeds best-tracking and the sub-optimizer init point.

**Rhythm note.** Under the bridge, one ``tell`` = one generation =
one (external) ``OptHistory`` entry, in the ask-eval-tell order; the
stock l2co factories instead evaluate the *carried* population before
asking. Consequence: the initial sampled population is never a
counted entry here (the stock path evaluates it at scan step 0 and
then drops that entry), and the sub-optimizers receive one fewer
``tell`` than in the old path at equal generation count. The stage-1
fire therefore sees ``switch_generation`` post-update entries — the
same trajectory *semantics* the stage-1 model was trained on.
"""

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from l2co_optimizers import (
    OptimizationStep,
    SamplerFunction,
    SubOpt,
    optstep_to_subopt,
)
from l2co_tasks import Task

# Local
from l2co._src.models.l2co_model import L2COModel
from l2co._src.models.stage1 import Stage1Model
from l2co._src.tree_utils import traj_and_stair

# =============================================================================

__all__ = ["Params", "State", "l2co_population_strategy"]


#                                                             Params and State
# =============================================================================


class Params(eqx.Module):
    """Per-generation hyperparameter container.

    Empty: sub-optimizer hyperparameters are captured at construction
    time inside the :class:`SubOpt` closures and never reach this
    surface. Kept as an explicit class for evosax-API parity and to
    leave room for future per-generation overrides.
    """


class State(eqx.Module):
    """Wrapper state threaded across the ask/tell loop.

    Attributes
    ----------
    population : PyTree
        Last-known population, leading axis ``population_size``.
        Pop-based ``tell``s store the just-evaluated population
        verbatim; grad-based ``tell``s store ``[post_step_point,
        NaN, …]``.
    fitness : Float[Array, " population_size"]
        Fitness corresponding to :attr:`population`; all-``NaN``
        after a grad-based ``tell`` (the post-step point is not
        re-evaluated) and possibly at ``init`` (see the module
        docstring).
    sub_states : tuple[Any, ...]
        One entry per menu sub-optimizer; heterogeneous pytrees.
        All initialised at :meth:`~_L2COPopulationStrategy.init`;
        the sub-optimizer chosen by a *switching* stage-1 fire is
        re-initialised around :attr:`best_solution` at the fire
        (``reset_opt_state`` handshake).
    active_opt_idx : Int[Array, ""]
        Menu index of the currently-running sub-optimizer. Starts at
        the stage-0 choice and changes at most once (the stage-1
        fire).
    generation_counter : Int[Array, ""]
        Number of completed ``tell``s. The stage-1 fire happens in
        the first ``ask`` with ``generation_counter >=
        switch_generation``.
    fired : Bool[Array, ""]
        Whether the stage-1 fire has happened. Makes the fire
        idempotent under repeated ``ask``s.
    traj_mean : Float[Array, " switch_generation"]
        Per-generation mean loss over the active sub-optimizer's
        (unpadded) candidates, for generations ``1 ..
        switch_generation`` — the ``x_output_mean`` input of the
        stage-1 fire. NaN-initialised; frozen once full.
    traj_min : Float[Array, " switch_generation"]
        Per-generation minimum loss; the ``x_output_min`` input of
        the stage-1 fire.
    best_solution : PyTree
        Best parameters seen so far (single-solution pytree). Slot 0
        of the ``best_one`` handshake population.
    best_fitness : Float[Array, ""]
        Lowest fitness seen so far; ``+inf`` until a finite fitness
        is observed.
    """

    population: PyTree
    fitness: Float[Array, " population_size"]
    sub_states: tuple[Any, ...]
    active_opt_idx: Int[Array, ""]
    generation_counter: Int[Array, ""]
    fired: Bool[Array, ""]
    traj_mean: Float[Array, " switch_generation"]
    traj_min: Float[Array, " switch_generation"]
    best_solution: PyTree
    best_fitness: Float[Array, ""]


def _nan_safe_argmin(fitness: Float[Array, " n"]) -> Int[Array, ""]:
    """Index of the smallest non-NaN entry (slot 0 if all-NaN).

    Parameters
    ----------
    fitness : Float[Array, " n"]
        Fitness vector, possibly containing ``NaN``.

    Returns
    -------
    Int[Array, ""]
        Argmin over the NaN-masked vector.
    """
    safe = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
    return jnp.argmin(safe)


#                                                              Wrapper factory
# =============================================================================


def l2co_population_strategy(
    model: L2COModel,
    task: Task,
    sampler: SamplerFunction,
    t: float,
    max_iterations: int,
    opt_state_handshake: str = "legacy_reset",
) -> _L2COPopulationStrategy:
    """Wrap a trained L2CO model as an evosax-style strategy.

    Performs the stage-0 fire eagerly (so sklearn stage-0 models
    never enter a jitted region), resolves the menu against the task,
    and validates that the stage-1 fire is traceable — all failures
    surface here, at construction, rather than mid-run.

    Parameters
    ----------
    model : L2COModel
        Trained two-stage model. The menu is
        ``model.stage1_model.optimizers``; the stage-0 prediction
        must be a member of it (by ``OptimizationStep`` equality).
        The stage-1 model's ``__call__`` must be traceable
        (``EquinoxStage1Model``) — validated via
        :func:`jax.eval_shape`.
    task : Task
        The task being optimised. Used for the stage-0/stage-1
        feature vectors, the sub-optimizer ``solution=`` templates
        and popsize resolution, and ``task.global_min``.
        ``task.global_min`` must be declared: the stage-1 staircase
        features were trained against the true optimum, and a
        running-min fallback would be silently out-of-distribution
        (see ``docs/adr/0002``).
    sampler : SamplerFunction
        ``l2co.sampling`` function used to fill the post-fire
        handshake population beyond slot 0. Called as
        ``sampler(key, best_solution, popsize - 1)``.
    t : float
        Stage-0 fraction of the budget. The switch generation is
        ``round(t * max_iterations)``, clamped to
        ``[0, max_iterations]``; ``t == 1`` means the fire happens
        only if the caller runs past ``max_iterations`` generations.
        A resolved switch generation of ``0`` is rejected — the
        stage-1 model cannot fire on an empty trajectory.
    max_iterations : int
        The iteration budget the switch generation is computed
        against. Also sizes the trajectory buffers.
    opt_state_handshake : str, optional
        Which ``opt_state`` handshake the stage-1 switch applies.
        ``"legacy_reset"`` (default) re-initialises *every* incoming
        sub-optimizer around the best-so-far, reproducing the published
        semantics of the former ``reset_from_stage0``, which
        ``docs/adr/0004``'s feval accounting is written against.
        ``"per_optimizer"`` instead honours each optimizer's
        :data:`~l2co._src.optimizers.handshake_policy.HANDSHAKE_POLICY`
        entry, so only optimizers that genuinely cannot survive an
        iterate jump (``lbfgs``) are reset and adaptive ES state is
        preserved -- matching what the rl2co environment does. Opt-in
        because it changes L2CO results; see ``docs/adr/0009``.

    Returns
    -------
    _L2COPopulationStrategy
        Strategy ready for an ask/eval/tell loop (or the
        ``UpdateClass`` bridge in
        :mod:`l2co._src.optimizers.update_class`).

    Raises
    ------
    ValueError
        If the menu is empty, the stage-0 prediction is not in the
        menu, ``task.global_min`` is ``None``, the resolved switch
        generation is ``0``, the stage-1 model is not traceable, its
        logit head does not match the menu size, or
        ``opt_state_handshake`` is not one of ``"legacy_reset"`` /
        ``"per_optimizer"``.
    """
    if opt_state_handshake not in ("legacy_reset", "per_optimizer"):
        raise ValueError(
            f"opt_state_handshake must be 'legacy_reset' or "
            f"'per_optimizer'; got {opt_state_handshake!r}"
        )

    menu: tuple[OptimizationStep, ...] = tuple(model.stage1_model.optimizers)
    if not menu:
        raise ValueError("model.stage1_model.optimizers must be non-empty")

    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive; {max_iterations=}")

    switch_generation = int(round(t * max_iterations))
    switch_generation = max(min(switch_generation, max_iterations), 0)
    if switch_generation == 0:
        raise ValueError(
            f"switch generation resolved to 0 ({t=}, {max_iterations=}): "
            f"the stage-1 model cannot fire on an empty trajectory. Use "
            f"t large enough that round(t * max_iterations) >= 1."
        )

    if task.global_min is None:
        raise ValueError(
            "task.global_min is required: the stage-1 staircase features "
            "are normalised against the true optimum (a running-min "
            "fallback would be out-of-distribution). Declare global_min "
            "on the task or use a plain optimizer."
        )
    global_min_const = jnp.asarray(task.global_min, dtype=float)

    # Stage-0 fire — eager, at construction (sklearn-safe).
    stage0_step: OptimizationStep = model.predict_stage0(
        task_features=task.tag
    )
    try:
        stage0_idx = menu.index(stage0_step)
    except ValueError:
        raise ValueError(
            f"stage-0 predicted {stage0_step.name!r}, which is not in the "
            f"stage-1 menu {[step.name for step in menu]}. The two stages "
            f"must share one menu (use L2COModel.swap_optimizers)."
        ) from None

    sub_optimizers = tuple(
        optstep_to_subopt(
            step,
            model=task.model,
            loss_fn=task.loss_fn,
            pass_rng=task.pass_rng,
        )
        for step in menu
    )
    population_size = max(sub.popsize for sub in sub_optimizers)
    name_hashes_arr = jnp.array(
        [sub.name_hash for sub in sub_optimizers], dtype=int
    )

    # Stage-1 feature vectors — both static at construction: the task
    # features by definition, the optimizer features because the
    # stage-0 choice (the only optimizer stage-1 ever sees) is fixed.
    stage1 = model.stage1_model
    task_features = stage1.feature_processor_task.transform(task.tag)
    optimizer_features = stage1.feature_processor_optimizer.transform(
        {"optimizer": stage0_step.name}
    )

    # Validate the in-graph stage-1 fire once, abstractly: catches a
    # non-traceable stage-1 model (anything but an Equinox one) and a
    # logit head that does not match the menu, without spending a
    # forward pass.
    def _abstract_fire(
        mean: Float[Array, " switch_generation"],
        min_: Float[Array, " switch_generation"],
    ) -> jax.Array:
        x_history = traj_and_stair(mean, min_, global_min_const)
        return stage1((task_features, optimizer_features, x_history))

    traj_struct = jax.ShapeDtypeStruct(
        (switch_generation,), jnp.result_type(float)
    )
    try:
        logits_struct = jax.eval_shape(
            _abstract_fire, traj_struct, traj_struct
        )
    except Exception as e:
        raise ValueError(
            f"the stage-1 model is not traceable and cannot fire inside "
            f"the jitted wrapper (got {type(stage1).__name__}; use an "
            f"EquinoxStage1Model): {e}"
        ) from e
    if logits_struct.shape[-1] != len(menu):
        raise ValueError(
            f"stage-1 logit head size {logits_struct.shape[-1]} does not "
            f"match the menu size {len(menu)}"
        )

    return _L2COPopulationStrategy(
        stage1_model=stage1,
        sub_optimizers=sub_optimizers,
        name_hashes_arr=name_hashes_arr,
        task_features=task_features,
        optimizer_features=optimizer_features,
        global_min_const=global_min_const,
        stage0_idx=stage0_idx,
        switch_generation=switch_generation,
        population_size=population_size,
        sampler=sampler,
        opt_state_handshake=opt_state_handshake,
    )


#                                                               Strategy class
# =============================================================================


class _L2COPopulationStrategy(eqx.Module):
    """Trained L2CO strategy as a duck-typed evosax strategy.

    Instances are built by :func:`l2co_population_strategy` — the
    constructor takes the already-resolved fields directly.

    Attributes
    ----------
    stage1_model : Stage1Model
        Trained stage-1 network; fires in-graph at the switch
        generation.
    sub_optimizers : tuple[SubOpt, ...]
        Static adapters for the menu (order = logit order).
    name_hashes_arr : Int[Array, " n_opt"]
        Per-menu-entry hashes, stamped into the external history so
        each entry records which optimizer produced it.
    task_features : Float[Array, " task_dim"]
        Precomputed stage-1 task feature vector.
    optimizer_features : Float[Array, " optimizer_dim"]
        Precomputed stage-1 optimizer feature vector for the stage-0
        choice (the only optimizer stage-1 ever observes).
    global_min_const : Float[Array, ""]
        ``task.global_min``; required, see the factory.
    stage0_idx : int
        Static. Menu index of the stage-0 fire's choice; the initial
        ``active_opt_idx``, and the reference against which the
        stage-1 fire's choice is compared to detect a switch.
    switch_generation : int
        Static. Number of completed generations after which the
        stage-1 fire happens; also the trajectory buffer length.
    population_size : int
        Static. ``max(sub.popsize for sub in sub_optimizers)``.
    sampler : SamplerFunction
        Static. Fills the post-fire handshake population beyond
        slot 0.
    opt_state_handshake : str
        Static. ``"legacy_reset"`` (default) resets every incoming
        sub-optimizer on a switching fire; ``"per_optimizer"`` resets
        only those whose ``SubOpt.opt_state_handshake`` is
        ``"reset"``. See :meth:`_reset_active_substate` and
        ``docs/adr/0009``.
    """

    stage1_model: Stage1Model
    sub_optimizers: tuple[SubOpt, ...]
    name_hashes_arr: Int[Array, " n_opt"]
    task_features: Float[Array, " task_dim"]
    optimizer_features: Float[Array, " optimizer_dim"]
    global_min_const: Float[Array, ""]
    stage0_idx: int = eqx.field(static=True)
    switch_generation: int = eqx.field(static=True)
    population_size: int = eqx.field(static=True)
    sampler: SamplerFunction = eqx.field(static=True)
    opt_state_handshake: str = eqx.field(static=True, default="legacy_reset")

    @property
    def n_opt(self) -> int:
        """Number of menu sub-optimizers (``len(self.sub_optimizers)``)."""
        return len(self.sub_optimizers)

    @property
    def default_params(self) -> Params:
        """Return the empty per-generation :class:`Params`.

        Returns
        -------
        Params
            Empty params instance; pass straight through ``init``,
            ``ask``, and ``tell``.
        """
        return Params()

    # ---------- init ----------------------------------------------------

    def init(
        self,
        key: PRNGKeyArray,
        population: PyTree,
        fitness: Float[Array, " population_size"],
        params: Params,
    ) -> State:
        """Build the initial :class:`State`.

        Initialises every menu sub-optimizer's internal state at the
        best candidate of ``(population, fitness)`` — with an
        all-``NaN`` ``fitness`` that degenerates to
        ``population[0]``, the old path's init point (see the module
        docstring). The stage-0 choice becomes the active
        sub-optimizer; the trajectory buffers start empty.

        Parameters
        ----------
        key : PRNGKeyArray
            PRNG key for sub-optimizer initialisation.
        population : PyTree
            Initial population, leading axis ``population_size``.
        fitness : Float[Array, " population_size"]
            Initial fitness; ``NaN`` entries mark unevaluated or
            padded slots (an all-``NaN`` vector is valid).
        params : Params
            Per-generation params (empty; evosax-API parity).

        Returns
        -------
        State
            Initial state ready for the first :meth:`ask`.

        Raises
        ------
        ValueError
            If ``population`` (any leaf) or ``fitness`` does not
            have leading axis :attr:`population_size`.
        """
        fitness = jnp.asarray(fitness, dtype=float)
        pop_leaves = jax.tree.leaves(population)
        if fitness.shape[0] != self.population_size or any(
            leaf.shape[0] != self.population_size for leaf in pop_leaves
        ):
            raise ValueError(
                f"init expects population and fitness with leading axis "
                f"population_size={self.population_size} (= max menu "
                f"popsize); got fitness leading axis {fitness.shape[0]} "
                f"and population leading axes "
                f"{sorted({leaf.shape[0] for leaf in pop_leaves})}."
            )

        best_idx = _nan_safe_argmin(fitness)
        best_candidate = jax.tree.map(lambda x: x[best_idx], population)
        safe_fit = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
        best_fitness = safe_fit[best_idx]

        init_keys = jr.split(key, self.n_opt)
        sub_states: tuple[Any, ...] = tuple(
            sub.init_fn(best_candidate, k)
            for sub, k in zip(self.sub_optimizers, init_keys, strict=True)
        )

        return State(
            population=population,
            fitness=fitness,
            sub_states=sub_states,
            active_opt_idx=jnp.array(self.stage0_idx, dtype=int),
            generation_counter=jnp.array(0, dtype=int),
            fired=jnp.array(False),
            traj_mean=jnp.full((self.switch_generation,), jnp.nan),
            traj_min=jnp.full((self.switch_generation,), jnp.nan),
            best_solution=best_candidate,
            best_fitness=best_fitness,
        )

    # ---------- helpers -------------------------------------------------

    def _fire_stage1(self, state: State) -> State:
        """Run the stage-1 fire and load the new ``active_opt_idx``.

        Deterministic (argmax over logits) — no key needed. The
        trajectory buffers are full by construction when this runs
        (``generation_counter >= switch_generation``).

        Parameters
        ----------
        state : State
            Pre-fire state (``fired`` is ``False``).

        Returns
        -------
        State
            State with the stage-1 choice active and ``fired`` set.
        """
        x_history = traj_and_stair(
            state.traj_mean, state.traj_min, self.global_min_const
        )
        logits = self.stage1_model(
            (self.task_features, self.optimizer_features, x_history)
        )
        new_idx = jnp.argmax(logits).astype(int)
        return eqx.tree_at(
            lambda s: (s.active_opt_idx, s.fired),
            state,
            (new_idx, jnp.array(True)),
        )

    def _reset_active_substate(self, state: State, key: PRNGKeyArray) -> State:
        """Re-initialise the active sub-optimizer around the best.

        The ``reset_opt_state`` half of the switching handshake:
        dispatches on ``state.active_opt_idx`` (the fresh stage-1
        choice) and replaces that sub-optimizer's internal state with
        ``sub.init_fn(state.best_solution, key)``, reproducing the
        former ``reset_from_stage0`` semantics (distribution-based ES
        re-centred on the best-so-far; population-based ES adopts the
        evaluated handshake population on the following ``tell``).

        Parameters
        ----------
        state : State
            Post-fire state.
        key : PRNGKeyArray
            Sub-optimizer initialisation key.

        Under ``opt_state_handshake="per_optimizer"`` only
        sub-optimizers whose
        :attr:`~l2co._src.optimizers.sub_optimizer.SubOpt.
        opt_state_handshake` is ``"reset"`` are re-initialised; the
        others keep their adapted state, matching the rl2co
        environment. The flag is static, so the untouched slots become
        identity branches at trace time (``docs/adr/0009``).

        Returns
        -------
        State
            State with the incoming sub-optimizer's state reset, or
            unchanged when its policy says continue.
        """
        per_optimizer = self.opt_state_handshake == "per_optimizer"

        def make_branch(i: int):
            sub_i = self.sub_optimizers[i]

            if per_optimizer and sub_i.opt_state_handshake != "reset":
                return lambda state: state

            def branch(state: State) -> State:
                new_sub_state = sub_i.init_fn(state.best_solution, key)
                new_sub_states = eqx.tree_at(
                    lambda t: t[i], state.sub_states, new_sub_state
                )
                return eqx.tree_at(
                    lambda s: s.sub_states, state, new_sub_states
                )

            return branch

        if per_optimizer and all(
            sub.opt_state_handshake != "reset" for sub in self.sub_optimizers
        ):
            return state

        branches = tuple(make_branch(i) for i in range(self.n_opt))
        return jax.lax.switch(state.active_opt_idx, branches, state)

    def _current_point(self, state: State) -> PyTree:
        """Pick the current iterate: ``state.population[0]``.

        The point a grad sub-optimizer steps from. After a grad
        ``tell`` slot 0 holds the previous post-step point, so
        consecutive grad steps chain; after a pop ``tell`` slot 0 is
        the last generation's first candidate.

        Parameters
        ----------
        state : State
            Current state.

        Returns
        -------
        PyTree
            Single-solution pytree (no leading axis).
        """
        return jax.tree.map(lambda x: x[0], state.population)

    def _push_trajectory_entry(
        self,
        state: State,
        fitness: Float[Array, " population_size"],
    ) -> State:
        """Record one generation's ``(mean, min)`` loss.

        Aggregates ``fitness[: sub.popsize]`` of the active
        sub-optimizer (static slice via :func:`jax.lax.switch`) and
        writes it at index ``generation_counter``. Writes are guarded
        to ``generation_counter < switch_generation`` — after the
        stage-1 fire the buffers are frozen (their only consumer has
        run).

        Parameters
        ----------
        state : State
            Pre-increment state (``generation_counter`` still names
            this generation's slot).
        fitness : Float[Array, " population_size"]
            Per-candidate losses; slots ``[sub.popsize:]`` NaN.

        Returns
        -------
        State
            State with the trajectory buffers updated.
        """

        def make_branch(i: int):
            popsize_i = self.sub_optimizers[i].popsize

            def branch(
                fitness: Float[Array, " population_size"],
            ) -> tuple[Float[Array, ""], Float[Array, ""]]:
                fit_slice = fitness[:popsize_i]
                return jnp.nanmean(fit_slice), jnp.nanmin(fit_slice)

            return branch

        branches = tuple(make_branch(i) for i in range(self.n_opt))
        mean_loss, min_loss = jax.lax.switch(
            state.active_opt_idx, branches, fitness
        )

        idx = state.generation_counter
        write = idx < self.switch_generation
        safe_idx = jnp.minimum(idx, self.switch_generation - 1)
        new_mean = state.traj_mean.at[safe_idx].set(
            jnp.where(write, mean_loss, state.traj_mean[safe_idx])
        )
        new_min = state.traj_min.at[safe_idx].set(
            jnp.where(write, min_loss, state.traj_min[safe_idx])
        )
        return eqx.tree_at(
            lambda s: (s.traj_mean, s.traj_min), state, (new_mean, new_min)
        )

    def _update_best(
        self,
        state: State,
        population: PyTree,
        fitness: Float[Array, " population_size"],
    ) -> State:
        """Update ``best_solution`` / ``best_fitness`` from this round.

        NaN-safe: padded fitness slots are remapped to ``+inf``
        before the argmin so they cannot win.

        Parameters
        ----------
        state : State
            Current state.
        population : PyTree
            Candidates from the last :meth:`ask`.
        fitness : Float[Array, " population_size"]
            Per-candidate losses.

        Returns
        -------
        State
            State with best-tracking possibly replaced.
        """
        safe = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
        idx = jnp.argmin(safe)
        candidate_fit = safe[idx]
        candidate_sol = jax.tree.map(lambda x: x[idx], population)
        update = candidate_fit < state.best_fitness
        new_best_fit = jnp.where(update, candidate_fit, state.best_fitness)
        new_best_sol = jax.tree.map(
            lambda new, old: jnp.where(update, new, old),
            candidate_sol,
            state.best_solution,
        )
        return eqx.tree_at(
            lambda s: (s.best_solution, s.best_fitness),
            state,
            (new_best_sol, new_best_fit),
        )

    def _ask_dispatch(
        self,
        state: State,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, State]:
        """Dispatch one ``ask`` of the active sub-optimizer.

        Each :func:`jax.lax.switch` branch returns a population with
        leading axis ``population_size``, NaN-padded for slots
        ``[sub.popsize:]``. The grad branch returns
        ``[current_point, NaN, …]`` (the grad step happens in
        :meth:`tell` once ``grads`` arrives); pop branches let the ES
        propose from its own state.

        Parameters
        ----------
        state : State
            Current state.
        key : PRNGKeyArray
            Forwarded to the active sub-optimizer's ask.

        Returns
        -------
        tuple[PyTree, State]
            ``(population, new_state)``.
        """

        def _pad_with_nan(arr: Array, sub_popsize: int) -> Array:
            """Pad ``arr`` on axis 0 to ``population_size`` with NaN."""
            pad = self.population_size - sub_popsize
            if pad == 0:
                return arr
            pad_width = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
            return jnp.pad(
                arr.astype(float),
                pad_width,
                mode="constant",
                constant_values=jnp.nan,
            )

        def make_branch(i: int):
            sub_i = self.sub_optimizers[i]
            popsize_i = sub_i.popsize

            if sub_i.is_pop:

                def branch(state: State) -> tuple[PyTree, State]:
                    new_pop, new_sub_state = sub_i.step_fn(
                        None, state.sub_states[i], key
                    )
                    padded = jax.tree.map(
                        lambda x: _pad_with_nan(x, popsize_i), new_pop
                    )
                    new_sub_states = eqx.tree_at(
                        lambda t: t[i], state.sub_states, new_sub_state
                    )
                    return padded, eqx.tree_at(
                        lambda s: s.sub_states, state, new_sub_states
                    )

                return branch

            def branch(state: State) -> tuple[PyTree, State]:
                point = self._current_point(state)
                population = jax.tree.map(
                    lambda m: (
                        jnp.full(
                            (self.population_size, *m.shape),
                            jnp.nan,
                            dtype=float,
                        )
                        .at[0]
                        .set(m.astype(float))
                    ),
                    point,
                )
                return population, state

            return branch

        branches = tuple(make_branch(i) for i in range(self.n_opt))
        return jax.lax.switch(state.active_opt_idx, branches, state)

    def _handshake_population(self, state: State, key: PRNGKeyArray) -> PyTree:
        """Build the post-fire handshake population (``best_one``).

        Emitted for the one generation immediately after a
        *switching* stage-1 fire, replacing the incoming
        sub-optimizer's own ``ask``: slot 0 is
        ``state.best_solution``, slots ``[1: popsize]`` are draws
        from :attr:`sampler` shaped against it, slots
        ``[popsize:]`` are NaN. The evaluated result reaches the
        (freshly reset) sub-optimizer through the following
        :meth:`tell`.

        Parameters
        ----------
        state : State
            Post-fire state (``active_opt_idx`` already updated).
        key : PRNGKeyArray
            Key for :attr:`sampler`.

        Returns
        -------
        PyTree
            Handshake population, leading axis ``population_size``.
        """

        def make_branch(i: int):
            popsize_i = self.sub_optimizers[i].popsize

            def branch(state: State) -> PyTree:
                best = state.best_solution
                population = jax.tree.map(
                    lambda b: (
                        jnp.full(
                            (self.population_size, *b.shape),
                            jnp.nan,
                            dtype=float,
                        )
                        .at[0]
                        .set(b.astype(float))
                    ),
                    best,
                )
                if popsize_i > 1:
                    samples = self.sampler(key, best, popsize_i - 1)
                    population = jax.tree.map(
                        lambda p, s: p.at[1:popsize_i].set(s.astype(float)),
                        population,
                        samples,
                    )
                return population

            return branch

        branches = tuple(make_branch(i) for i in range(self.n_opt))
        return jax.lax.switch(state.active_opt_idx, branches, state)

    def _tell_dispatch(
        self,
        state: State,
        population: PyTree,
        fitness: Float[Array, " population_size"],
        grads: PyTree | None,
        value_fn: Callable | None,
        key: PRNGKeyArray,
    ) -> State:
        """Dispatch one ``tell`` (or grad step) for the active sub-opt.

        Pop branch: slice to ``[: sub.popsize]``, forward to
        ``sub.tell_fn``, store the user-supplied snapshot verbatim.
        Grad branch: read ``grads[0]`` at the iterate
        ``population[0]``, apply the optax update, store the
        post-step point at slot 0 with all-NaN fitness. ``value_fn``
        (with ``value`` = ``fitness[0]``, ``grad`` = ``grads[0]``) is
        forwarded to line-search transforms when supplied.

        Parameters
        ----------
        state : State
            Current state.
        population : PyTree
            Candidates from the matching :meth:`ask`.
        fitness : Float[Array, " population_size"]
            Per-candidate losses.
        grads : PyTree | None
            Per-candidate gradients; required for grad
            sub-optimizers.
        value_fn : Callable | None
            Line-search probe callable (``optax.lbfgs``); ``None``
            reproduces the stock single-argument ``update``.
        key : PRNGKeyArray
            Forwarded to the sub-optimizer.

        Returns
        -------
        State
            State with the active sub-optimizer advanced and
            ``population``/``fitness`` updated.
        """

        def make_branch(i: int):
            sub_i = self.sub_optimizers[i]
            popsize_i = sub_i.popsize

            if sub_i.is_pop:

                def branch(state: State) -> State:
                    pop_slice = jax.tree.map(
                        lambda x: x[:popsize_i], population
                    )
                    fit_slice = fitness[:popsize_i]
                    new_sub_state = sub_i.tell_fn(
                        state.sub_states[i], pop_slice, fit_slice, key
                    )
                    new_sub_states = eqx.tree_at(
                        lambda t: t[i], state.sub_states, new_sub_state
                    )
                    return eqx.tree_at(
                        lambda s: (s.sub_states, s.population, s.fitness),
                        state,
                        (new_sub_states, population, fitness),
                    )

                return branch

            def branch(state: State) -> State:
                if grads is None:
                    raise ValueError(
                        f"Grad sub-optimizer {i} requires ``grads`` to be "
                        f"passed to ``tell``; got None."
                    )
                pre_step = jax.tree.map(lambda x: x[0], population)
                g0 = jax.tree.map(lambda x: x[0], grads)
                extra_args = (
                    {}
                    if value_fn is None
                    else {
                        "value": fitness[0],
                        "grad": g0,
                        "value_fn": value_fn,
                    }
                )
                updates, new_sub_state = sub_i.step_fn(
                    g0, pre_step, state.sub_states[i], key, **extra_args
                )
                new_point = optax.apply_updates(pre_step, updates)
                new_population = jax.tree.map(
                    lambda x: (
                        jnp.full(
                            (self.population_size, *x.shape),
                            jnp.nan,
                            dtype=float,
                        )
                        .at[0]
                        .set(x.astype(float))
                    ),
                    new_point,
                )
                new_fitness = jnp.full(
                    (self.population_size,), jnp.nan, dtype=float
                )
                new_sub_states = eqx.tree_at(
                    lambda t: t[i], state.sub_states, new_sub_state
                )
                return eqx.tree_at(
                    lambda s: (s.sub_states, s.population, s.fitness),
                    state,
                    (new_sub_states, new_population, new_fitness),
                )

            return branch

        branches = tuple(make_branch(i) for i in range(self.n_opt))
        return jax.lax.switch(state.active_opt_idx, branches, state)

    # ---------- ask -----------------------------------------------------

    def ask(
        self,
        key: PRNGKeyArray,
        state: State,
        params: Params,
    ) -> tuple[PyTree, State]:
        """Sample one generation's population.

        Fires stage-1 exactly once, in the first call with
        ``generation_counter >= switch_generation``. On a switching
        fire the incoming sub-optimizer's state is reset around the
        best-so-far and the emitted population is the ``best_one``
        handshake; a fire that re-selects the stage-0 optimizer, and
        every other generation, dispatches the active sub-optimizer's
        own ``ask``.

        Parameters
        ----------
        key : PRNGKeyArray
            Per-call PRNG key, split into a reset key and an ask
            key.
        state : State
            Previous state.
        params : Params
            Per-generation params (empty).

        Returns
        -------
        tuple[PyTree, State]
            ``(population, new_state)`` with ``population`` leading
            axis ``population_size``, NaN-padded past the active
            sub-optimizer's popsize.
        """
        key_reset, key_ask = jr.split(key)

        do_fire = jnp.logical_and(
            jnp.logical_not(state.fired),
            state.generation_counter >= self.switch_generation,
        )
        # Wrap the branch in a lambda: jax hashes branch callables, and
        # a *bound method* hashes through its instance — whose
        # ``MixedEmbedding.embedders`` list is unhashable. Lambdas hash
        # by identity.
        state = jax.lax.cond(
            do_fire, lambda s: self._fire_stage1(s), lambda s: s, state
        )
        # A handshake happens exactly when the fire switched away from
        # the stage-0 optimizer. ``active_opt_idx`` only ever changes
        # at the fire, so comparing against the static stage-0 index
        # is sufficient — and on non-fire generations ``do_fire``
        # gates the whole branch off.
        do_handshake = jnp.logical_and(
            do_fire, state.active_opt_idx != self.stage0_idx
        )
        state = jax.lax.cond(
            do_handshake,
            lambda s: self._reset_active_substate(s, key_reset),
            lambda s: s,
            state,
        )
        return jax.lax.cond(
            do_handshake,
            lambda s: (self._handshake_population(s, key_ask), s),
            lambda s: self._ask_dispatch(s, key_ask),
            state,
        )

    # ---------- tell ----------------------------------------------------

    def tell(
        self,
        key: PRNGKeyArray,
        population: PyTree,
        fitness: Float[Array, " population_size"],
        state: State,
        params: Params,
        *,
        grads: PyTree | None = None,
        value_fn: Callable | None = None,
    ) -> tuple[State, dict]:
        """Consume one generation's fitness (and optional grads).

        Records the generation's ``(mean, min)`` loss into the
        trajectory buffers (until the stage-1 fire), updates
        best-tracking, dispatches the active sub-optimizer's ``tell``
        (pop) or grad step (grad), and advances the generation
        counter.

        Parameters
        ----------
        key : PRNGKeyArray
            Per-call PRNG key, forwarded to the sub-optimizer.
        population : PyTree
            Candidates returned by the matching :meth:`ask`.
        fitness : Float[Array, " population_size"]
            Per-candidate losses (to be minimised); slots
            ``[sub.popsize:]`` are expected NaN.
        state : State
            Post-``ask`` state.
        params : Params
            Per-generation params (empty).
        grads : PyTree | None
            Per-candidate gradients, leading axis
            ``population_size``. Required when the active
            sub-optimizer is gradient-based.
        value_fn : Callable | None
            Callable ``params -> scalar loss`` for line-search grad
            sub-optimizers (``optax.lbfgs``); see the module
            docstring.

        Returns
        -------
        tuple[State, dict]
            ``(new_state, metrics)`` with
            ``generation_counter``, ``best_fitness_in_generation``,
            ``best_fitness``, ``active_opt_idx``, and ``fired``.
        """
        state = self._push_trajectory_entry(state, fitness)
        state = self._update_best(state, population, fitness)
        state = self._tell_dispatch(
            state, population, fitness, grads, value_fn, key
        )
        state = eqx.tree_at(
            lambda s: s.generation_counter,
            state,
            state.generation_counter + 1,
        )

        safe = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
        metrics = {
            "generation_counter": state.generation_counter,
            "best_fitness_in_generation": jnp.min(safe),
            "best_fitness": state.best_fitness,
            "active_opt_idx": state.active_opt_idx,
            "fired": state.fired,
        }
        return state, metrics
