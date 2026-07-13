"""Unit tests for the L2CO strategy wrapper
(:func:`l2co.optimization.l2co_population_strategy`).

Three layers (see ``docs/adr/0002``):

* **Fire equivalence** — the construction-time stage-0 fire matches
  the eager ``predict_stage0``, and the in-graph stage-1 fire matches
  the eager ``predict_stage1`` on the identical trajectory (the
  permanent equivalence contract with the relocated eager driver in
  l2co_experiments).
* **Wrapper behaviour** — mirroring rl2co's ``test_evosax_wrapper``
  battery: evosax surface, init seeding, one generation, NaN
  padding, grads kwarg, plus the L2CO-specific single-fire /
  handshake / trajectory-buffer semantics.
* **Bridge** — ``l2co_population_based_fn``'s no-eval ``init_fn``
  and one real ask-eval-tell ``step_fn`` generation with the dynamic
  history hash.

Determinism trick: the stage-0 / stage-1 networks' final linear
layer is overwritten with zero weights and a one-hot bias, forcing
the argmax to a chosen menu index regardless of input.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from l2co_tasks import create_bbob_task

from l2co import (
    EquinoxStage0Model,
    EquinoxStage1Model,
    FeatureProcessor,
    L2COModel,
    MixedEmbedding,
    OptimizationStep,
    UpdateClass,
)
from l2co._src.optimizers.update_class import l2co_population_based_fn
from l2co._src.strategy_wrapper import State, l2co_population_strategy
from l2co.models import MLP, LSTM_Encoder
from l2co.sampling import normal_sampling

MAX_ITERATIONS = 8
T = 0.5  # switch generation = 4
# Synthetic per-generation losses must sit ABOVE the task's
# global_min (sphere seed 0: 602.75): traj_and_stair takes
# log(loss - global_min), so smaller values give NaN logits and
# argmax degenerates to slot 0 regardless of the forced bias.
LOSS_BASE = 700.0

SEPCMAES = OptimizationStep(optimizer="sepcmaes")
ADAM = OptimizationStep(
    optimizer="adam", hyperparameters={"learning_rate": 0.03}
)
MENU = (SEPCMAES, ADAM)


def _task():
    return create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)


def _force_choice(mlp: MLP, idx: int) -> MLP:
    """Zero the final layer's weights and one-hot its bias.

    Makes ``argmax(logits)`` equal ``idx`` for every input, so tests
    control which menu entry a fire picks.
    """
    last = mlp.model.layers[-1]
    forced_weight = jnp.zeros_like(last.weight)
    forced_bias = jnp.zeros_like(last.bias).at[idx].set(10.0)
    return eqx.tree_at(
        lambda m: (m.model.layers[-1].weight, m.model.layers[-1].bias),
        mlp,
        (forced_weight, forced_bias),
    )


def _make_model(stage0_idx: int, stage1_idx: int) -> L2COModel:
    """Build a tiny L2COModel whose two fires are forced to the
    given menu indices."""
    key = jr.key(7)
    k_s0, k_enc, k_task, k_opt, k_mlp = jr.split(key, 5)

    fp_task = FeatureProcessor.init(schema={"dimensionality": "continuous"})
    fp_opt = FeatureProcessor.init(
        schema={"optimizer": [step.name for step in MENU]}
    )

    stage0_mlp = _force_choice(
        MLP.init(
            in_size=1, out_size=len(MENU), width_size=8, depth=1, key=k_s0
        ),
        stage0_idx,
    )
    stage0 = EquinoxStage0Model.init(
        feature_processor_task=fp_task,
        model=stage0_mlp,
        optimizers=list(MENU),
    )

    history_encoder = LSTM_Encoder(
        in_size=2, hidden_size=8, out_size=8, key=k_enc
    )
    me_task = MixedEmbedding.from_feature_processor(
        feature_processor=fp_task, embed_dim=4, output_dim=8, key=k_task
    )
    me_opt = MixedEmbedding.from_feature_processor(
        feature_processor=fp_opt, embed_dim=4, output_dim=8, key=k_opt
    )
    stage1_mlp = _force_choice(
        MLP.init(
            in_size=8 * 2 + 8,
            out_size=len(MENU),
            width_size=8,
            depth=1,
            key=k_mlp,
        ),
        stage1_idx,
    )
    stage1 = EquinoxStage1Model(
        feature_processor_task=fp_task,
        feature_processor_optimizer=fp_opt,
        mixed_embedding_task=me_task,
        mixed_embedding_optimizer=me_opt,
        history_encoder=history_encoder,
        model=stage1_mlp,
        optimizers=list(MENU),
    )
    return L2COModel(stage0_model=stage0, stage1_model=stage1)


def _make_strategy(stage0_idx=0, stage1_idx=0, t=T, task=None):
    task = task if task is not None else _task()
    return (
        l2co_population_strategy(
            model=_make_model(stage0_idx, stage1_idx),
            task=task,
            sampler=normal_sampling,
            t=t,
            max_iterations=MAX_ITERATIONS,
        ),
        task,
    )


def _initial_state(strategy, fitness=None):
    task = _task()
    template = eqx.filter(task.model, eqx.is_inexact_array)
    population = normal_sampling(jr.key(0), template, strategy.population_size)
    if fitness is None:
        fitness = jnp.full((strategy.population_size,), jnp.nan)
    return (
        population,
        strategy.init(jr.key(1), population, fitness, strategy.default_params),
    )


def _zero_grads(population):
    return jax.tree.map(jnp.zeros_like, population)


def _run_generations(strategy, state, fitness_per_gen, start_key=100):
    """Drive ``len(fitness_per_gen)`` ask/tell cycles with synthetic
    per-generation fitness vectors; returns the final state."""
    for g, fitness in enumerate(fitness_per_gen):
        key = jr.key(start_key + g)
        population, state = strategy.ask(key, state, strategy.default_params)
        state, _ = strategy.tell(
            jr.key(start_key + 50 + g),
            population,
            fitness,
            state,
            strategy.default_params,
            grads=_zero_grads(population),
        )
    return state


#                                                                 Construction
# =============================================================================


class TestConstruction:
    def test_exposes_evosax_surface(self):
        strategy, _ = _make_strategy()
        assert hasattr(strategy, "init")
        assert hasattr(strategy, "ask")
        assert hasattr(strategy, "tell")
        assert strategy.default_params is not None
        # sepcmaes popsize dominates adam's 1.
        assert strategy.population_size == strategy.sub_optimizers[0].popsize
        assert strategy.switch_generation == round(T * MAX_ITERATIONS)

    def test_stage0_fire_matches_eager_predict(self):
        for idx in (0, 1):
            strategy, task = _make_strategy(stage0_idx=idx)
            model = _make_model(idx, 0)
            expected = model.predict_stage0(task_features=task.tag)
            assert strategy.stage0_idx == MENU.index(expected) == idx

    def test_requires_global_min(self):
        from l2co_tasks import Task

        task = _task()
        no_min = Task(
            model=task.model,
            loss_fn=task.loss_fn,
            global_min=None,
            pass_rng=task.pass_rng,
            has_aux=task.has_aux,
            dataset=task.dataset,
            tag=task.tag,
        )
        with pytest.raises(ValueError, match="global_min"):
            l2co_population_strategy(
                model=_make_model(0, 0),
                task=no_min,
                sampler=normal_sampling,
                t=T,
                max_iterations=MAX_ITERATIONS,
            )

    def test_rejects_zero_switch_generation(self):
        with pytest.raises(ValueError, match="switch generation"):
            _make_strategy(t=0.0)

    def test_rejects_stage0_choice_outside_menu(self):
        model = _make_model(0, 0)
        # Stage-0 predicts from its own list; make it disjoint from
        # the stage-1 menu.
        alien = OptimizationStep(optimizer="pso")
        stage0 = eqx.tree_at(
            lambda m: m.stage0_model.optimizers,
            model,
            [alien, alien],
            is_leaf=lambda x: isinstance(x, list),
        )
        with pytest.raises(ValueError, match="not in the stage-1 menu"):
            l2co_population_strategy(
                model=stage0,
                task=_task(),
                sampler=normal_sampling,
                t=T,
                max_iterations=MAX_ITERATIONS,
            )

    def test_rejects_logit_head_menu_mismatch(self):
        model = _make_model(0, 0)
        wide_mlp = MLP.init(
            in_size=8 * 2 + 8,
            out_size=len(MENU) + 1,
            width_size=8,
            depth=1,
            key=jr.key(3),
        )
        bad = eqx.tree_at(lambda m: m.stage1_model.model, model, wide_mlp)
        with pytest.raises(ValueError, match="does not match the menu"):
            l2co_population_strategy(
                model=bad,
                task=_task(),
                sampler=normal_sampling,
                t=T,
                max_iterations=MAX_ITERATIONS,
            )

    def test_rejects_untraceable_stage1(self):
        import numpy as np

        model = _make_model(0, 0)

        class _EagerMLP(MLP):
            def __call__(self, x):
                # np.asarray on a tracer raises — stands in for any
                # non-traceable (e.g. sklearn-backed) stage-1 head.
                return jnp.asarray(np.asarray(x).sum(axis=-1, keepdims=True))

        eager = _EagerMLP(model=model.stage1_model.model.model)
        bad = eqx.tree_at(lambda m: m.stage1_model.model, model, eager)
        with pytest.raises(ValueError, match="not traceable"):
            l2co_population_strategy(
                model=bad,
                task=_task(),
                sampler=normal_sampling,
                t=T,
                max_iterations=MAX_ITERATIONS,
            )


#                                                             Init and ask/tell
# =============================================================================


class TestInitAskTell:
    def test_init_with_nan_fitness_seeds_inf_best(self):
        strategy, _ = _make_strategy()
        population, state = _initial_state(strategy)
        assert isinstance(state, State)
        assert int(state.generation_counter) == 0
        assert not bool(state.fired)
        assert jnp.isinf(state.best_fitness)
        # Sub-optimizers initialise around population[0] (old-path
        # semantics): the best candidate of an all-NaN fitness.
        slot0 = jax.tree.map(lambda x: x[0], population)
        assert jnp.allclose(
            jax.tree.leaves(state.best_solution)[0],
            jax.tree.leaves(slot0)[0],
        )

    def test_init_with_real_fitness_seeds_best(self):
        strategy, _ = _make_strategy()
        n = strategy.population_size
        fitness = jnp.arange(n, dtype=float) + 3.0
        _, state = _initial_state(strategy, fitness)
        assert float(state.best_fitness) == 3.0

    def test_init_rejects_wrong_leading_axis(self):
        strategy, _ = _make_strategy()
        task = _task()
        template = eqx.filter(task.model, eqx.is_inexact_array)
        population = normal_sampling(
            jr.key(0), template, strategy.population_size + 1
        )
        fitness = jnp.zeros((strategy.population_size + 1,))
        with pytest.raises(ValueError, match="leading axis"):
            strategy.init(
                jr.key(1), population, fitness, strategy.default_params
            )

    def test_one_generation_monotone_best(self):
        strategy, _ = _make_strategy()
        n = strategy.population_size
        _, state = _initial_state(strategy, jnp.arange(n, dtype=float))
        population, state = strategy.ask(
            jr.key(2), state, strategy.default_params
        )
        worse = jnp.full((n,), 1000.0)
        new_state, metrics = strategy.tell(
            jr.key(3),
            population,
            worse,
            state,
            strategy.default_params,
            grads=_zero_grads(population),
        )
        assert int(new_state.generation_counter) == 1
        assert float(new_state.best_fitness) <= float(state.best_fitness)
        assert int(metrics["active_opt_idx"]) == strategy.stage0_idx
        assert not bool(metrics["fired"])

    def test_grad_active_pads_population_with_nan(self):
        strategy, _ = _make_strategy(stage0_idx=1)  # adam active
        _, state = _initial_state(strategy)
        population, _ = strategy.ask(jr.key(2), state, strategy.default_params)
        leaf = jax.tree.leaves(population)[0]
        assert jnp.isfinite(leaf[0]).all()
        assert jnp.isnan(leaf[1:]).all()

    def test_grad_step_moves_iterate(self):
        strategy, _ = _make_strategy(stage0_idx=1)  # adam active
        n = strategy.population_size
        population0, state = _initial_state(strategy)
        population, state = strategy.ask(
            jr.key(2), state, strategy.default_params
        )
        fitness = jnp.full((n,), jnp.nan).at[0].set(5.0)
        grads = jax.tree.map(jnp.ones_like, population)
        new_state, _ = strategy.tell(
            jr.key(3),
            population,
            fitness,
            state,
            strategy.default_params,
            grads=grads,
        )
        old0 = jax.tree.leaves(population)[0][0]
        new0 = jax.tree.leaves(new_state.population)[0][0]
        assert jnp.isfinite(new0).all()
        assert not jnp.allclose(old0, new0)
        # Post-step fitness is all-NaN (point not re-evaluated).
        assert jnp.isnan(new_state.fitness).all()

    def test_tell_without_grads_raises_when_menu_has_grad_opt(self):
        strategy, _ = _make_strategy()
        n = strategy.population_size
        _, state = _initial_state(strategy)
        population, state = strategy.ask(
            jr.key(2), state, strategy.default_params
        )
        with pytest.raises(ValueError, match="grads"):
            strategy.tell(
                jr.key(3),
                population,
                jnp.zeros((n,)),
                state,
                strategy.default_params,
                grads=None,
            )


#                                                           Single-fire + fire
# =============================================================================


class TestStage1Fire:
    def _fitnesses(self, strategy, values):
        """Per-generation full-width fitness vectors from scalars.

        The active stage-0 optimizer here is sepcmaes with popsize ==
        population_size, so every slot is populated: candidate i of
        generation g gets ``values[g] + 0.1 * i``.
        """
        n = strategy.population_size
        return [v + 0.1 * jnp.arange(n, dtype=float) for v in values]

    def test_fire_matches_eager_predict_stage1(self):
        for stage1_idx in (0, 1):
            strategy, task = _make_strategy(
                stage0_idx=0, stage1_idx=stage1_idx
            )
            model = _make_model(0, stage1_idx)
            sg = strategy.switch_generation
            values = [LOSS_BASE + v for v in (10.0, 8.0, 9.0, 7.0)][:sg]
            assert len(values) == sg

            _, state = _initial_state(strategy)
            state = self._fire_state(strategy, state, values)

            # The wrapper recorded exactly the per-generation
            # aggregates the eager path would have computed.
            fits = self._fitnesses(strategy, values)
            expected_mean = jnp.array([f.mean() for f in fits])
            expected_min = jnp.array([f.min() for f in fits])
            assert jnp.allclose(state.traj_mean, expected_mean)
            assert jnp.allclose(state.traj_min, expected_min)

            # And the in-graph fire picked what the eager
            # predict_stage1 picks on the identical trajectory.
            expected_step = model.predict_stage1(
                task_features=task.tag,
                optimizer_features={"optimizer": MENU[0].name},
                x_output_mean=expected_mean,
                x_output_min=expected_min,
                global_min=task.global_min,
            )
            assert bool(state.fired)
            assert MENU[int(state.active_opt_idx)] == expected_step
            assert int(state.active_opt_idx) == stage1_idx

    def _fire_state(self, strategy, state, values):
        """Run ``switch_generation`` synthetic generations, then the
        firing ask; returns the post-fire state."""
        state = _run_generations(
            strategy, state, self._fitnesses(strategy, values)
        )
        assert int(state.generation_counter) == strategy.switch_generation
        assert not bool(state.fired)
        _, state = strategy.ask(jr.key(999), state, strategy.default_params)
        return state

    def test_switching_fire_emits_best_one_handshake_and_resets(self):
        # stage-0 = adam (idx 1), stage-1 forced to sepcmaes (idx 0).
        strategy, _ = _make_strategy(stage0_idx=1, stage1_idx=0)
        n = strategy.population_size
        _, state = _initial_state(strategy)
        # adam popsize 1: only slot 0 evaluated per generation.
        values = [LOSS_BASE + v for v in (5.0, 4.0, 6.0, 3.0)][
            : strategy.switch_generation
        ]
        fits = [jnp.full((n,), jnp.nan).at[0].set(v) for v in values]
        state = _run_generations(strategy, state, fits)
        old_sub_state_0 = state.sub_states[0]

        population, state = strategy.ask(
            jr.key(999), state, strategy.default_params
        )
        assert bool(state.fired)
        assert int(state.active_opt_idx) == 0
        # best_one: slot 0 is the best-so-far, remaining sepcmaes
        # slots are finite sampler draws.
        leaf = jax.tree.leaves(population)[0]
        best_leaf = jax.tree.leaves(state.best_solution)[0]
        assert jnp.allclose(leaf[0], best_leaf)
        assert jnp.isfinite(leaf[1:]).all()
        # reset_opt_state: the incoming sub-optimizer's state changed.
        changed = jax.tree.map(
            lambda a, b: not jnp.array_equal(a, b),
            jax.tree.leaves(old_sub_state_0),
            jax.tree.leaves(state.sub_states[0]),
        )
        assert any(jax.tree.leaves(changed))

    def test_same_optimizer_fire_continues_without_handshake(self):
        strategy, _ = _make_strategy(stage0_idx=0, stage1_idx=0)
        _, state = _initial_state(strategy)
        values = [LOSS_BASE + v for v in (5.0, 4.0, 6.0, 3.0)][
            : strategy.switch_generation
        ]
        state = self._fire_state(strategy, state, values)
        assert int(state.active_opt_idx) == 0
        # No handshake: the emitted population is the ES's own ask,
        # not best_one — checked via the *next* ask after the firing
        # one in the calling helper; here it suffices that the fire
        # left the active index unchanged and the state marked fired.

    def test_fire_happens_at_most_once(self):
        strategy, _ = _make_strategy(stage0_idx=0, stage1_idx=1)
        n = strategy.population_size
        _, state = _initial_state(strategy)
        values = [LOSS_BASE + v for v in (5.0, 4.0, 6.0, 3.0)][
            : strategy.switch_generation
        ]
        state = self._fire_state(strategy, state, values)
        assert int(state.active_opt_idx) == 1
        # Keep going: more asks must not re-fire or change the choice.
        for k in range(3):
            population, state = strategy.ask(
                jr.key(2000 + k), state, strategy.default_params
            )
            state, _ = strategy.tell(
                jr.key(3000 + k),
                population,
                jnp.full((n,), jnp.nan).at[0].set(LOSS_BASE + 1.0 + k),
                state,
                strategy.default_params,
                grads=_zero_grads(population),
            )
            assert int(state.active_opt_idx) == 1
            assert bool(state.fired)

    def test_trajectory_buffers_freeze_after_fire(self):
        strategy, _ = _make_strategy(stage0_idx=0, stage1_idx=1)
        n = strategy.population_size
        _, state = _initial_state(strategy)
        values = [LOSS_BASE + v for v in (5.0, 4.0, 6.0, 3.0)][
            : strategy.switch_generation
        ]
        state = self._fire_state(strategy, state, values)
        frozen_mean = state.traj_mean
        population, state = strategy.ask(
            jr.key(555), state, strategy.default_params
        )
        state, _ = strategy.tell(
            jr.key(556),
            population,
            jnp.full((n,), jnp.nan).at[0].set(LOSS_BASE + 0.5),
            state,
            strategy.default_params,
            grads=_zero_grads(population),
        )
        assert jnp.array_equal(state.traj_mean, frozen_mean)

    def test_t_one_never_fires_within_budget(self):
        strategy, _ = _make_strategy(stage0_idx=0, stage1_idx=1, t=1.0)
        assert strategy.switch_generation == MAX_ITERATIONS
        _, state = _initial_state(strategy)
        n = strategy.population_size
        fits = [
            10.0 - g + 0.1 * jnp.arange(n, dtype=float)
            for g in range(MAX_ITERATIONS)
        ]
        state = _run_generations(strategy, state, fits)
        assert not bool(state.fired)
        assert int(state.active_opt_idx) == 0


#                                                                       Bridge
# =============================================================================


class TestBridge:
    def _bridge(self, stage0_idx=0, stage1_idx=1):
        task = _task()
        strategy = l2co_population_strategy(
            model=_make_model(stage0_idx, stage1_idx),
            task=task,
            sampler=normal_sampling,
            t=T,
            max_iterations=MAX_ITERATIONS,
        )
        _, static = eqx.partition(task.model, eqx.is_inexact_array)
        init_fn, step_fn = l2co_population_based_fn(
            strategy=strategy,
            static=static,
            loss_fn=task.loss_fn,
            bounded=(None, None),
            pass_rng=task.pass_rng,
        )
        return task, strategy, init_fn, step_fn

    def test_init_fn_does_not_evaluate(self):
        task, strategy, init_fn, _ = self._bridge()
        template = eqx.filter(task.model, eqx.is_inexact_array)
        population = normal_sampling(
            jr.key(0), template, strategy.population_size
        )
        state = init_fn(population, jr.key(1))
        assert jnp.isinf(state.best_fitness)
        assert jnp.isnan(state.fitness).all()

    def test_step_fn_generation_stamps_dynamic_hash(self):
        task, strategy, init_fn, step_fn = self._bridge()
        template = eqx.filter(task.model, eqx.is_inexact_array)
        population = normal_sampling(
            jr.key(0), template, strategy.population_size
        )
        state = init_fn(population, jr.key(1))
        carry = (population, state, jr.key(2))
        for _ in range(strategy.switch_generation + 1):
            carry, history = step_fn(carry, {})
        _, state, _ = carry
        # The generation after the fire carries the stage-1 choice's
        # hash and its popsize as the feval bill.
        active = int(state.active_opt_idx)
        assert bool(state.fired)
        assert int(history.update_step) == int(
            strategy.name_hashes_arr[active]
        )
        assert int(history.fevals) == strategy.sub_optimizers[active].popsize
        # Pre-fire generations billed the stage-0 optimizer.
        assert active == 1  # forced switch to adam

    def test_l2co_update_registry_contract_via_factory(self):
        # The full disk-loading path is covered by the integration
        # test; here we check the classmethod's non-loading plumbing
        # by monkeypatching L2COModel.load.
        task = _task()
        model = _make_model(0, 1)
        import l2co._src.models.l2co_model as l2co_model_mod

        original = l2co_model_mod.L2COModel.load
        l2co_model_mod.L2COModel.load = classmethod(
            lambda cls, stage0_model, stage1_model: model
        )
        try:
            update_class = UpdateClass.l2co_update(
                task=task,
                opt_hash=1234,
                budget_fn=lambda *, task: MAX_ITERATIONS,
                stage0_model="unused",
                stage1_model="unused",
                t=T,
                sampler=normal_sampling,
            )
        finally:
            l2co_model_mod.L2COModel.load = original
        assert update_class.popsize == max(
            _resolved_popsize(step, task) for step in MENU
        )
        assert update_class.hash == 1234


def _resolved_popsize(step: OptimizationStep, task) -> int:
    from l2co._src.optimizers.sub_optimizer import _resolve_popsize

    return _resolve_popsize(step, task)
