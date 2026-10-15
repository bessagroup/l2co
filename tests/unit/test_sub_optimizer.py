"""``optstep_to_subopt``: an optimizer's parts plus its handshake.

The name dispatch, popsize, family and transfer ports are
``l2co_optimizers.optimizer_parts``' job and are tested there. What is
l2co's (ADR 0019) is the wrapping: the parts arrive in a ``SubOpt``
unchanged, the handshake comes from :data:`HANDSHAKE_POLICY`, and the
gradient adapter's ``step_fn`` drives a line-search transform.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
import l2co_optimizers
import pytest
from l2co_optimizers import OptimizationStep, optimizer_parts

# Local
import l2co.optimizers
from l2co.optimizers import (
    HANDSHAKE_POLICY,
    SubOpt,
    handshake_policy_for,
    optstep_to_subopt,
)

from .toy_problems import sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


@pytest.fixture(scope="module")
def problem():
    return sphere_problem(4)


def _step(name: str, **hyperparameters) -> OptimizationStep:
    return OptimizationStep(optimizer=name, hyperparameters=hyperparameters)


def _same_ports(a, b):
    """Two builds of one optimizer's switch ports declare the same."""
    assert a is not None and b is not None
    assert a.read_point is b.read_point
    assert a.read_shape is b.read_shape
    assert a.shape_rates == b.shape_rates
    assert a.restart_keeps == b.restart_keeps


class TestOwnedByL2CO:
    """The switching layer is l2co's, not a re-export (ADR 0019)."""

    @pytest.mark.parametrize(
        "name",
        [
            "SubOpt",
            "optstep_to_subopt",
            "grad_sub_optimizer",
            "pop_sub_optimizer",
            "menu_loss_and_grad",
            "HandshakePolicy",
            "HANDSHAKE_POLICY",
            "handshake_policy_for",
            "population_best",
        ],
    )
    def test_defined_in_switching(self, name):
        assert not hasattr(l2co_optimizers, name)
        obj = getattr(l2co.optimizers, name)
        module = getattr(obj, "__module__", None)
        if module is not None and not isinstance(obj, dict):
            assert module.startswith("l2co._src.switching.")


class TestGradientWrapping:
    """Gradient parts become a gradient ``SubOpt``."""

    def test_lbfgs(self, problem):
        step = _step("lbfgs", max_linesearch_steps=7)
        sub = optstep_to_subopt(step, **problem)
        parts = optimizer_parts(step, **problem)
        assert isinstance(sub, SubOpt)
        assert sub.is_pop is False
        assert sub.popsize == 1
        assert sub.tell_fn is None
        assert sub.name_hash == parts.name_hash
        assert sub.family == parts.family
        assert sub.max_fevals_per_step == 8
        assert sub.fevals_bound == 8

    def test_step_takes_a_line_searched_step(self, problem):
        # The line search needs the value/grad/value_fn extra args; a
        # plain step_fn call without them would TypeError. A successful
        # step on a quadratic must decrease the loss.
        sub = optstep_to_subopt(_step("lbfgs"), **problem)

        def value_fn(p):
            return jnp.sum(p**2)

        params = jnp.full((4,), 2.0)
        sub_state = sub.init_fn(params, jr.key(0))
        grads = jax.grad(value_fn)(params)
        update, new_state = sub.step_fn(
            grads,
            params,
            sub_state,
            jr.key(1),
            value=value_fn(params),
            grad=grads,
            value_fn=value_fn,
        )
        assert float(value_fn(params + update)) < float(value_fn(params))
        # Line-search transforms bill their true trial count.
        assert int(sub.bill_fevals(sub_state, new_state)) >= 1


class TestPopulationWrapping:
    """Population parts become a population ``SubOpt``."""

    @pytest.mark.parametrize("name", ["cmaes", "differentialevolution"])
    def test_parts_pass_through(self, name, problem):
        step = _step(name)
        sub = optstep_to_subopt(step, **problem)
        parts = optimizer_parts(step, **problem)
        assert sub.is_pop is True
        assert sub.popsize == parts.popsize
        assert sub.own_ask == parts.own_ask
        assert sub.family == parts.family
        assert sub.max_fevals_per_step is None
        # The switch ports come along, so the switch reads the same point,
        # shape and rates on this path as on the UpdateClass one.
        _same_ports(sub.switch_ports, parts.switch_ports)
        params = problem["model"]
        state = sub.init_fn(params, jr.key(0))
        population, state = sub.step_fn(params, state, jr.key(1))
        assert population.shape == (sub.popsize, *params.shape)
        fitness = jax.vmap(problem["loss_fn"])(population)
        sub.tell_fn(state, population, fitness, jr.key(2))


class TestSwitchPortsAttached:
    """A gradient ``SubOpt`` carries its parts' switch ports too."""

    def test_gradient_ports_pass_through(self, problem):
        step = _step("adam", learning_rate=1e-2)
        sub = optstep_to_subopt(step, **problem)
        parts = optimizer_parts(step, **problem)
        _same_ports(sub.switch_ports, parts.switch_ports)
        # A gradient optimizer's point is the iterate the loop holds.
        assert sub.switch_ports.read_point is None


class TestHandshakeAttached:
    """The handshake comes from the table, not from the parts."""

    @pytest.mark.parametrize("name", sorted(HANDSHAKE_POLICY))
    def test_matches_table(self, name, problem):
        hyperparameters = (
            {"learning_rate": 0.01}
            if name in l2co_optimizers.normalized_optax_normal
            else {}
        )
        step = _step(name, **hyperparameters)
        try:
            sub = optstep_to_subopt(step, **problem)
        except ValueError:
            pytest.skip(f"{name} cannot be unpacked into parts")
        policy = handshake_policy_for(step)
        assert sub.opt_state_handshake == policy.opt_state
        assert sub.population_handshake == policy.population


class TestUnresolvableName:
    def test_unregistered_name_raises(self, problem):
        with pytest.raises(ValueError, match="not recognized"):
            optstep_to_subopt(_step("no_such_optimizer"), **problem)
