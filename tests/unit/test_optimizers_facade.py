"""The ``l2co.optimizers`` authoring facade (``docs/adr/0007``).

Locks the contract that everything needed to author an optimizer is
reachable from the public facade and is the *same object* as its
``l2co._src`` origin, and that the pre-rename ``sub_optimizer``
underscore names survive as deprecated shims.
"""

import importlib

import pytest

import l2co.optimizers as facade


def test_all_symbols_are_importable():
    missing = [name for name in facade.__all__ if not hasattr(facade, name)]
    assert not missing, f"facade advertises but does not expose: {missing}"


@pytest.mark.parametrize(
    ("name", "module"),
    [
        # Registry & construction
        ("UpdateClass", "l2co._src.optimizers.update_class"),
        ("register_optimizer", "l2co._src.optimizers.mapping"),
        ("optimizer_mapping", "l2co._src.optimizers.mapping"),
        ("optimizers", "l2co._src.optimizers.mapping"),
        # Contract types
        ("InitFunction", "l2co._src.typing"),
        ("StepFunction", "l2co._src.typing"),
        ("StopFunction", "l2co._src.typing"),
        ("InputParameters", "l2co._src.typing"),
        ("OptState", "l2co._src.typing"),
        ("SamplerFunction", "l2co._src.typing"),
        ("PopSize", "l2co._src.typing"),
        # Loss helpers
        ("OptHistory", "l2co._src.opt_history"),
        ("vmapped_loss", "l2co._src.tree_utils"),
        ("vmapped_loss_and_grad", "l2co._src.tree_utils"),
        ("vmapped_loss_and_grad_with_rng", "l2co._src.tree_utils"),
        ("vmapped_loss_with_rng", "l2co._src.tree_utils"),
        # Sub-optimizer primitives
        ("SubOpt", "l2co._src.optimizers.sub_optimizer"),
        ("optstep_to_subopt", "l2co._src.optimizers.sub_optimizer"),
        ("resolve_popsize", "l2co._src.optimizers.sub_optimizer"),
        ("grad_sub_optimizer", "l2co._src.optimizers.sub_optimizer"),
        ("pop_sub_optimizer", "l2co._src.optimizers.sub_optimizer"),
        ("l2co_native_evosax", "l2co._src.optimizers.sub_optimizer"),
        ("CONSTRUCTOR_HYPERPARAMETERS", "l2co._src.optimizers.sub_optimizer"),
        # Base optimizer factories / registries
        (
            "normalized_optax_normal",
            "l2co._src.optimizers.optax_implementations",
        ),
        (
            "normalized_optax_from_state",
            "l2co._src.optimizers.optax_implementations",
        ),
        ("random_search_update", "l2co._src.optimizers.random_search"),
    ],
)
def test_facade_symbol_is_its_src_origin(name, module):
    origin = getattr(importlib.import_module(module), name)
    assert getattr(facade, name) is origin


DEPRECATED_ALIASES = {
    "_SubOpt": "SubOpt",
    "_optstep_to_subopt": "optstep_to_subopt",
    "_resolve_popsize": "resolve_popsize",
    "_CONSTRUCTOR_HYPERPARAMETERS": "CONSTRUCTOR_HYPERPARAMETERS",
}


@pytest.mark.parametrize(("old", "new"), DEPRECATED_ALIASES.items())
def test_sub_optimizer_underscore_alias_is_deprecated_shim(old, new):
    mod = importlib.import_module("l2co._src.optimizers.sub_optimizer")
    with pytest.warns(DeprecationWarning, match=old):
        aliased = getattr(mod, old)
    assert aliased is getattr(mod, new)


def test_unknown_sub_optimizer_attr_raises():
    mod = importlib.import_module("l2co._src.optimizers.sub_optimizer")
    with pytest.raises(AttributeError):
        _ = mod.does_not_exist
