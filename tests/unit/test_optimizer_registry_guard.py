"""Guard the Task-level optimizer registry (``docs/adr/0007``, ``0018``).

* ``import l2co`` self-registers the ``"l2co"`` meta-optimizer into
  l2co's own registry (``meta_optimizers``), exactly the way importing
  rl2co registers ``"rl2co"``. It is never in the bare registry.
* Every name -- meta or bare -- resolves through ``l2co.optimizer_mapping``
  to one callable per name, case-insensitively.
* A meta-optimizer may not shadow a bare optimizer.
"""

from __future__ import annotations

import pytest
from l2co_optimizers import optimizers as bare_optimizers

import l2co
from l2co._src.meta_optimizer import L2COUpdateClass


def _all_names() -> list[str]:
    return [*bare_optimizers, *l2co.optimizers.meta_optimizers]


def test_l2co_meta_optimizer_self_registers_on_import() -> None:
    assert "l2co" in l2co.optimizers.meta_optimizers
    assert "l2co" not in bare_optimizers
    # ``==`` (not ``is``): accessing a classmethod yields a fresh bound-method
    # object each time, but they compare equal (same __func__ / __self__).
    assert (
        l2co.optimizer_mapping("l2co")
        == L2COUpdateClass.l2co_population_update
    )


def test_every_registered_optimizer_resolves() -> None:
    for name in _all_names():
        factory = l2co.optimizer_mapping(name)
        assert callable(factory), f"{name!r} factory is not callable"


def test_optimizer_lookup_is_case_insensitive() -> None:
    for name in _all_names():
        assert l2co.optimizer_mapping(name.upper()) == l2co.optimizer_mapping(
            name
        )


def test_bare_names_resolve_to_one_object() -> None:
    for name in bare_optimizers:
        assert l2co.optimizer_mapping(name) is l2co.optimizer_mapping(name)


def test_unknown_optimizer_raises() -> None:
    with pytest.raises(ValueError, match="not recognized"):
        l2co.optimizer_mapping("no_such_optimizer")


def test_meta_optimizer_cannot_shadow_a_bare_one() -> None:
    with pytest.raises(ValueError, match="bare optimizer"):
        l2co.register_optimizer("CMA_ES", lambda **_: None)
    assert "cmaes" not in l2co.optimizers.meta_optimizers
