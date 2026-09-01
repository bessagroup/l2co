"""Guard the optimizer registry invariants (``docs/adr/0007``).

Two properties the meta-optimizer eviction relies on:

* ``import l2co`` self-registers the ``"l2co"`` meta-optimizer (it is no
  longer hardcoded in the base ``optimizer_mapping``; it wires itself in
  from ``l2co._src.meta_optimizer`` as an import side effect, exactly the
  way importing rl2co registers ``"rl2co"``).
* Every registered optimizer resolves through ``optimizer_mapping`` to a
  callable factory, case-insensitively.
"""

from __future__ import annotations

import l2co
from l2co._src.meta_optimizer import L2COUpdateClass


def test_l2co_meta_optimizer_self_registers_on_import() -> None:
    assert "l2co" in l2co.optimizers.optimizers
    # ``==`` (not ``is``): accessing a classmethod yields a fresh bound-method
    # object each time, but they compare equal (same __func__ / __self__).
    assert (
        l2co.optimizer_mapping("l2co")
        == L2COUpdateClass.l2co_population_update
    )


def test_every_registered_optimizer_resolves() -> None:
    for name in list(l2co.optimizers.optimizers):
        factory = l2co.optimizer_mapping(name)
        assert callable(factory), f"{name!r} factory is not callable"


def test_optimizer_lookup_is_case_insensitive() -> None:
    for name in list(l2co.optimizers.optimizers):
        assert l2co.optimizer_mapping(name.upper()) is l2co.optimizer_mapping(
            name
        )


def test_unknown_optimizer_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="not recognized"):
        l2co.optimizer_mapping("no_such_optimizer")
