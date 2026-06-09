"""Shared pytest fixtures for the L2CO test suite.

Lives at the top of ``tests/`` so both ``tests/unit/`` and
``tests/integration/`` inherit these fixtures automatically.
"""

from __future__ import annotations

import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray


@pytest.fixture
def rng() -> PRNGKeyArray:
    """Deterministic ``PRNGKeyArray`` for tests that need one."""
    return jr.key(0)
