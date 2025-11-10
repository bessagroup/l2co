"""
Module for optimizer schedules.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

# Third-party
import jax.numpy as jnp

# Local
from .stopping_criteria import STOPPING_CRITERIA

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptimizationStep:
    def __init__(
        self,
        optimizer: str,
        stopping_criteria: dict[str, Any] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ):
        if hyperparameters is None:
            hyperparameters = {}

        if stopping_criteria is None:
            stopping_criteria = {}

        self.optimizer = optimizer
        self.hyperparameters = dict(hyperparameters)
        self.stopping_criteria = dict(stopping_criteria)
        self.tag = {"opt_name": self.name, **hyperparameters}

    def __repr__(self):
        return (
            f"Optimizer: {self.optimizer}, "
            f"Hyperparameters: {self.hyperparameters}, "
            f"Stopping criteria: {self.stopping_criteria}"
        )

    def __str__(self):
        return (
            f"Optimizer: {self.optimizer}\n"
            f"  Hyperparameters: {self.hyperparameters}\n"
            f"  Stopping criteria: {self.stopping_criteria}\n"
        )

    def __eq__(self, other):
        if not isinstance(other, OptimizationStep):
            return False
        return (
            self.optimizer == other.optimizer
            and self.hyperparameters == other.hyperparameters
        )

    def __hash__(self):
        tag_hashable = (
            tuple(sorted(self.tag.items()))
            if isinstance(self.tag, dict)
            else None
        )

        return hash(tag_hashable)

    @property
    def hash(self) -> int:
        t = sorted(self.tag.items())
        return jnp.array(
            int(hashlib.sha256(str(t).encode()).hexdigest()[:8], 16),
            dtype=jnp.uint32,
        )

    @property
    def tag_hashable(self) -> tuple[tuple[str, Any]]:
        def make_hashable(value):
            """Recursively convert lists to tuples and dicts to frozensets."""
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            elif isinstance(value, dict):
                return frozenset(
                    (k, make_hashable(v)) for k, v in value.items()
                )
            return value  # Leave other types unchanged

        tag_hashable = {
            "opt_name": make_hashable(self.name),
            "hyperparameters": make_hashable(self.hyperparameters),
        }

        # Use frozenset for order independence
        return frozenset(tag_hashable.items())

    @property
    def name(self) -> str:
        hyperparameters = "_".join(
            [f"{k}={v}" for k, v in self.hyperparameters.items()]
        )

        _name = f"{self.optimizer}_{hyperparameters}"

        # Trim any leading/trailing underscores
        return _name.strip("_")

    @property
    def popsize(self) -> int:
        return self.hyperparameters.get("popsize", 1)

    @property
    def stopping_fn(self) -> Callable:
        if not self.stopping_criteria:
            return lambda *args, **kwargs: False

        fn = STOPPING_CRITERIA[self.stopping_criteria.pop("type")]

        return partial(fn, **self.stopping_criteria)

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> OptimizationStep:
        return cls(
            optimizer=config["optimizer"],
            hyperparameters=config.get("hyperparameters", None),
            stopping_criteria=config.get("stopping_criteria", None),
        )

    @staticmethod
    def save(object: OptimizationStep, path: str) -> str:
        object.save_to_json(path)
        return Path(path).with_suffix(".json")

    @classmethod
    def load(cls, filepath: str) -> OptimizationStep:
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "hyperparameters": self.hyperparameters,
            "stopping_criteria": self.stopping_criteria,
        }

    def save_to_json(self, filepath: Path | str):
        with open(Path(filepath).with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationStep:
        return cls(
            optimizer=data["optimizer"],
            hyperparameters=data.get("hyperparameters"),
            stopping_criteria=data.get("stopping_criteria"),
        )

    def copy(self) -> OptimizationStep:
        return deepcopy(self)
