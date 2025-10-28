"""
Module for optimizer schedules.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

# Third-party
import jax
import jax.numpy as jnp

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptimizationStep:
    def __init__(
        self, optimizer: str, hyperparameters: dict[str, Any] | None = None
    ):
        if hyperparameters is None:
            hyperparameters = {}

        self.optimizer = optimizer
        self.hyperparameters = dict(hyperparameters)
        self.tag = {"opt_name": self.name, **hyperparameters}

    def __repr__(self):
        return (
            f"Optimizer: {self.optimizer}, "
            f"Hyperparameters: {self.hyperparameters}"
        )

    def __str__(self):
        return (
            f"Optimizer: {self.optimizer}\n"
            f"  Hyperparameters: {self.hyperparameters}\n"
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

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> OptimizationStep:
        if "hyperparameters" not in config:
            return cls(optimizer=config["optimizer"])
        else:
            return cls(
                optimizer=config["optimizer"],
                hyperparameters=config["hyperparameters"],
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
        }

    def save_to_json(self, filepath: Path | str):
        with open(Path(filepath).with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationStep:
        return cls(
            optimizer=data["optimizer"],
            hyperparameters=data.get("hyperparameters"),
        )

    def copy(self) -> OptimizationStep:
        return deepcopy(self)


# =============================================================================


class OptimizationSchedule:
    def __init__(
        self,
        steps: OptimizationStep | list[OptimizationStep],
        budgets: float | list[float],
    ):
        if isinstance(steps, OptimizationStep):
            steps = [steps]

        if isinstance(budgets, (float, int)):
            budgets = [budgets]

        self.steps = [step.copy() for step in steps]

        names = [step.name for step in steps]
        self.name = (
            names[0]
            if all(name == names[0] for name in names)
            else "+".join(names)
        )

        self.tag = {
            "opt_name": self.name,
            "steps": [step.tag for step in steps],
            "budgets": budgets,
        }
        self.budgets = budgets

    def __lt__(self, __o: OptimizationSchedule) -> bool:
        return self.name < __o.name

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

        tag_hashable = {k: make_hashable(v) for k, v in self.tag.items()}

        # Use frozenset for order independence
        return frozenset(tag_hashable.items())

    @property
    def hash(self) -> str:
        return hashlib.sha256(str(self.tag_hashable).encode()).hexdigest()[:16]

    def __iter__(self) -> Iterator[OptimizationStep]:
        return iter(self.steps)

    def __repr__(self):
        return (
            f"OptimizationSchedule(tag={self.name}, "
            f"steps={self.steps}), "
            f"budgets={self.budgets}"
        )

    def __str__(self):
        step_details = "\n".join(
            [f"Step {i + 1}:\n{step}" for i, step in enumerate(self.steps)]
        )
        return (
            f"Optimization Schedule ({self.name}):\n{step_details}\n "
            f"- budgets: {self.budgets}"
        )

    def __eq__(self, other):
        if not isinstance(other, OptimizationSchedule):
            return False
        return (
            self.tag == other.tag
            and self.steps == other.steps
            and self.budgets == other.budgets
        )

    def __hash__(self):
        return hash(self.tag_hashable)

    def __len__(self):
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "budgets": self.budgets,
        }

    def to_policy(schedule: OptimizationSchedule):
        budgets = jnp.array(schedule.budgets)
        n = budgets.size
        logits = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, -jnp.inf)
        cum_budgets = jnp.cumsum(budgets)

        def policy_fn(obs: jax.Array, time: float) -> jax.Array:
            which = jnp.sum(time >= cum_budgets)
            return jax.lax.dynamic_index_in_dim(logits, which)

        return policy_fn

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationSchedule:
        steps = [
            OptimizationStep.from_dict(step_data)
            for step_data in data["steps"]
        ]
        return cls(steps=steps, budgets=data["budgets"])

    @classmethod
    def from_json_string(cls, json_string: str) -> OptimizationSchedule:
        data = json.loads(json_string)
        return cls.from_dict(data)

    def save_to_json(self, filepath: Path | str):
        with open(Path(filepath).with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def save(object: OptimizationSchedule, path: str) -> str:
        object.save_to_json(path)
        return Path(path).with_suffix(".json")

    @classmethod
    def load(cls, filepath: str) -> OptimizationSchedule:
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> OptimizationSchedule:
        return deepcopy(self)
