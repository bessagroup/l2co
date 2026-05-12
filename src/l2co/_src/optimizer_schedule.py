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
from pathlib import Path
from typing import Any

from jax.tree_util import Partial

# Third-party
# Local
from .stopping_criteria import STOPPING_CRITERIA

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptimizationStep:
    """Configuration for an optimization step.

    This class defines the optimizer, its hyperparameters, and stopping
    criteria for a single optimization step in a schedule.

    Parameters
    ----------
    optimizer : str
        Name of the optimizer to use.
    stopping_criteria : dict[str, Any] | None, optional
        Dictionary defining stopping criteria configuration, by default None.
    hyperparameters : dict[str, Any] | None, optional
        Dictionary of optimizer hyperparameters, by default None.

    Attributes
    ----------
    optimizer : str
        Name of the optimizer.
    hyperparameters : dict[str, Any]
        Optimizer hyperparameters.
    stopping_criteria : dict[str, Any]
        Stopping criteria configuration.
    tag : dict[str, Any]
        Tag containing optimizer name and hyperparameters.
    """

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
        """Generate a hash value for this optimization step.

        Returns
        -------
        int
            Hash value based on the tag.
        """
        t = sorted(self.tag.items())
        return int(hashlib.sha256(str(t).encode()).hexdigest()[:6], 16)

    @property
    def tag_hashable(self) -> tuple[tuple[str, Any]]:
        """Create a hashable representation of the tag.

        Returns
        -------
        tuple[tuple[str, Any]]
            Hashable frozenset of tag items.
        """

        def make_hashable(value):
            """Recursively convert lists to tuples and dicts to frozensets."""
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            elif isinstance(value, dict):
                return frozenset(
                    (k, make_hashable(v)) for k, v in value.items()
                )
            # Leave other types unchanged
            return value

        tag_hashable = {
            "opt_name": make_hashable(self.name),
            "hyperparameters": make_hashable(self.hyperparameters),
        }

        # Use frozenset for order independence
        return frozenset(tag_hashable.items())

    @property
    def name(self) -> str:
        """Generate a descriptive name for this optimization step.

        Returns
        -------
        str
            Name combining optimizer and hyperparameters.
        """
        if self.optimizer == "rl2co":
            model_path = self.hyperparameters.get("model", "")
            stem = Path(model_path).stem if model_path else ""
            return f"rl2co_{stem}".strip("_")

        hyperparameters = "_".join(
            [f"{k}={v}" for k, v in self.hyperparameters.items()]
        )

        _name = f"{self.optimizer}_{hyperparameters}"

        # Trim any leading/trailing underscores
        return _name.strip("_")

    @property
    def popsize(self) -> int:
        """Get the population size from hyperparameters.

        Returns
        -------
        int
            Population size, defaults to 1 if not specified.
        """
        return self.hyperparameters.get("popsize", 1)

    @property
    def stopping_fn(self) -> Callable:
        """Create a stopping function from the stopping criteria.

        Returns
        -------
        Callable
            Stopping function with configured parameters.
        """
        if not self.stopping_criteria:
            return lambda *args, **kwargs: False

        stop_crit = deepcopy(self.stopping_criteria)

        fn = STOPPING_CRITERIA[stop_crit.pop("type")]

        return Partial(fn, **stop_crit)

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> OptimizationStep:
        """Create an OptimizationStep from a YAML configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary from YAML.

        Returns
        -------
        OptimizationStep
            New optimization step instance.
        """
        return cls(
            optimizer=config["optimizer"],
            hyperparameters=config.get("hyperparameters", None),
            stopping_criteria=config.get("stopping_criteria", None),
        )

    @staticmethod
    def save(object: OptimizationStep, path: str) -> str:
        """Save an OptimizationStep to a JSON file.

        Parameters
        ----------
        object : OptimizationStep
            Optimization step to save.
        path : str
            File path to save to.

        Returns
        -------
        str
            Path to saved JSON file.
        """
        object.save_to_json(path)
        return Path(path).with_suffix(".json")

    @classmethod
    def load(cls, filepath: str) -> OptimizationStep:
        """Load an OptimizationStep from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file.

        Returns
        -------
        OptimizationStep
            Loaded optimization step instance.
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing optimizer configuration.
        """
        return {
            "optimizer": self.optimizer,
            "hyperparameters": self.hyperparameters,
            "stopping_criteria": self.stopping_criteria,
        }

    def save_to_json(self, filepath: Path | str):
        """Save this optimization step to a JSON file.

        Parameters
        ----------
        filepath : Path | str
            Path to save the JSON file.
        """
        with open(Path(filepath).with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationStep:
        """Create an OptimizationStep from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing optimizer configuration.

        Returns
        -------
        OptimizationStep
            New optimization step instance.
        """
        return cls(
            optimizer=data["optimizer"],
            hyperparameters=data.get("hyperparameters"),
            stopping_criteria=data.get("stopping_criteria"),
        )

    def copy(self) -> OptimizationStep:
        """Create a deep copy of this optimization step.

        Returns
        -------
        OptimizationStep
            Deep copy of this instance.
        """
        return deepcopy(self)
