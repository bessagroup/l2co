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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Third-party
from jax.tree_util import Partial

# Local
from .stopping_criteria import STOPPING_CRITERIA

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Hyperparameter key holding an explicit human-facing schedule alias. When
# set, :attr:`OptimizationStep.name` returns it verbatim instead of the
# derived ``{optimizer}_{k=v}`` label — used to key a learned-optimizer
# evaluation by its training run id (e.g. ``rl2co_<run_id>``) rather than by
# its checkpoint file stem. It is a label only: policy identity stays
# weights-based (:func:`l2co._src.databank.provenance.policy_identity_hash`
# excludes this key).
ALIAS_HYPERPARAMETER = "alias"


@dataclass
class OptimizationStep:
    """Configuration for an optimization step.

    Defines the optimizer, its hyperparameters, and stopping criteria for
    a single step in an optimization schedule. Implemented as a dataclass:
    two instances compare equal iff their ``optimizer``, ``hyperparameters``,
    and ``stopping_criteria`` all match. Hashing, however, is based only on
    ``optimizer`` and ``hyperparameters`` (see :meth:`__hash__`), so steps
    that differ only in their stopping criteria collide in sets/dicts —
    valid behavior, since Python only requires equal objects to share a
    hash, not the reverse.

    Parameters
    ----------
    optimizer : str
        Name of the optimizer to use.
    stopping_criteria : dict[str, Any], optional
        Stopping criteria configuration, by default an empty dict.
    hyperparameters : dict[str, Any], optional
        Optimizer hyperparameters, by default an empty dict.

    Attributes
    ----------
    optimizer : str
        Name of the optimizer.
    hyperparameters : dict[str, Any]
        Optimizer hyperparameters.
    stopping_criteria : dict[str, Any]
        Stopping criteria configuration.
    tag : dict[str, Any]
        Derived property: ``{"opt_name": self.name, **self.hyperparameters}``.
        Recomputed on each access — always reflects the current
        ``optimizer`` / ``hyperparameters``.
    """

    optimizer: str
    stopping_criteria: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> dict[str, Any]:
        """Derived label combining ``opt_name`` with the hyperparameters.

        Recomputed on each access from :attr:`name` and
        :attr:`hyperparameters`, so it always reflects current state.

        Returns
        -------
        dict[str, Any]
            ``{"opt_name": self.name, **self.hyperparameters}``.
        """
        return {"opt_name": self.name, **self.hyperparameters}

    def __hash__(self):
        """Hash based on :attr:`tag_hashable` (name + hyperparameters).

        Intentionally narrower than the dataclass-generated ``__eq__``,
        which also compares ``stopping_criteria``. The asymmetry is fine:
        Python only requires ``a == b ⇒ hash(a) == hash(b)``.

        Goes through :attr:`tag_hashable` (lists → tuples, dicts →
        frozensets) because hyperparameters may hold unhashable
        containers — e.g. an rl2co schedule's
        ``n_iterations_per_action`` list — and a plain
        ``tuple(sorted(tag.items()))`` would raise ``TypeError`` the
        moment such a step is used as a dict key.
        """
        return hash(self.tag_hashable)

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

        An explicit :data:`ALIAS_HYPERPARAMETER` (``"alias"``), when set,
        is returned verbatim and short-circuits every derivation below —
        the eval pipeline uses it to key a learned-optimizer run by its
        training run id (``rl2co_<run_id>`` /
        ``l2co_<s0_run_id>_<s1_run_id>``) rather than by the checkpoint
        file stem, which is not a stable identity (a multi-file stage-0
        checkpoint cannot be renamed without changing its weights hash).

        Otherwise, learned meta-optimizers get compact model-derived
        names instead of the generic ``{optimizer}_{k=v}`` join, whose
        path-valued hyperparameters would otherwise leak into every
        SCHEDULE label, plot legend, and databank key: ``rl2co`` names by
        its checkpoint stem (``rl2co_10``), ``l2co`` by both stage
        checkpoint stems plus the stage-0 fraction
        (``l2co_s0-10_s1-25_t-0.05``).

        Returns
        -------
        str
            The explicit alias when set, else a name combining optimizer
            and hyperparameters.
        """
        alias = self.hyperparameters.get(ALIAS_HYPERPARAMETER)
        if alias:
            return str(alias)

        if self.optimizer == "rl2co":
            model_path = self.hyperparameters.get("model", "")
            stem = Path(model_path).stem if model_path else ""
            return f"rl2co_{stem}".strip("_")

        elif self.optimizer == "l2co":
            parts = ["l2co"]
            for key, prefix in (
                ("stage0_model", "s0"),
                ("stage1_model", "s1"),
            ):
                path = self.hyperparameters.get(key, "")
                if path:
                    parts.append(f"{prefix}-{Path(path).stem}")
            t = self.hyperparameters.get("t")
            if t is not None:
                parts.append(f"t-{t}")
            return "_".join(parts)

        elif self.optimizer.startswith("randomsearch"):
            return "randomsearch"

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
            hyperparameters=config.get("hyperparameters") or {},
            stopping_criteria=config.get("stopping_criteria") or {},
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
            hyperparameters=data.get("hyperparameters") or {},
            stopping_criteria=data.get("stopping_criteria") or {},
        )
