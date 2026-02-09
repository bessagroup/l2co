"""
Module for the task abstraction.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

# Third-party
import cloudpickle
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

# Local
from .tree_utils import (
    load_dataset,
    load_jnparray,
    load_model,
    save_jnparray,
    save_model,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class DatasetDict(TypedDict):
    """
    Dictionary describing a dataset configuration.

    Attributes
    ----------
    dataset_path : str
        Path to the dataset file.
    batch_size : int
        Batch size for loading the dataset.
    seed : int
        Random seed for dataset operations.
    """

    dataset_path: str
    batch_size: int
    seed: int


# =============================================================================


class Task(eqx.Module):
    """
    Represents an optimization task.

    Attributes
    ----------
    model : PyTree
        The model used for the task.
    loss_fn : Callable
        The loss function for the task.
    global_min : float
        The global minimum value for the task, if known.
    pass_rng : bool
        Whether to pass a random key to the loss function.
    has_aux : bool
        Whether the loss function returns auxiliary outputs.
    dataset : DatasetDict
        Dataset information.
    tag : dict[str, Any]
        Metadata tags for the task.
    """

    model: PyTree
    loss_fn: Callable = eqx.field(static=True)
    global_min: float = eqx.field(static=True, default=None)
    pass_rng: bool = eqx.field(static=True, default=False)
    has_aux: bool = eqx.field(static=True, default=False)
    dataset: DatasetDict = eqx.field(default_factory=dict, static=True)
    tag: dict[str, Any] = eqx.field(default_factory=dict, static=True)

    @property
    def loaded_dataset(self) -> dict[str, Any]:
        """
        Loads the dataset specified in the task.

        Returns
        -------
        dict[str, Any]
            Loaded dataset or empty dict if no path is specified.
        """
        if self.dataset.get("dataset_path") is None:
            return dict()
        return load_dataset(self.dataset["dataset_path"])

    @property
    def batch_size(self) -> int | None:
        """
        Returns the batch size for the dataset.

        Returns
        -------
        int or None
            Batch size if specified, else None.
        """
        return self.dataset.get("batch_size", None)

    @property
    def name(self) -> str:
        """
        Returns the name of the task.

        Returns
        -------
        str
            Task name if specified, else empty string.
        """
        return self.tag.get("task_name", "")

    @property
    def dimensionality(self) -> int:
        """
        Returns the total number of parameters in the model.

        Returns
        -------
        int
            Dimensionality of the model.
        """
        return sum(
            p.size
            for p in jtu.tree_leaves(
                eqx.filter(self.model, eqx.is_inexact_array)
            )
        )

    @property
    def tag_hashable(self) -> tuple[tuple[str, Any]]:
        """
        Returns a hashable representation of the tag dictionary.

        Returns
        -------
        frozenset of tuple[str, Any]
            Hashable version of the tag dictionary as a frozenset of
            key-value pairs.
        """

        def make_hashable(value):
            """
            Recursively convert lists to tuples and dicts to frozensets.

            Parameters
            ----------
            value : Any
                Value to convert.

            Returns
            -------
            Any
                Hashable value.
            """
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            elif isinstance(value, dict):
                return frozenset(
                    (k, make_hashable(v)) for k, v in value.items()
                )
            return value  # Leave other types unchanged

        tag_hashable = {k: make_hashable(v) for k, v in self.tag.items()}

        # Add global_min to the hashable tag
        tag_hashable["global_min"] = self.global_min

        # Use frozenset for order independence
        return frozenset(tag_hashable.items())

    @property
    def hash(self) -> str:
        """
        Returns a short SHA256 hash of the tag.

        Returns
        -------
        str
            16-character hash string.
        """
        t = sorted(self.tag_hashable)
        return hashlib.sha256(str(t).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], filepath: str | None = None
    ) -> Task:
        """
        Create a Task instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing task data.
        filepath : str or None, optional
            Path to resolve model files, by default None.

        Returns
        -------
        Task
            Instantiated Task object.
        """
        if "model" in data["model_path"]:
            try:
                model = load_model(
                    model_path=data["model_path"]["model"],
                    shape_path=data["model_path"]["shape"],
                )
            except FileNotFoundError:
                new_path_model = (
                    Path(filepath).parent
                    / Path(data["model_path"]["model"]).name
                )
                new_path_shape = (
                    Path(filepath).parent
                    / Path(data["model_path"]["shape"]).name
                )
                model = load_model(
                    model_path=new_path_model, shape_path=new_path_shape
                )
        else:
            try:
                model = load_jnparray(data["model_path"]["array"])
            except FileNotFoundError:
                new_path_array = (
                    Path(filepath).parent
                    / Path(data["model_path"]["array"]).name
                )
                model = load_jnparray(new_path_array)

        # Loss function is loaded with pickle from a hex string
        loss_fn = cloudpickle.loads(bytes.fromhex(data["loss_fn"]))

        dataset = data.get("dataset", {})

        if dataset is None:
            dataset = {}

        tag = data.get("tag", {})
        has_aux = data["has_aux"]
        pass_rng = data["pass_rng"]
        global_min = data.get("global_min", None)
        return cls(
            model=model,
            global_min=global_min,
            pass_rng=pass_rng,
            loss_fn=loss_fn,
            dataset=dataset,
            has_aux=has_aux,
            tag=tag,
        )

    def save_to_json(self, filepath: Path):
        """
        Save the task to a JSON file.

        Parameters
        ----------
        filepath : Path
            Path to save the JSON file.
        """
        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(filepath), f, indent=4)

    @staticmethod
    def save(object: Task, path: str) -> str:
        """
        Save a Task object to a JSON file.

        Parameters
        ----------
        object : Task
            Task object to save.
        path : str
            Path to save the JSON file.

        Returns
        -------
        str
            Path to the saved JSON file.
        """
        object.save_to_json(Path(path))
        return Path(path).with_suffix(".json")

    def to_dict(self, path: Path) -> dict:
        """
        Convert the task to a dictionary representation for serialization.

        Parameters
        ----------
        path : Path
            Path used for saving model files.

        Returns
        -------
        dict
            Dictionary representation of the task.
        """
        path = path.resolve()  # Ensure absolute path

        # if model is eqx.Module, save as eqx.Module
        if isinstance(self.model, eqx.Module):
            save_model(self.model, path)
            model_path = {
                "model": str(path.with_suffix(".eqx").resolve()),
                "shape": str(path.with_suffix(".pkl").resolve()),
            }
        else:
            save_jnparray(self.model, path)
            model_path = {"array": str(path.with_suffix(".npy").resolve())}

        loss_fn = cloudpickle.dumps(self.loss_fn).hex()

        if self.dataset.get("dataset_path") is None:
            dataset = None
        else:
            dataset = self.dataset.copy()
            dataset["dataset_path"] = str(Path(self.dataset["dataset_path"]))

        return {
            "model_path": model_path,
            "loss_fn": loss_fn,
            "pass_rng": self.pass_rng,
            "dataset": dataset,
            "tag": self.tag,
            "has_aux": self.has_aux,
            "global_min": self.global_min,
        }

    @classmethod
    def load(cls, filepath: str) -> Task:
        """
        Load a task from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        Task
            The loaded task.
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data, filepath)

    def __repr__(self):
        """
        Returns a string representation of the Task object.

        Returns
        -------
        str
            String representation.
        """
        return f"Task(tag={self.tag})"

    def __str__(self):
        """
        Returns a formatted string describing the Task.

        Returns
        -------
        str
            Formatted string.
        """
        return f"Task:\n  Tag: {self.tag}\n"

    def __hash__(self):
        """
        Returns the hash of the Task object.

        Returns
        -------
        int
            Hash value.
        """
        return hash(self.tag_hashable)

    def __eq__(self, other):
        """
        Checks equality with another Task object.

        Parameters
        ----------
        other : Any
            Object to compare.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        return (
            isinstance(other, Task) and self.tag_hashable == other.tag_hashable
        )


# =============================================================================
