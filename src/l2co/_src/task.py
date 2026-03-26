"""
Module for the task abstraction.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

# Third-party
import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
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

    @staticmethod
    def _reconstruct_model_skeleton(hyperparams: dict):
        """
        Reconstruct a model skeleton (PyTree of ShapeDtypeStructs) from header
        metadata, ready for use as the target of
        ``eqx.tree_deserialise_leaves``.

        Parameters
        ----------
        hyperparams : dict
            Decoded JSON header produced by :meth:`to_dict`.

        Returns
        -------
        PyTree
            Skeleton with the same structure as the original model.
        """
        if hyperparams["model_type"] == "eqx_module":
            return cloudpickle.loads(
                bytes.fromhex(hyperparams["model_shape_hex"])
            )
        else:
            return jax.ShapeDtypeStruct(
                tuple(hyperparams["model_shape"]),
                jnp.dtype(hyperparams["model_dtype"]),
            )

    @classmethod
    def _from_header(cls, hyperparams: dict, model) -> Task:
        """
        Assemble a Task from a decoded JSON header and an already-loaded model.

        Parameters
        ----------
        hyperparams : dict
            Decoded JSON header produced by :meth:`to_dict`.
        model : PyTree
            Model with weights filled in by ``eqx.tree_deserialise_leaves``.

        Returns
        -------
        Task
            Fully assembled Task.
        """
        loss_fn = cloudpickle.loads(bytes.fromhex(hyperparams["loss_fn"]))
        dataset = hyperparams.get("dataset") or {}
        return cls(
            model=model,
            loss_fn=loss_fn,
            global_min=hyperparams.get("global_min"),
            pass_rng=hyperparams["pass_rng"],
            has_aux=hyperparams["has_aux"],
            dataset=dataset,
            tag=hyperparams.get("tag", {}),
        )

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], filepath: str | None = None
    ) -> Task:
        """
        Create a Task from a legacy JSON-format dictionary.

        This supports the old multi-file format (produced by the old
        ``to_dict(path)`` / ``save_to_json`` methods). For new code use
        :meth:`Task.load` with a ``.eqx`` path.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary as written by the old ``to_dict(path)`` method,
            containing a ``model_path`` key with file-path references.
        filepath : str or None, optional
            Path used to resolve relative model file paths, by default None.

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
        Save the task to disk.

        .. deprecated::
            Use :meth:`Task.save` instead. This method now delegates to
            ``Task.save`` and writes a single ``.eqx`` file rather than
            a JSON file.

        Parameters
        ----------
        filepath : Path
            Base path (extension is ignored; ``.eqx`` will be used).
        """
        warnings.warn(
            "save_to_json is deprecated; use Task.save() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        Task.save(self, str(filepath))

    @staticmethod
    def save(object: Task, path: str) -> str:
        """
        Save a Task to a single self-contained ``.eqx`` file.

        The file format mirrors :class:`Stage1Loader`: the first line is a
        UTF-8 JSON header containing all metadata (loss function, dataset
        reference, tags, model shape), followed immediately by the binary
        leaf data written by ``eqx.tree_serialise_leaves``.  The dataset
        ``.npz`` file is **not** copied — only its path is stored in the
        header, so the dataset is stored only once on disk.

        Parameters
        ----------
        object : Task
            Task object to save.
        path : str
            Base path for the output file (extension is replaced with
            ``.eqx``).

        Returns
        -------
        str
            Path to the saved ``.eqx`` file.
        """
        _path = Path(path)
        hyperparams = object.to_dict()
        # Store dataset_path relative to the .eqx file so the task is
        # portable: moving the .eqx and .npz files together keeps the
        # reference valid regardless of machine or working directory.
        if hyperparams.get("dataset") and hyperparams["dataset"].get(
            "dataset_path"
        ):
            abs_dataset = Path(
                hyperparams["dataset"]["dataset_path"]
            ).resolve()
            abs_task_dir = _path.resolve().parent
            hyperparams["dataset"]["dataset_path"] = os.path.relpath(
                abs_dataset, abs_task_dir
            )
        with open(_path.with_suffix(".eqx"), "wb") as f:
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, object.model)
        return str(_path.with_suffix(".eqx"))

    def to_dict(self) -> dict:
        """
        Convert the task to a metadata dictionary for serialization.

        No files are written by this method. Model weights are stored
        separately via ``eqx.tree_serialise_leaves`` in :meth:`save`.

        Returns
        -------
        dict
            Metadata dictionary suitable for use as a JSON header.
        """
        loss_fn_hex = cloudpickle.dumps(self.loss_fn).hex()

        if isinstance(self.model, eqx.Module):
            model_shape = eqx.filter_eval_shape(lambda: self.model, None)
            model_info = {
                "model_type": "eqx_module",
                "model_shape_hex": cloudpickle.dumps(model_shape).hex(),
            }
        else:
            arr = jnp.asarray(self.model)
            model_info = {
                "model_type": "jnp_array",
                "model_shape": list(arr.shape),
                "model_dtype": str(arr.dtype),
            }

        if self.dataset.get("dataset_path") is None:
            dataset = None
        else:
            dataset = dict(self.dataset)
            dataset["dataset_path"] = str(Path(self.dataset["dataset_path"]))

        return {
            **model_info,
            "loss_fn": loss_fn_hex,
            "pass_rng": self.pass_rng,
            "has_aux": self.has_aux,
            "global_min": self.global_min,
            "dataset": dataset,
            "tag": self.tag,
        }

    @classmethod
    def load(cls, filepath: str) -> Task:
        """
        Load a Task from disk.

        Supports both the new single-file ``.eqx`` format and the legacy
        multi-file ``.json`` format for backward compatibility.

        Parameters
        ----------
        filepath : str
            Path to the task file.  For the new format supply either the
            base path or the ``.eqx`` path; for the legacy format supply
            the ``.json`` path.

        Returns
        -------
        Task
            The loaded task.
        """
        _path = Path(filepath)
        if _path.suffix == ".json":
            # Legacy multi-file format: .json + .eqx/.pkl or .npy
            warnings.warn(
                f"Loading legacy .json Task format from {filepath!r}. "
                "Re-save with Task.save() to migrate to the new single-file "
                ".eqx format.",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(_path) as f:
                data = json.load(f)
            return cls.from_dict(data, filepath)
        # New single-file format
        eqx_path = _path.with_suffix(".eqx").resolve()
        with open(eqx_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            # Resolve a relative dataset_path against the .eqx file's
            # directory so loading works regardless of CWD or machine.
            if hyperparams.get("dataset") and hyperparams["dataset"].get(
                "dataset_path"
            ):
                ds_path = Path(hyperparams["dataset"]["dataset_path"])
                if not ds_path.is_absolute():
                    hyperparams["dataset"]["dataset_path"] = str(
                        (eqx_path.parent / ds_path).resolve()
                    )
            skeleton = cls._reconstruct_model_skeleton(hyperparams)
            model = eqx.tree_deserialise_leaves(f, skeleton)
        return cls._from_header(hyperparams, model)

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
