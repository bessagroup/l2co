"""
Module for the task abstraction.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

# Third-party
import cloudpickle
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

# Local
from .tree_utils import (load_dataset, load_jnparray, load_model,
                         save_jnparray, save_model)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class DatasetDict(TypedDict):
    dataset_path: str
    batch_size: int
    seed: int


# =============================================================================


class Task:
    """
    Represents an optimization task.

    Parameters
    ----------
    model : PyTree
        The model used for the task.
    loss_fn : Callable
        The loss function for the task.
    pass_rng : bool, optional
        Whether to pass a random key to the loss function, by default False.
    dataset : Optional[DatasetDict], optional
        Dataset information, by default None.
    has_aux : bool, optional
        Whether the loss function returns auxiliary outputs, by default False.
    tag : Optional[Dict[str, Any]], optional
        Metadata tags for the task, by default None.

    Methods
    -------
    save_to_json(filepath)
        Save the task to a JSON file.
    to_dict(path)
        Convert the task to a dictionary representation.
    """

    def __init__(
        self,
        model: PyTree,
        loss_fn: Callable,
        pass_rng: bool = False,
        dataset: Optional[DatasetDict] = None,
        has_aux: bool = False,
        tag: Optional[Dict[str, Any]] = None,
    ):
        if tag is None:
            tag = {}

        # Dataset
        if dataset is None:
            self.dataset = {'dataset_path': None,
                            'batch_size': None, 'seed': None}
        else:
            self.dataset = dataset.copy()
            # resolve relative path
            self.dataset['dataset_path'] = str(Path(
                self.dataset['dataset_path']).resolve())

        self.model = model
        self.pass_rng = pass_rng
        self.loss_fn = loss_fn
        self.has_aux = has_aux
        self.tag = tag

    @property
    def loaded_dataset(self) -> Dict[str, Any] | None:
        return load_dataset(self.dataset['dataset_path'])

    @property
    def batch_size(self) -> int | None:
        return self.dataset['batch_size']

    @property
    def name(self) -> str:
        return self.tag.get('task_name', '')

    @property
    def dimensionality(self) -> int:
        return sum(
            p.size for p in jtu.tree_leaves(
                eqx.filter(self.model, eqx.is_inexact_array))
        )

    @property
    def tag_hashable(self) -> tuple[tuple[str, Any]]:
        def make_hashable(value):
            """Recursively convert lists to tuples and dicts to frozensets."""
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            elif isinstance(value, dict):
                return frozenset((k, make_hashable(v))
                                 for k, v in value.items())
            return value  # Leave other types unchanged

        tag_hashable = {k: make_hashable(v) for k, v in self.tag.items()}

        # Use frozenset for order independence
        return frozenset(tag_hashable.items())

    @property
    def hash(self) -> str:
        t = sorted(self.tag_hashable)
        return hashlib.sha256(str(t).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  filepath: Optional[str] = None) -> Task:

        if 'model' in data['model_path']:
            try:
                model = load_model(model_path=data['model_path']['model'],
                                   shape_path=data['model_path']['shape'])
            except FileNotFoundError:
                new_path_model = Path(filepath).parent / \
                    Path(data['model_path']['model']).name
                new_path_shape = Path(filepath).parent / \
                    Path(data['model_path']['shape']).name
                model = load_model(model_path=new_path_model,
                                   shape_path=new_path_shape)
        else:
            try:
                model = load_jnparray(data['model_path']['array'])
            except FileNotFoundError:
                new_path_array = Path(filepath).parent / \
                    Path(data['model_path']['array']).name
                model = load_jnparray(new_path_array)

        # Loss function is loaded with pickle from a hex string
        loss_fn = cloudpickle.loads(
            bytes.fromhex(data['loss_fn']))

        dataset = data.get('dataset', None)
        tag = data.get('tag', None)
        has_aux = data['has_aux']
        pass_rng = data['pass_rng']
        return cls(model=model, pass_rng=pass_rng, loss_fn=loss_fn,
                   dataset=dataset, has_aux=has_aux, tag=tag)

    def save_to_json(self, filepath: Path):
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(filepath), f, indent=4)

    @staticmethod
    def save(object: Task, path: str) -> str:
        object.save_to_json(Path(path))
        return Path(path).with_suffix('.json')

    def to_dict(self, path: Path) -> str:
        path = path.resolve()  # Ensure absolute path

        # if model is eqx.Module, save as eqx.Module
        if isinstance(self.model, eqx.Module):
            save_model(self.model, path)
            model_path = {'model': str(path.with_suffix('.eqx').resolve()),
                          'shape': str(path.with_suffix('.pkl').resolve())}
        else:
            save_jnparray(self.model, path)
            model_path = {'array': str(path.with_suffix('.npy').resolve())}

        loss_fn = cloudpickle.dumps(self.loss_fn).hex()

        if self.dataset['dataset_path'] is None:
            dataset = None

        else:
            dataset = self.dataset.copy()
            dataset['dataset_path'] = str(
                Path(self.dataset['dataset_path']))

        return {
            'model_path': model_path,
            'loss_fn': loss_fn,
            'pass_rng': self.pass_rng,
            'dataset': dataset,
            'tag': self.tag,
            'has_aux': self.has_aux
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
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, filepath)

    def __repr__(self):
        return (
            f"Task(tag={self.tag}"
        )

    def __str__(self):
        return (
            f"Task:\n"
            f"  Tag: {self.tag}\n"
        )

    def __hash__(self):
        return hash(self.tag_hashable)

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return self.tag == other.tag


# =============================================================================
