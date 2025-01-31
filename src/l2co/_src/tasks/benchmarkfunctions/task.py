from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cloudpickle
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import PyTree
from opfunu import create_benchmark_loss_function

from ... import FUNCTION_MAPPING as MY_FUNCTION_MAPPING
from .dataloader import batch_iter

# =============================================================================


def save_dataset(dataset: Dict[str, jnp.ndarray], path: str | Path):
    jnp.savez(path, **dataset)


def load_dataset(path: str | Path) -> Dict[str, jnp.ndarray]:
    loaded_data = jnp.load(Path(path).with_suffix('.npz'))
    return {k: jnp.array(v) for k, v in loaded_data.items()}

# =============================================================================


def save_model(model, path: str | Path) -> None:
    _path = Path(path)
    model_shape = eqx.filter_eval_shape(lambda x: model, None)

    with open(_path.with_suffix('.pkl'), 'wb') as f:
        cloudpickle.dump(model_shape, f)

    eqx.tree_serialise_leaves(_path.with_suffix('.eqx'), model)


def load_model(model_path: str, shape_path: str) -> eqx.Module:
    _model_path = Path(model_path)
    _shape_path = Path(shape_path)
    with open(_shape_path.with_suffix('.pkl'), 'rb') as f:
        model_shape = cloudpickle.load(f)

    model = eqx.tree_deserialise_leaves(
        _model_path.with_suffix('.eqx'), model_shape)

    return model


def load_jnparray(path: str | Path) -> jnp.ndarray:
    return jnp.load(Path(path)).flatten()


def save_jnparray(model, path: str | Path) -> None:
    jnp.save(path, model)

# =============================================================================


class Task:
    def __init__(
        self,
        model: PyTree,
        loss_fn: Callable,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        tag: Optional[Dict[str, Any]] = None,
    ):
        if tag is None:
            tag = {}

        if seed is None:
            seed = 0

        if dataset_path is not None:
            dataset = load_dataset(dataset_path)
        else:
            dataset = None

        self.dataset_path = dataset_path

        self.model = model

        self.seed = seed
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.tag = tag

        self.learning_data = batch_iter(
            dataset=dataset,
            batch_size=self.batch_size,
            key=jr.key(self.seed))

    @property
    def name(self) -> str:
        return self.tag.get('task_name', '')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:

        if 'model' in data['model_path']:
            model = load_model(model_path=data['model_path']['model'],
                               shape_path=data['model_path']['shape'])
        else:
            model = load_jnparray(data['model_path']['array'])

        # Loss function is loaded with pickle from a hex string
        loss_fn = cloudpickle.loads(
            bytes.fromhex(data['loss_fn']))

        dataset_path = data.get('dataset_path', None)
        batch_size = data.get('batch_size', None)
        seed = data.get('seed', None)
        tag = data.get('tag', None)
        return cls(model=model, loss_fn=loss_fn, dataset_path=dataset_path,
                   batch_size=batch_size, seed=seed, tag=tag)

    def save_to_json(self, filepath: Path):
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(filepath), f, indent=4)

    @staticmethod
    def save(object: Task, path: str) -> str:
        object.save_to_json(Path(path))
        return Path(path).with_suffix('.json')

    def to_dict(self, path: Path) -> str:
        # if model is eqx.Module, save as eqx.Module
        if isinstance(self.model, eqx.Module):
            save_model(self.model, path)
            model_path = {'model': str(path.with_suffix('.eqx')),
                          'shape': str(path.with_suffix('.pkl'))}
        else:
            save_jnparray(self.model, path)
            model_path = {'array': str(path.with_suffix('.npy'))}

        loss_fn = cloudpickle.dumps(self.loss_fn).hex()

        return {
            'model_path': model_path,
            'loss_fn': loss_fn,
            'dataset_path': self.dataset_path,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'tag': self.tag
        }

    @classmethod
    def load(cls, filepath: str) -> Task:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self):
        return (
            f"Task(tag={self.tag}, model={self.model}, loss_fn={self.loss_fn})"
        )

    def __str__(self):
        return (
            f"Task:\n"
            f"  Tag: {self.tag}\n"
            f"  Model: {self.model}\n"
            f"  Loss Function: {self.loss_fn}"
        )

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return self.tag == other.tag


def create_benchmark_tasks(
    dims: List[int], fns: List[str],
        noises: List[float], seed: int,
        noise_distribution: List[float],
        n_samples: int) -> List[Task]:

    dim_filters = {d: list(filter(
        lambda f: MY_FUNCTION_MAPPING[f.lower()].is_dim_compatible(d)
        and MY_FUNCTION_MAPPING[f.lower()].differentiable, list(fns)))
        for d in dims}

    rng = np.random.default_rng(seed=seed)

    dims_list = rng.choice(
        dims,
        size=n_samples)
    function_names_list = [rng.choice(dim_filters[d]) for d in dims_list]
    noise_list = rng.choice(noises,
                            size=n_samples,
                            p=noise_distribution)
    seed_list = rng.integers(0, 2**30, size=n_samples)

    benchmark_list = [create_benchmark_loss_function(
        fn_name=str(fn_name),
        seed=int(s),
        scale_bounds=jnp.tile(
            jnp.array([0., 1.]), (d, 1)),
        dimensionality=int(d),
        noise=float(n),
        offset=True) for fn_name, d, n, s in zip(
            function_names_list, dims_list, noise_list, seed_list)]

    return [Task(model=model, loss_fn=loss_fn, tag=tag)
            for model, loss_fn, tag in benchmark_list]
