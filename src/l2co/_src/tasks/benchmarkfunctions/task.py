from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from f3dasm import Block, ExperimentData
from jaxtyping import PyTree
from opfunu import create_benchmark_loss_function

from ... import FUNCTION_MAPPING as MY_FUNCTION_MAPPING

# =============================================================================


def save_dataset(dataset: Dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def load_dataset(path: Optional[str | Path]) -> Dict[str, jnp.ndarray]:
    if path is None:
        return {}
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


class DatasetDict(TypedDict):
    dataset_path: str
    batch_size: int
    seed: int


TagDict = Dict[str, Any]

# =============================================================================


class Task:
    def __init__(
        self,
        model: PyTree,
        loss_fn: Callable,
        dataset: Optional[DatasetDict] = None,
        loss_fn_data: Optional[DatasetDict] = None,
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

        # Loss fn data
        if loss_fn_data is None:
            self.loss_fn_data = {'dataset_path': None,
                                 'batch_size': None, 'seed': None}
        else:
            self.loss_fn_data = loss_fn_data.copy()
            # resolve relative path
            self.loss_fn_data['dataset_path'] = str(Path(
                self.loss_fn_data['dataset_path']).resolve())

        self.model = model
        self.loss_fn = loss_fn
        self.has_aux = has_aux
        self.tag = tag

    @property
    def loaded_dataset(self) -> Dict[str, Any] | None:
        return load_dataset(self.dataset['dataset_path'])

    @property
    def loaded_loss_fn_data(self) -> Dict[str, Any] | None:
        return load_dataset(self.loss_fn_data['dataset_path'])

    @property
    def batch_size(self) -> int | None:
        return self.dataset['batch_size']

    @property
    def name(self) -> str:
        return self.tag.get('task_name', '')

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
    def from_dict(cls, data: Dict[str, Any]) -> Task:

        if 'model' in data['model_path']:
            model = load_model(model_path=data['model_path']['model'],
                               shape_path=data['model_path']['shape'])
        else:
            model = load_jnparray(data['model_path']['array'])

        # Loss function is loaded with pickle from a hex string
        loss_fn = cloudpickle.loads(
            bytes.fromhex(data['loss_fn']))

        dataset = data.get('dataset', None)
        loss_fn_data = data.get('loss_fn_data', None)
        tag = data.get('tag', None)
        has_aux = data['has_aux']
        return cls(model=model, loss_fn=loss_fn, dataset=dataset,
                   loss_fn_data=loss_fn_data, has_aux=has_aux, tag=tag)

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

        if self.loss_fn_data['dataset_path'] is None:
            loss_fn_data = None

        else:
            loss_fn_data = self.loss_fn_data.copy()
            loss_fn_data['dataset_path'] = str(
                Path(self.loss_fn_data['dataset_path']))

        return {
            'model_path': model_path,
            'loss_fn': loss_fn,
            'dataset': dataset,
            'loss_fn_data': loss_fn_data,
            'tag': self.tag,
            'has_aux': self.has_aux
        }

    @classmethod
    def load(cls, filepath: str) -> Task:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

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


class BenchmarkFuncSampler(Block):
    def __init__(self, noise_distribution: List[float]):
        self.noise_distribution = noise_distribution

    def call(self, data: ExperimentData, n_samples: int,
             seed: int) -> ExperimentData:
        dims = data.domain.input_space['dimensionality'].categories
        fns = data.domain.input_space['fn_name'].categories
        noises = data.domain.input_space['noise'].categories

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
                                p=self.noise_distribution)
        seed_list = rng.integers(0, 2**30, size=n_samples)

        df = pd.DataFrame({
            "dimensionality": dims_list,
            "fn_name": function_names_list,
            "noise": noise_list,
            "seed": seed_list
        })

        return ExperimentData(domain=data.domain, input_data=df)


def random_keys_dataset(dataset_size: int, *, key: jax.Array
                        ) -> Dict[str, jax.Array]:
    return {'key': jr.randint(key=key, shape=(dataset_size,),
                              minval=0, maxval=2**16)}


def create_benchmark_task(
    fn_name: str, dataset_path: str, seed: int, noise: float,
        dimensionality: int, offset: bool = True,
        scale_output: bool = True, grad_noise: float = 0.0) -> Task:

    # add '_seed' to dataset_path
    _path = Path(dataset_path + f'_{seed}').with_suffix('.npz')

    if not _path.exists():
        dataset = random_keys_dataset(
            dataset_size=1000, key=jr.key(seed))
        save_dataset(dataset, _path)

    model, loss_fn, tag = create_benchmark_loss_function(
        fn_name=str(fn_name),
        seed=int(seed),
        scale_bounds=jnp.tile(
            jnp.array([0., 1.]), (dimensionality, 1)),
        dimensionality=int(dimensionality),
        noise=float(noise),
        offset=offset,
        scale_output=scale_output,
        grad_noise=grad_noise,
    )

    return Task(model=model, loss_fn=loss_fn,
                loss_fn_data={'dataset_path': _path,
                              'batch_size': 1,
                              'seed': seed},
                tag=tag)
