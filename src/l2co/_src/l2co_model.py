from __future__ import annotations

import json
# Standard
from collections import defaultdict
from functools import partial
from pathlib import Path
from time import time
from typing import Callable, Dict, Iterable, Optional

# Third-party
import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import joblib
import numpy as np
import optax
import pandas as pd
import xarray as xr
from jaxtyping import PyTree
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline, make_pipeline

# Local
from ._logging import logger
from .data_loader import OptimizationDynamics
from .optimizerschedule import OptimizationStep
from .preprocessing import DataFrameColumnTransformer, EnsureColumnsTransformer
from .tasks.benchmarkfunctions.dataloader import batch_iter

#                                                                Basis function
# =============================================================================


def sigmoid_mean(x, a, b, c):
    """Sigmoid function used for fitting."""
    return a / (1 + jnp.exp(-b * (x - c)))


def loss_fn(params, x, y):
    """Mean Squared Error (MSE) loss function."""
    a, b, c = params
    y_pred = sigmoid_mean(x, a, b, c)
    return jnp.mean((y - y_pred) ** 2)


def fit_basis_function(y: jnp.ndarray, max_iter: int = 100, tol: float = 1e-6):
    x = jnp.linspace(0., 1., len(y))
    opt = optax.lbfgs()
    init_params = jnp.array([1.0, 1.0, 0.0])
    fun = partial(loss_fn, x=x, y=y)
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, 'count')
        grad = optax.tree_utils.tree_get(state, 'grad')
        err = optax.tree_utils.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, _ = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params

#                                                                         Model
# =============================================================================


class MLPWithDropout(eqx.Module):
    layers: list
    dropout: list
    seed: int

    def __init__(self, in_size: int, out_size: int,
                 width_size: int, depth: int,
                 dropout_rate: float, key: int):
        self.seed = key
        layer_keys = jax.random.split(jr.key(key), depth)
        self.layers = []
        self.dropout = []
        for i in range(depth):
            input_dim = in_size if i == 0 else width_size
            output_dim = out_size if i == depth - 1 else width_size
            self.layers.append(eqx.nn.Linear(
                in_features=input_dim,
                out_features=output_dim, key=layer_keys[i]))
            if i < depth - 1:  # No dropout after the last layer
                self.dropout.append(eqx.nn.Dropout(
                    p=dropout_rate))

    def __call__(self, x: jnp.ndarray, key: Optional[jnp.ndarray] = None,
                 inference: bool = False) -> jnp.ndarray:
        if key is None:
            key = jax.random.PRNGKey(self.seed)
        dropout_keys = jax.random.split(key, num=len(self.dropout))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout
                x = jax.nn.relu(x)
                x = self.dropout[i](x, inference=inference,
                                    key=dropout_keys[i])
        return x


class MixedModel(eqx.Module):
    mlp: eqx.Module
    feature_fn: Callable

    def __init__(self, width: int, depth: int, in_size: int, out_size: int,
                 feature_fn: Callable, key: int,
                 ) -> None:
        # self.feature_fn = partial(feature_fn, n_terms=n_terms)
        self.feature_fn = feature_fn
        self.mlp = MLPWithDropout(
            in_size=in_size, out_size=out_size,
            width_size=width, depth=depth,
            dropout_rate=0.1, key=key)

    def __call__(self, X: jnp.ndarray,
                 key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        X_features = self.feature_fn(X)
        return self.mlp(x=X_features, inference=False, key=key)

# =============================================================================


def mse_loss_fn(model: PyTree, X_trajectory: jnp.ndarray, Y: jnp.ndarray):
    y_pred = jax.vmap(model)(X_trajectory)
    return jnp.mean((Y - y_pred) ** 2)

# =============================================================================


class L2CO_model:
    def __init__(
        self,
            stage0_model: Pipeline,
            stage0_model_optimizers: Dict[str, OptimizationStep],
            stage1_models: Dict[str, eqx.Module],
            stage1_models_optimizers: Dict[str, Dict[str, OptimizationStep]]):
        self.stage0_model = stage0_model
        self.stage0_optimizers = stage0_model_optimizers
        self.stage1_models = stage1_models
        self.stage1_models_optimizers = stage1_models_optimizers

    @classmethod
    def init(
        cls,
        stage0_model: BaseEstimator,
        stage1_model_class: eqx.Module,
        feature_types: Dict[str, str],
    ):
        # Stage0 model initialization
        stage0_model = make_pipeline(
            EnsureColumnsTransformer(),
            DataFrameColumnTransformer(feature_types),
            MultiOutputRegressor(stage0_model)
        )

        # Stage1 model initialization
        stage1_models: Dict[str, eqx.Module] = defaultdict(
            stage1_model_class)

        stage1_models_optimizers: \
            Dict[str, Dict[str, OptimizationStep]] = defaultdict(dict)

        return cls(stage0_model=stage0_model,
                   stage0_model_optimizers=dict(),
                   stage1_models=stage1_models,
                   stage1_models_optimizers=stage1_models_optimizers,)

    def save(self, path: str):
        _path = Path(path)

        _path.mkdir(parents=True, exist_ok=True)

        # Stage0 model
        joblib.dump(
            self.stage0_model, (_path / 'stage0_model').with_suffix('.pkl'))

        # Stage0 model optimizers
        with open(
                (_path / 'stage0_optimizers').with_suffix('.json'), 'w') as f:
            json.dump({k: v.to_dict()
                      for k, v in self.stage0_optimizers.items()}, f, indent=4)

        # Stage1 models
        for name, model in self.stage1_models.items():
            model_shape = eqx.filter_eval_shape(lambda x: model, None)

            # Stage1 model statics
            with open((_path / f"stage1_model_{name}").with_suffix(
                    '.pkl'), 'wb') as f:
                cloudpickle.dump(model_shape, f)

            # Stage1 model params
            eqx.tree_serialise_leaves(
                (_path / f"stage1_model_{name}").with_suffix('.eqx'), model)

            # Stage1 model optimizers
            with open(
                (_path / f"stage1_model_{name}").with_suffix('.json'),
                    'w') as f:
                json.dump(
                    {k: v.to_dict()
                     for k, v in self.stage1_models_optimizers[name].items()
                     }, f, indent=4)

    @classmethod
    def load(cls, path: str) -> L2CO_model:
        _path = Path(path)

        # Stage0 model
        stage0_model = joblib.load(
            (_path / 'stage0_model').with_suffix('.pkl'))

        # Stage0 model optimizers
        with open(
                (_path / 'stage0_optimizers').with_suffix('.json'), 'r') as f:
            stage0_optimizers = {
                k: OptimizationStep.from_dict(v)
                for k, v in json.load(f).items()}

        # Stage1 models
        stage1_models = {}
        stage1_models_optimizers = {}
        for model_path in {p.with_suffix('') for p
                           in _path.glob('stage1_model_*')}:
            with open(model_path.with_suffix('.pkl'), 'rb') as f:
                model_shape = cloudpickle.load(f)

            model = eqx.tree_deserialise_leaves(model_path.with_suffix('.eqx'),
                                                like=model_shape)

            with open(model_path.with_suffix('.json'), 'r') as f:
                optimizers = {
                    k: OptimizationStep.from_dict(v)
                    for k, v in json.load(f).items()}

            stage1_models_optimizers[model_path.stem.split(
                '_')[-1]] = optimizers

            stage1_models[model_path.stem.split('_')[-1]] = model

        return cls(stage0_model=stage0_model,
                   stage0_model_optimizers=stage0_optimizers,
                   stage1_models=stage1_models,
                   stage1_models_optimizers=stage1_models_optimizers)

    def train(self, path: str, optimizer: optax.GradientTransformation,
              epochs: int, batch_size: int, seed: int, loss_fn: Callable):
        _path = Path(path)
        data_files = set(p.stem for p in list(_path.glob('*')))

        for i, file in enumerate(data_files):
            t1 = time()
            logger.info((
                f"[{i+1}/{len(data_files)}] - Start training model "
                f"with data {file}"))

            data = OptimizationDynamics.load(_path / file)

            self.train_model(
                data=data,
                optimizer=optimizer,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                loss_fn=loss_fn
            )
            logger.info((f"[{i+1}/{len(data_files)}] - "
                        f"Trained model with data {file} in "
                         f"{time() - t1:.2f}s")
                        )

    def train_model(self, data: OptimizationDynamics,
                    optimizer: optax.GradientTransformation,
                    epochs: int, batch_size: int, seed: int,
                    loss_fn: Callable):
        if data.previous_step is None:
            model = self.stage0_model
            model = train_stage0_model(model=self.stage0_model,
                                       x=data.X_features,
                                       y=data.Y_binary())
            self.stage0_model = model
            self.stage0_optimizers = data.opt_steps

        else:
            model = self.stage1_models[data.previous_step.name]
            model = train_stage1_model(
                model=model, data=data.to_dict(),
                optimizer=optimizer,
                epochs=epochs,
                batch_size=batch_size,
                loss_fn=loss_fn,
                seed=seed
            )

            self.stage1_models[data.previous_step.name] = model
            self.stage1_models_optimizers[data.previous_step.name]\
                = data.opt_steps

    def predict(self,
                data: OptimizationDynamics) -> Iterable[OptimizationStep]:
        if data.previous_step is None:
            y_pred = predict_stage0_model(
                model=self.stage0_model, x=data.X_features)
            optimizers = self.stage0_optimizers
            xarr_y = xr.DataArray(
                np.array(y_pred), dims=['batch_id', 'optimizer'], coords={
                    'batch_id': range(y_pred.shape[0]),
                    'optimizer': list(optimizers.keys())})
            # Take the max since the value corresponds to True or False
            # For succesfull optimization
            y_opt = xarr_y.idxmax('optimizer')
        else:
            model = self.stage1_models[data.previous_step.name]
            y_pred = predict_stage1_model(
                model=model, data=data.to_dict())
            optimizers = self.stage1_models_optimizers[data.previous_step.name]
            xarr_y = xr.DataArray(
                np.array(y_pred), dims=['batch_id', 'optimizer'], coords={
                    'batch_id': range(y_pred.shape[0]),
                    'optimizer': list(optimizers.keys())})
            # Take the min since the value corresponds to the predicted
            # normalized loss
            y_opt = xarr_y.idxmin('optimizer')

        steps = [optimizers[str(i.values)]
                 for i in y_opt]

        return steps


def train_stage0_model(model: Pipeline, x: pd.DataFrame, y: jnp.ndarray):
    logger.debug(f"Training stage0 model {model}")
    model.fit(x, y)
    return model


def predict_stage0_model(model: Pipeline, x: pd.DataFrame) -> jnp.ndarray:
    logger.debug(f"Predicting with stage0 model {model}")
    return model.predict(x)


def train_stage1_model(model: eqx.Module, data: Dict[str, jnp.ndarray],
                       batch_size: int, loss_fn: Callable, epochs: int,
                       optimizer: optax.GradientTransformation, seed: int):
    logger.debug(f"Training stage1 model {model}")

    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    data_iter = batch_iter(
        dataset=data, n_evals=epochs, batch_size=batch_size, key=seed)

    @jax.jit
    def update_fn(params, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            eqx.combine(params, static), **batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for batch, epoch in zip(data_iter, range(epochs)):
        params, opt_state = update_fn(params, opt_state, batch)

    model = eqx.combine(params, static)
    return model


def predict_stage1_model(model: eqx.Module, data: Dict[str, jnp.ndarray]
                         ) -> jnp.ndarray:
    logger.debug(f"Predicting with stage1 model {model}")
    return jax.vmap(model)(data['X_trajectory'])
