from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Iterable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd
import xarray as xr
from jaxtyping import PyTree
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline, make_pipeline

from l2co import preprocessing_factory

from .data_loader import OptimizationDynamics
from .tasks.benchmarkfunctions.dataloader import batch_iter
from .tasks.benchmarkfunctions.optimizerschedule import OptimizationSchedule

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
    key: jnp.ndarray
    dropout_key: jnp.ndarray

    def __init__(self, in_size: int, out_size: int,
                 width_size: int, depth: int,
                 dropout_rate: float, key: jnp.ndarray):
        self.key, * \
            layer_keys, self.dropout_key = jax.random.split(key, depth + 2)
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

    def __call__(self, x: jnp.ndarray, inference: bool = False,
                 key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if key is None:
            key = self.dropout_key

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
    key: jnp.ndarray

    def __init__(self, width: int, depth: int, in_size: int, out_size: int,
                 feature_fn: Callable, key: jnp.ndarray,
                 ) -> None:
        self.key = key
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


class L2CO_model:
    def __init__(self,
                 stage0_model: BaseEstimator,
                 stage1_model_class: eqx.Module,
                 feature_types: Dict[str, str],
                 stage1_epochs: int,
                 stage1_batch_size: int,
                 stage1_optimizer: optax.GradientTransformation,
                 seed: int,
                 stage1_loss_fn: Optional[Callable] = None,
                 ):
        # Stage0 model initialization
        preprocessor = preprocessing_factory(feature_types)
        self.stage0_model = make_pipeline(
            preprocessor, MultiOutputRegressor(stage0_model))

        # Stage1 model initialization
        self.stage1_models: Dict[str, eqx.Module] = defaultdict(
            stage1_model_class)

        self.stage1_epochs = stage1_epochs
        self.stage1_batch_size = stage1_batch_size
        self.stage1_optimizer = stage1_optimizer

        if stage1_loss_fn is None:
            self.stage1_loss_fn = mse_loss_fn
        else:
            self.stage1_loss_fn = stage1_loss_fn

        self.seed = seed

    def train(self, data: OptimizationDynamics):
        if data.previous_step is None:
            model = self.stage0_model
            model = train_stage0_model(model=self.stage0_model,
                                       x=data.X_features,
                                       y=data.Y_binary())
            self.stage0_model = model

        else:
            model = self.stage1_models[data.previous_step.name]
            model = train_stage1_model(
                model=model, data=data.to_dict(),
                optimizer=self.stage1_optimizer,
                epochs=self.stage1_epochs,
                batch_size=self.stage1_batch_size,
                loss_fn=self.stage1_loss_fn,
                seed=self.seed)

            self.stage1_models[data.previous_step.name] = model

    def predict(self,
                data: OptimizationDynamics) -> Iterable[OptimizationSchedule]:
        if data.previous_step is None:
            y_pred = predict_stage0_model(
                model=self.stage0_model, x=data.X_features)

        else:
            model = self.stage1_models[data.previous_step.name]
            y_pred = predict_stage1_model(
                model=model, data=data.to_dict())

        xarr_y = xr.DataArray(
            np.array(y_pred), dims=['batch_id', 'optimizer'], coords={
                'batch_id': range(y_pred.shape[0]),
                'optimizer': list(data.opt_steps.keys())})

        schedules = [data.opt_steps[str(i.values)]
                     for i in xarr_y.idxmin('optimizer')]

        return schedules


def train_stage0_model(model: Pipeline, x: pd.DataFrame, y: jnp.ndarray):
    print(f"Training stage0 model {model}")
    model.fit(x, y)
    return model


def predict_stage0_model(model: Pipeline, x: pd.DataFrame) -> jnp.ndarray:
    print(f"Predicting with stage0 model {model}")
    return model.predict(x)


def train_stage1_model(model: eqx.Module, data: Dict[str, jnp.ndarray],
                       batch_size: int, loss_fn: Callable, epochs: int,
                       optimizer: optax.GradientTransformation, seed: int):
    print(f"Training stage1 model {model}")

    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    data_iter = batch_iter(
        dataset=data, batch_size=batch_size, key=jr.key(seed))

    @ jax.jit
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


def predict_stage1_model(model: eqx.Module, data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    print(f"Predicting with stage1 model {model}")
    return jax.vmap(model)(data['X_trajectory'])
