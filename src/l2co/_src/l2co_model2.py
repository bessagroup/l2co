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
import jax.nn as jnn
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
from .batching import batch_iter
from .data_loader import OptimizationDynamics
from .optimizerschedule import OptimizationStep
from .preprocessing import DataFrameColumnTransformer, EnsureColumnsTransformer


class LSTM_Encoder(eqx.Module):
    hidden_size: int
    cell: eqx.nn.LSTMCell

    def __init__(self, in_size, hidden_size, *, key):
        self.hidden_size = hidden_size
        # self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key)

    def __call__(self, x):
        def scan_fn(state, input):
            return (self.cell(input, state), None)

        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))
        (h, c), _ = jax.lax.scan(scan_fn, init_state, x)
        return h


class MyModel(eqx.Module):
    lstm_layer: eqx.Module
    mlp_layer: eqx.Module

    def __init__(self, in_size: int, out_size: int, hidden_size: int,
                 width_size: int, depth: int, *, key: jax.Array):
        lstm_key, mlp_key = jr.split(key)
        self.lstm_layer = LSTM_Encoder(
            in_size=in_size, hidden_size=hidden_size, key=lstm_key)
        self.mlp_layer = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=mlp_key,
            activation=jnn.relu,
            final_activation=jnn.sigmoid
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.lstm_layer(x)
        x = self.mlp_layer(x)
        return x

# =============================================================================


@eqx.filter_jit
def mse_loss(model, x, y):
    y_hat = jax.vmap(model)(x)
    return jnp.mean((y - y_hat) ** 2)


def train_stage1_model(model: eqx.Module, data: Dict[str, jax.Array],
                       loss_fn: Callable,
                       epochs: int,
                       optimizer: optax.GradientTransformation,
                       ):

    params, static = eqx.partition(model, eqx.is_inexact_array)

    @eqx.filter_jit
    def make_step(params, opt_state, data):
        loss, grads = eqx.filter_value_and_grad(
            loss_fn)(eqx.combine(params, static), **data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    def body(carry, _):
        params, opt_state = carry
        train_loss, params, opt_state = make_step(
            params, opt_state, data)

        return (params, opt_state), {'train_loss': train_loss}

    opt_state = optimizer.init(params)
    (params, _), losses = jax.lax.scan(
        body, (params, opt_state), None, length=epochs)
    return eqx.combine(params, static)


def predict_stage1_model(model: eqx.Module, data: Dict[str, jax.Array]
                         ) -> jax.Array:
    y_hat = jax.vmap(model)(**data)

    # Select prediction with lowest performance value
    return jnp.argmin(y_hat, axis=1)

    # =============================================================================


def train_stage0_model(model: Pipeline, x: pd.DataFrame, y: jnp.ndarray):
    logger.debug(f"Training stage0 model {model}")
    model.fit(x, y)
    return model


def predict_stage0_model(model: Pipeline, x: pd.DataFrame) -> ...:
    logger.debug(f"Predicting with stage0 model {model}")
    y_hat = model.predict(x)

    # Select prediction with lowest performance value
    return jnp.argmin(y_hat, axis=1)

# =============================================================================


class L2CO_model:
    def __init__(self,
                 stage0_model: Pipeline,
                 stage1_models: Dict[str, eqx.Module]):
        self.stage0_model = stage0_model
        self.stage1_models = stage1_models
        self.opt_steps = None

    @classmethod
    def init(
        cls, stage0_model: BaseEstimator, stage1_model_class: eqx.Module,
        feature_types: Dict[str, str]
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

        return cls(stage0_model=stage0_model,
                   stage1_models=stage1_models)

    def train(self, dynamics_list: Iterable[OptimizationDynamics],
              opt_steps: Dict[str, OptimizationStep],
              optimizer: optax.GradientTransformation,
              epochs: int):

        self.opt_steps = opt_steps

        for dynamics in dynamics_list:
            if dynamics.previous_step is None:
                train_stage0_model(model=self.stage0_model,
                                   x=dynamics.X_features, y=dynamics.Y)

            else:
                self.stage1_models[
                    dynamics.previous_step] = train_stage1_model(
                    model=self.stage1_models[dynamics.previous_step],
                    data={'x': dynamics.X_trajectory, 'y': dynamics.Y},
                    loss_fn=mse_loss,
                    epochs=epochs,
                    optimizer=optimizer)

    def predict(self, dynamics: OptimizationDynamics):
        if dynamics.previous_step is None:
            y_hat = predict_stage0_model(model=self.stage0_model,
                                         x=dynamics.X_features)
        else:
            y_hat = predict_stage1_model(
                model=self.stage1_models[dynamics.previous_step],
                data={'x': dynamics.X_trajectory}
            )

        return y_hat
