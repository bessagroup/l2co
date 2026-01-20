from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import joblib
import numpy as np
import optax
import xarray as xr
from jax_tqdm import scan_tqdm
from jaxtyping import PRNGKeyArray
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from l2co.naming import SCHEDULE

from .data_loader import OptimizationDynamics
from .naming import BUDGET, KIND, PROBLEM
from .optimizerschedule import OptimizationStep
from .task_features import FeatureProcessor, MixedEmbedding

# =============================================================================


T = TypeVar("T", bound="AbstractModel")


class AbstractModel(eqx.Module):
    """
    Abstract base class for policy models.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize model to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model configuration.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def reconstruct(cls: type[T], data: dict[str, Any]) -> T:
        """
        Reconstruct model from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing model configuration.

        Returns
        -------
        T
            Reconstructed model instance.
        """
        raise NotImplementedError


# =============================================================================


class L2COModel(AbstractModel):
    feature_processor: FeatureProcessor
    model: BaseEstimator | AbstractModel
    type: Literal["scikit", "equinox"]
    optimizers: Iterable[OptimizationStep]

    @classmethod
    def init(
        cls,
        feature_processor: FeatureProcessor,
        model: BaseEstimator | AbstractModel,
        type: Literal["scikit", "equinox"],
        optimizers: Iterable[OptimizationStep],
    ) -> L2COModel:
        if type not in ["scikit", "equinox"]:
            raise ValueError(f"Unknown model type {type}")

        if type == "scikit":
            model = MultiOutputRegressor(model)

        return cls(
            feature_processor=feature_processor,
            model=model,
            type=type,
            optimizers=optimizers,
        )

    def train(
        self, data: OptimizationDynamics, **kwargs
    ) -> tuple[L2COModel, dict]:
        if self.type == "scikit":
            model, metrics = train_scikit(
                model=self.model,
                x=jnp.array(data.x_features),
                y=jnp.array(data.y),
                **kwargs,
            )
        elif self.type == "equinox":
            model, metrics = train_equinox(
                model=self.model,
                x=jnp.array(data.x_features),
                y=jnp.array(data.y),
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model type {self.type}")

        l2co_model = L2COModel(
            feature_processor=self.feature_processor,
            model=model,
            type=self.type,
            optimizers=self.optimizers,
        )

        return l2co_model, metrics

    def predict_logits(self, x: dict[str, Any]) -> jax.Array:
        x_array = self.feature_processor.transform(x)
        if self.type == "scikit":
            logits = predict_scikit(self.model, x_array)
        elif self.type == "equinox":
            logits = predict_equinox(self.model, x_array)
        else:
            raise ValueError(f"Unknown model type {self.type}")

        return logits

    def predict(
        self, x: dict[str, Any], key: PRNGKeyArray, n_realizations: int = 1
    ) -> OptimizationStep:
        logits = self.predict_logits(x)
        keys = jr.split(key, num=n_realizations)
        actions = jax.vmap(jr.categorical, in_axes=(0, None))(keys, logits)

        action_list = list(map(int, actions))
        steps = [self.optimizers[i] for i in action_list]
        return steps

    def batch_predict(self, x: dict[str, Any], key: PRNGKeyArray) -> jax.Array:
        x_array = jnp.stack([self.predict_logits(xi) for xi in x])
        if self.type == "scikit":
            logits = predict_scikit(self.model, x_array)
        elif self.type == "equinox":
            logits = predict_equinox(self.model, x_array)
        else:
            raise ValueError(f"Unknown model type {self.type}")

        # Sample from logits
        keys = jr.split(key, num=len(x))
        actions = jax.vmap(jr.categorical)(keys, logits)
        action_list = list(map(int, actions))
        steps = [self.optimizers[i] for i in action_list]
        return steps

    def to_dict(self) -> dict[str, Any]:
        match self.type:
            case "scikit":
                model = self.model
            case "equinox":
                model = self.model.to_dict()

        return {
            "feature_processor": self.feature_processor.to_dict(),
            "model": model,
            "type": self.type,
            "optimizers": [step.to_dict() for step in self.optimizers],
        }

    @classmethod
    def reconstruct(
        cls,
        feature_processor: dict,
        optimizers: Iterable[dict],
        model: dict | BaseEstimator,
        type: Literal["scikit", "equinox"],
        key: PRNGKeyArray,
        **kwargs,
    ) -> L2COModel:
        feature_processor = FeatureProcessor.reconstruct(**feature_processor)
        optimizers = [
            OptimizationStep.from_yaml(step_dict) for step_dict in optimizers
        ]
        match type:
            case "scikit":
                pass
            case "equinox":
                model = make(key=key, **model)
            case _:
                raise ValueError(f"Unknown model type {type}")

        return cls(
            feature_processor=feature_processor,
            model=model,
            optimizers=optimizers,
            type=type,
            **kwargs,
        )


class HistoryModel(AbstractModel):
    mixed_embedding: MixedEmbedding
    feature_processor: FeatureProcessor
    history_encoder: eqx.Module
    model: eqx.Module
    optimizers: Iterable[OptimizationStep]

    def __call__(self, x: tuple[jax.Array, jax.Array]) -> jax.Array:
        x_features, x_history = x
        history_emb = self.history_encoder(x_history)
        features_emb = self.mixed_embedding(x_features)
        combined = jnp.concatenate([features_emb, history_emb], axis=-1)
        return self.model(combined)

    def train(
        self,
        data: OptimizationDynamics,
        i: int | None,
        optimizer: optax.GradientTransformation,
        num_steps: int = 100,
        verbose: bool = False,
        scan: bool = True,
    ) -> tuple[HistoryModel, dict]:
        # 1. Filter problems with invalid targets (must be valid for all
        # # schedules)
        # Note: We assume the model predicts the full y vector, so y must
        # be valid.
        y_mask = ~data.y.isnull().any(SCHEDULE)
        data = data.select(problems=y_mask)

        # 2. Split by problem to prevent leakage
        data_train, data_val = data.train_test_split(
            train_fraction=0.8, seed=42
        )

        def make_dataset(d: OptimizationDynamics) -> tuple[tuple, jax.Array]:
            if i is not None:
                # Single schedule mode
                xh = d.x_history.isel({SCHEDULE: i}, drop=True)  # (P, B, K)

                # Filter indices with valid history
                valid_mask = ~xh.isnull().any([BUDGET, KIND])
                xh = xh.where(valid_mask, drop=True)

                # Align features and targets
                xf = d.x_features.sel({PROBLEM: xh[PROBLEM]})
                y = d.y.sel({PROBLEM: xh[PROBLEM]})

                return (jnp.array(xf), jnp.array(xh)), jnp.array(y)
            else:
                # All schedules mode
                # Stack (PROBLEM, SCHEDULE) into a single sample dimension
                xh = d.x_history.stack(sample=(PROBLEM, SCHEDULE))

                # Filter indices with valid history
                valid_mask = ~xh.isnull().any([BUDGET, KIND])
                xh = xh.where(valid_mask, drop=True)

                # Align features and targets
                # xh[PROBLEM] gives the problem index for each valid sample
                xf = d.x_features.sel({PROBLEM: xh[PROBLEM]})
                y = d.y.sel({PROBLEM: xh[PROBLEM]})

                # Transpose xh to (sample, B, K) to match expected input
                #  shape for batching
                xh = xh.transpose("sample", BUDGET, "kind")

                return (jnp.array(xf), jnp.array(xh)), jnp.array(y)

        x_train, y_train = make_dataset(data_train)
        x_val, y_val = make_dataset(data_val)

        model, metrics = train_equinox(
            model=self,
            optimizer=optimizer,
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            num_steps=num_steps,
            verbose=verbose,
            scan=scan,
        )
        return model, metrics

    def evaluate(
        self,
        key: PRNGKeyArray,
        data: OptimizationDynamics,
        i: int | None,
    ) -> jax.Array:
        if i is not None:
            x_eval = (
                jnp.array(
                    data.x_features,
                ),
                jnp.array(data.x_history.isel(schedule=i, drop=True)),
            )

            logits = eqx.filter_vmap(self)(x_eval)
            # actions: jax.Array = jr.categorical(key, logits)
            actions: jax.Array = jnp.argmax(logits, axis=-1)

            optimizers = [self.optimizers[k] for k in actions]
            y = (
                xr.DataArray(
                    optimizers,
                    dims="problem",
                    coords={"problem": data.x_features[PROBLEM]},
                )
                .assign_coords({PROBLEM: data.x_features[PROBLEM]})
                .drop_vars(PROBLEM)
            )
        else:
            # i is None: Evaluate on ALL (Problem, Schedule) pairs

            # Stack to create single sample dimension
            xh_stacked = data.x_history.stack(sample=(PROBLEM, SCHEDULE))

            # Align features
            # xh_stacked[PROBLEM] gives the problem index for each sample
            xf_aligned = data.x_features.sel({PROBLEM: xh_stacked[PROBLEM]})
            x_eval = (
                jnp.array(xf_aligned),
                jnp.array(xh_stacked.transpose("sample", BUDGET, "kind")),
            )

            logits = eqx.filter_vmap(self)(x_eval)
            actions: jax.Array = jnp.argmax(logits, axis=-1)

            optimizers = [self.optimizers[k] for k in actions]

            # Create DataArray with stacked coord
            y_flat = xr.DataArray(
                optimizers,
                dims="sample",
                coords={"sample": xh_stacked["sample"]},
            )

            # Unstack back to (PROBLEM, SCHEDULE)
            y = y_flat.unstack("sample")

        return y

    def predict_logits(
        self,
        x_feature: dict[str, Any],
        x_history: jax.Array,
    ) -> jax.Array:
        x = (self.feature_processor.transform(x_feature), x_history)
        return self(x)

    def predict(
        self,
        x_feature: dict[str, Any],
        x_history: jax.Array,
        key: PRNGKeyArray,
    ) -> OptimizationStep:
        logits = self.predict_logits(x_feature, x_history)
        action = jr.categorical(key, logits)
        step = self.optimizers[int(action)]
        return step

    def reconstruct() -> HistoryModel: ...

    def to_dict(self) -> dict[str, Any]: ...


class LSTM_Encoder(eqx.Module):
    """
    LSTM Encoder module.

    Parameters
    ----------
    in_size : int
        Input size of the LSTM encoder.
    hidden_size : int
        Hidden size of the LSTM encoder.
    key : jax.Array
        Random key for initializing the LSTM cell.

    Attributes
    ----------
    hidden_size : int
        Hidden size of the LSTM encoder.
    cell : eqx.nn.LSTMCell
        LSTM cell used in the encoder.
    linear : eqx.nn.Linear
        Linear layer applied to the final hidden state.
    """

    hidden_size: int
    out_size: int
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, *, key):
        self.hidden_size = hidden_size
        self.out_size = out_size
        # self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=key)

    def __call__(self, x):
        """
        Encode sequence using LSTM.

        Parameters
        ----------
        x : array-like
            Input sequence to encode.
        init_state : array-like
            Initial hidden state for LSTM.

        Returns
        -------
        array-like
            Final hidden state after encoding sequence.
        """

        def scan_fn(state, input):
            return (self.cell(input, state), None)

        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

        (h, c), _ = jax.lax.scan(scan_fn, init_state, x)
        h = self.linear(h)
        return h


# =============================================================================


class L2COLoader:
    """
    Utility class for loading and storing Equinox models.
    """

    @staticmethod
    def load(path: str) -> L2COModel:
        """
        Load model from file.

        Parameters
        ----------
        path : str
            Path to model file (without extension).

        Returns
        -------
        eqx.Module
            Loaded model instance.
        """
        _path = Path(path)
        key = jr.key(0)

        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            if hyperparams["type"] == "equinox":
                model = make(key=key, **hyperparams)
                return eqx.tree_deserialise_leaves(f, model)
            elif hyperparams["type"] == "scikit":
                hyperparams["model"] = joblib.load(hyperparams["model"])
                model = make(key=key, **hyperparams)
                return model
            else:
                raise ValueError(f"Unknown model type {hyperparams['type']}")

    @staticmethod
    def store(object: L2COModel, path: str):
        """
        Store model to file.

        Parameters
        ----------
        object : AbstractModel
            Model instance to store.
        path : str
            Path to save model (without extension).

        Returns
        -------
        str
            Path to saved model file.
        """
        _path = Path(path)
        hyperparams = object.to_dict()
        hyperparams["model_type"] = object.__class__.__name__.lower()

        if hyperparams["type"] == "scikit":
            joblib_path = _path.with_suffix(".joblib")
            joblib.dump(object.model, filename=joblib_path)
            hyperparams["model"] = str(joblib_path)

        with open(_path.with_suffix(".eqx"), "wb") as f:
            hyperparams_str = json.dumps(hyperparams)
            f.write((hyperparams_str + "\n").encode())
            if hyperparams["type"] == "equinox":
                eqx.tree_serialise_leaves(f, object)

        return str(_path)


# =============================================================================


def make(*, model_type: str, **kwargs) -> AbstractModel:
    """
    Factory function to create policy models.

    Parameters
    ----------
    model_type : str
        Type of model to create.
    **kwargs
        Model-specific configuration parameters.

    Returns
    -------
    AbstractModel
        Instantiated policy model.

    Raises
    ------
    ValueError
        If model_type is unknown.
    """
    match model_type:
        case "mlp":
            return MLP.reconstruct(**kwargs)
        case "l2comodel":
            return L2COModel.reconstruct(**kwargs)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================


def train_scikit(
    model: BaseEstimator,
    x: jax.Array,
    y: jax.Array,
    **kwargs,
) -> tuple[BaseEstimator, dict]:
    model.fit(x, y, **kwargs)
    return model, {}


def predict_scikit(
    model: BaseEstimator,
    x: jax.Array,
) -> jax.Array:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y_pred: np.ndarray = model.predict(x)
    return jnp.array(y_pred)


def predict_equinox(
    model: eqx.Module,
    x: jax.Array,
) -> jax.Array:
    logits = model(x)
    return logits


class MLP(AbstractModel):
    mixed_embedding: MixedEmbedding
    model: eqx.nn.MLP

    def __init__(
        self,
        mixed_embedding: MixedEmbedding,
        model: eqx.Module,
    ):
        self.mixed_embedding = mixed_embedding
        self.model = model

    def __call__(
        self,
        x: dict[str, Any],
    ) -> jax.Array:
        return self.model(self.mixed_embedding(x))

    def to_dict(self) -> dict[str, Any]:
        match self.model.activation:
            case jax.nn.relu:
                activation_str = "relu"
            case jax.nn.tanh:
                activation_str = "tanh"
        return {
            "model_type": self.__class__.__name__.lower(),
            "mixed_embedding": self.mixed_embedding.to_dict(),
            "model": {
                "in_size": self.model.in_size,
                "out_size": self.model.out_size,
                "width_size": self.model.width_size,
                "depth": self.model.depth,
                "activation": activation_str,
            },
        }

    @classmethod
    def reconstruct(
        cls,
        mixed_embedding: dict,
        model: dict,
        key: PRNGKeyArray,
        **kwargs,
    ) -> MLP:
        mixed_embedding = MixedEmbedding.reconstruct(
            **mixed_embedding, key=key
        )

        match model["activation"]:
            case "relu":
                model["activation"] = jax.nn.relu
            case "tanh":
                model["activation"] = jax.nn.tanh

        mlp_model = eqx.nn.MLP(
            **model,
            key=key,
        )

        return cls(
            mixed_embedding=mixed_embedding,
            model=mlp_model,
        )


def mse_loss(
    model: AbstractModel, x: dict[str, Any], y: jax.Array
) -> jax.Array:
    """Mean squared error between model predictions and targets."""
    preds = eqx.filter_vmap(model)(x)
    return jnp.mean((preds - y) ** 2)


# def train_equinox(
#     model: AbstractModel,
#     x: jax.Array,
#     y: jax.Array,
#     optimizer: optax.GradientTransformation,
#     num_steps: int = 100,
#     verbose: bool = False,
# ) -> tuple[AbstractModel, list[float]]:
#     """Train the model with Adam on an iterable of (x, y) batches."""
#     opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
#     loss_and_grad = eqx.filter_value_and_grad(mse_loss)

#     loss_history: list[float] = []

#     for _ in tqdm(range(num_steps), disable=not verbose):
#         loss, grads = loss_and_grad(model, x, y)
#         updates, opt_state = optimizer.update(grads, opt_state, model)
#         model = eqx.apply_updates(model, updates)

#         loss_history.append(float(loss))

#     return model, loss_history


def train_equinox(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    train_data: tuple[jax.Array, jax.Array],
    val_data: tuple[jax.Array, jax.Array] | None = None,  # Optional validation
    num_steps: int = 100,
    verbose: bool = False,
    scan: bool = True,
) -> tuple[eqx.Module, dict[str, jax.Array]]:
    x_train, y_train = train_data

    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    loss_fn = mse_loss
    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def make_step(carry, _):
        params, opt_state = carry
        model = eqx.combine(params, static)

        # Training Branch
        train_loss, grads = loss_and_grad_fn(model, *train_data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)

        # Validation Branch (Conditional)
        val_loss = (
            loss_fn(model, *val_data) if val_data is not None else jnp.nan
        )

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        return (new_params, opt_state), metrics

    # 2. Execution Logic (Scan vs Loop)
    if scan:
        step_fn = scan_tqdm(num_steps)(make_step) if verbose else make_step
        (params, opt_state), metrics = jax.lax.scan(
            step_fn,
            (params, opt_state),
            xs=jnp.arange(num_steps),
            length=num_steps,
        )
    else:
        metrics_list = []
        for _ in tqdm(range(num_steps), disable=not verbose):
            (params, opt_state), step_metrics = make_step(
                (params, opt_state), None
            )
            metrics_list.append(step_metrics)
        metrics = {
            k: jnp.stack([m[k] for m in metrics_list]) for k in metrics_list[0]
        }

    return eqx.combine(params, static), metrics
