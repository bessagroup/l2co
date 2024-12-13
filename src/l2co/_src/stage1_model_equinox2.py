# Standard
import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xarray as xr
from f3dasm.optimization import Optimizer
from jax.tree_util import PyTreeDef
from optax import GradientTransformation
from tqdm import tqdm

# Local
from .data_loader import OptimizationDynamics
from .stage0_model import Stage0Model

#                                                            Feature extraction
# =============================================================================


class FeatureFn(Protocol):
    def __call__(self, X: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        ...


def convolute(X: jnp.ndarray, population: int = 1):
    return jnp.convolve(X, jnp.ones(population) / float(population),
                        mode='valid')


def polynomial_features(X: jnp.ndarray, n_terms: int = 3) -> jnp.ndarray:
    # Create an array of linearly spaced values in [0, 1]
    X_ = jnp.linspace(0., 1., len(X))

    # Create polynomial features
    def poly_features(x):
        return jnp.array([x ** i for i in range(n_terms + 1)])

    # Generate features for each input X
    poly_features_vmap = jax.vmap(poly_features)
    X_poly = poly_features_vmap(X_)

    # Fit linear regression using normal equations: w=(X.T @ X)^(-1) @ X.T @ y
    y = X  # Target is the original unscaled X
    XTX = jnp.dot(X_poly.T, X_poly)
    XTy = jnp.dot(X_poly.T, X)
    coef = jnp.linalg.solve(XTX, XTy)

    # Calculate predicted values
    y_pred = jnp.dot(X_poly, coef)

    # Calculate R2 score
    residuals = y - y_pred
    ss_res = jnp.sum(residuals ** 2)  # Residual sum of squares
    ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)  # Total sum of squares
    r2 = 1 - ss_res / ss_tot  # R^2 score

    # Set NaN R² to -1 and clamp R² to be between -1 and 1
    capped_r2 = jnp.clip(jnp.nan_to_num(r2, nan=-1.0), -1.0, 1.0)

    # Concatenate coefficients and R2 score into a single array
    result = jnp.concatenate([coef, jnp.array([capped_r2])])

    return result


def quadric_curve(X):
    """
    Fit a quadratic curve (Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0) to a set of
    2D points using Eigenvalue Decomposition.

    Parameters
    ----------
    X : jnp.ndarray
        The input data

    Returns
    -------
    jnp.ndarray
        The coefficients of the quadratic curve

    Notes
    -----
    The quadratic curve is fitted to the data using the following steps:
    1. Construct the design matrix M
    2. Compute the Gram matrix (M^T M)
    3. Perform eigenvalue decomposition on the Gram matrix
    4. The smallest eigenvalue corresponds to the quadratic coefficients
    5. Normalize coefficients to ensure their sum is 1
    """
    x = jnp.linspace(0., 1., len(X))
    y = X

    # Construct the design matrix (as before)
    M = jnp.column_stack([x**2, x*y, y**2, x, y, jnp.ones_like(x)])

    # Compute the Gram matrix (M^T M), which is symmetric
    Gram_matrix = jnp.dot(M.T, M)

    # Perform eigenvalue decomposition on the Gram matrix
    _, eigenvectors = jnp.linalg.eigh(Gram_matrix)

    # The smallest eigenvalue corresponds to the quadratic coefficients
    # The eigenvector corresponding to the smallest eigenvalue
    # Take the eigenvector associated with the smallest eigenvalue
    coeffs = eigenvectors[:, 0]

    # Normalize coefficients to ensure their sum is 1
    coeffs /= jnp.sum(coeffs)

    return coeffs

# Loss function
# =============================================================================


class LossFn(Protocol):
    def __call__(self, model: eqx.Module, *args, **kwargs) -> jnp.ndarray:
        ...


def mse_l2_reg(model: eqx.Module, X: jnp.ndarray, Y: jnp.ndarray,
               lambda_reg: float = 1e-4) -> jnp.ndarray:
    y_pred = jax.vmap(model)(X)
    mse_loss = jnp.mean((y_pred - Y) ** 2)

    # Corrected L2 regularization using tree_leaves
    # L2 regularization over weights of layers in model.mlp
    l2_reg = sum(jnp.sum(jnp.square(layer.weight))
                 for layer in model.mlp.layers if hasattr(layer, 'weight'))

    total_loss = mse_loss + lambda_reg * l2_reg  # Add L2 regularization

    return total_loss


def mse(model: eqx.Module, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    y_pred = jax.vmap(model)(X)
    return jnp.mean((y_pred - Y) ** 2)


#                                                              MLP with dropout
# =============================================================================

class MLPWithDropout(eqx.Module):
    layers: list
    dropout: list
    key: jax.random.KeyArray
    dropout_key: jax.random.KeyArray

    def __init__(self, in_size: int, out_size: int,
                 width_size: int, depth: int,
                 dropout_rate: float, key: jax.random.KeyArray):
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
                 key: Optional[jax.random.KeyArray] = None) -> jnp.ndarray:
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

#                                                                         Model
# =============================================================================


class MixedModel(eqx.Module):
    # mlp: eqx.nn.MLP
    mlp: eqx.Module
    feature_fn: FeatureFn
    key: jax.random.KeyArray

    def __init__(self, width: int, depth: int, n_terms: int, n_optimizers: int,
                 feature_fn: FeatureFn, key: jax.random.KeyArray,
                 ) -> None:
        self.key = key
        # self.feature_fn = partial(feature_fn, n_terms=n_terms)
        self.feature_fn = jax.jit(feature_fn)
        self.mlp = MLPWithDropout(
            in_size=n_terms, out_size=n_optimizers,
            width_size=width, depth=depth,
            dropout_rate=0.1, key=key)
        # self.mlp = eqx.nn.MLP(in_size=n_terms, out_size=n_optimizers,
        #                       width_size=width, depth=depth,
        #                       key=key)

    def __call__(self, X: jnp.ndarray,
                 key: Optional[jax.random.KeyArray] = None) -> jnp.ndarray:
        X_features = self.feature_fn(X)
        return self.mlp(x=X_features, inference=False, key=key)

#                                                                 Training step
# =============================================================================


@eqx.filter_jit
def train_step(model: eqx.Module, opt_state: PyTreeDef,
               optimizer: optax.GradientTransformation,
               loss_fn: LossFn, *args, **kwargs
               ) -> Tuple[jnp.ndarray, eqx.Module, PyTreeDef]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *args, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_loop(
        model: eqx.Module, optimizer: optax.GradientTransformation,
        X: jnp.ndarray, Y: jnp.ndarray, n_epochs: int,
        loss_fn: LossFn, key: jax.random.KeyArray,
        verbose: bool = False) -> eqx.Module:

    params = eqx.filter(model, eqx.is_inexact_array)

    # Define the optimizer (Adam)
    opt_state = optimizer.init(params)

    # Regularization strength (you can tune this hyperparameter)
    update = partial(train_step, optimizer=optimizer,
                     loss_fn=loss_fn)

    if verbose:
        print(f"Training stage1 model for {n_epochs} epochs...")
        iterator = tqdm(range(n_epochs))

    else:
        iterator = range(n_epochs)

    for epoch in iterator:
        loss, model, opt_state = update(
            X=X, Y=Y, model=model, opt_state=opt_state)

    return model

#                                                                  Stage1_model
# =============================================================================


class Stage1Model():
    def __init__(self, optimizers: Dict[str, Optimizer],
                 model_stage1: MixedModel, model_stage0: Stage0Model,
                 **kwargs):
        """
        Stage1 model

        Parameters
        ----------
        optimizers : xarray.DataArray
            The optimizers
        model : SingleStage1Model
            The model
        t_start : int
            The start time
        t_switch : int
            The switch time
        rel_tolerance : float
            The relative tolerance
        name : str, optional
            The name of the model, by default 'stage1_model'
        """
        self.optimizers = optimizers
        self.models_stage1 = self.create_models_stage1(model_stage1)
        self.models_stage0 = self.create_models_stage0(model_stage0)

    def train(self, optimization_dynamics: Dict[str, OptimizationDynamics],
              n_epochs: int,
              optimizer: GradientTransformation,
              seed: int, save_model: bool = False,
              verbose: bool = False):
        """
        Train the model

        Parameters
        ----------
        """
        for opt_name, opt_data in optimization_dynamics.items():
            self.models_stage0[opt_name].train(opt_data.to_feature_data())

        for opt_name, opt_data in optimization_dynamics.items():
            _X = jnp.array(opt_data.X_trajectory)
            X = jax.vmap(
                partial(convolute,
                        population=self.optimizers[opt_name]._population))(
                jnp.atleast_2d(_X))

            Y = jnp.array(opt_data.Y)

            self.models_stage1[opt_name] = train_loop(
                model=self.models_stage1[opt_name],
                optimizer=optimizer,
                X=X,
                Y=Y,
                n_epochs=n_epochs,
                loss_fn=partial(
                    mse_l2_reg, lambda_reg=1e-3),
                key=jax.random.PRNGKey(
                    seed),
                verbose=verbose)

    def predict(self, optimization_dynamics: Dict[str, OptimizationDynamics],
                stage0_predictions: xr.DataArray) -> xr.DataArray:
        """
        Predict the best optimizer on the testing data

        Parameters
        ----------
        data_test : Dict[str, OptimizationDynamics]
            The testing data optimizationdynamics
        stage0_predictions : xarray.DataArray
            The predictions of the Stage0 model

        Returns
        -------
        xarray.DataArray
            The predictions of the Stage
        """
        ...
        list_y_pred_stage1 = []

        for opt_name, opt_data in optimization_dynamics.items():
            problems = stage0_predictions.where(
                stage0_predictions == opt_name, drop=True).problem

            if len(problems) == 0:
                continue

            _X = jnp.array(opt_data.select_problems(problems).X_trajectory)
            X = jax.vmap(
                partial(convolute,
                        population=self.optimizers[opt_name]._population))(
                jnp.atleast_2d(_X))

            y_pred = jax.vmap(self.models_stage1[opt_name])(X)

            y_pred_arr = xr.DataArray(
                y_pred, coords={'problem': problems,
                                'optimizer': opt_data._Y.optimizer}
            ).idxmin('optimizer')

            predictions = xr.apply_ufunc(
                partial(concat_with_plus, first_opt=opt_name),
                y_pred_arr,
                vectorize=True,
                dask='parallelized',
                output_dtypes=[object],
            )

            list_y_pred_stage1.append(predictions)

        return xr.concat(list_y_pred_stage1, dim='problem')

    def create_models_stage0(
            self, model: Stage0Model) -> Dict[str, Stage0Model]:
        """
        Create the Stage0 models

        Returns
        -------
        Dict[str, Stage0Model]
            The Stage0 models
        """
        return {optimizer_name: deepcopy(model)
                for optimizer_name in self.optimizers.keys()}

    def create_models_stage1(
            self, model: MixedModel) -> Dict[str, eqx.Module]:
        """
        Create the Stage1 models

        Returns
        -------
        Dict[str, SingleStage1Model]
            The Stage1 models

        Notes
        -----
        The Stage1 models are created for each optimizer in the optimizers
        """

        return {optimizer_name: deepcopy(model)
                for optimizer_name in self.optimizers.keys()}


def load(path: Path) -> Stage1Model:
    """
    Load the model from a file

    Parameters
    ----------
    path : Path
        The path to the model
    """
    filename = (path / 'stage1_model').with_suffix('.pkl')
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model


def concat_with_plus(pred_value: str, first_opt: str):
    if str(pred_value) == 'nan':
        return str(first_opt)

    if first_opt == pred_value:
        return str(first_opt)
    return str(first_opt) + '+' + str(pred_value)
