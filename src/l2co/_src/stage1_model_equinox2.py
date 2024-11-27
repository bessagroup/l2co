import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Protocol, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from jax.tree_util import PyTreeDef
from optax import GradientTransformation
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from l2co._src.data_loader import OptimizationDynamics

from .stage0_model import Stage0Model

#                                                            Feature extraction
# =============================================================================


class FeatureFn(Protocol):
    def __call__(self, X: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        ...


def convolute(X: jnp.ndarray, population: int = 1):
    return jnp.convolve(X, jnp.ones(population) / float(population), mode='valid')


def polynomial_features(X: jnp.ndarray, n_terms: int = 3) -> jnp.ndarray:
    # Create an array of linearly spaced values in [0, 1]
    X_ = jnp.linspace(0., 1., len(X))

    # Create polynomial features
    def poly_features(x):
        return jnp.array([x ** i for i in range(n_terms + 1)])

    # Generate features for each input X
    poly_features_vmap = jax.vmap(poly_features)
    X_poly = poly_features_vmap(X_)

    # Fit linear regression using normal equations: w = (X.T @ X)^(-1) @ X.T @ y
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


def get_pca_features(X):
    X_ = jnp.array([[i, value]
                   for i, value in zip(jnp.linspace(0., 1., len(X)), X)])
    # Step 1: Standardize the data (optional, but recommended for PCA)
    X_mean = jnp.mean(X_, axis=0)
    X_centered = X_ - X_mean

    # Step 2: Compute the covariance matrix
    covariance_matrix = jnp.cov(X_centered.T)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = jnp.linalg.eigh(covariance_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = jnp.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Step 5: Get the explained variance ratio (eigenvalue / total sum of eigenvalues)
    explained_variance_ratio = eigvals_sorted / jnp.sum(eigvals_sorted)

    # Step 6: Return the explained variance ratio and the principal components (top 2)
    return jnp.concatenate([explained_variance_ratio[:2], eigvecs_sorted[:, :2].flatten()])


def fit_fourier_series(X, num_features, return_reconstructed=False):
    """
    Process a single sequence, standardize, compute the FFT, and reconstruct with reduced dimensionality.

    Args:
    - X: Input array with shape (sequence_length,).
    - num_features: Number of features to retain from the FFT.

    Returns:
    - Standardized FFT coefficients and optionally the reconstructed signal.
    """

    # Standardize the data: mean 0, variance 1
    X_mean = jnp.mean(X)
    X_std = jnp.std(X)
    X_scaled = (X - X_mean) / (X_std + 1e-6)

    # Compute the FFT
    X_fft = jnp.fft.rfft(X_scaled)

    # Truncate FFT coefficients
    X_fft_truncated = X_fft[:num_features].real

    if return_reconstructed:
        # Reconstruct the signal using inverse FFT
        X_fft_reconstructed = jnp.fft.irfft(X_fft_truncated, n=X.shape[-1])
        return X_fft_truncated, X_fft_reconstructed * X_std + X_mean

    return X_fft_truncated


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


#                                                                         Model
# =============================================================================

class MixedModel(eqx.Module):
    mlp: eqx.nn.MLP
    feature_fn: FeatureFn
    key: jax.random.KeyArray

    def __init__(self, width: int, depth: int, n_terms: int, n_optimizers: int,
                 feature_fn: FeatureFn, key: jax.random.KeyArray,
                 ) -> None:
        self.key = key
        self.feature_fn = partial(feature_fn, n_terms=n_terms)
        self.mlp = eqx.nn.MLP(in_size=n_terms+2, out_size=n_optimizers,
                              width_size=width, depth=depth,
                              key=key)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        X_features = self.feature_fn(X)
        return self.mlp(X_features)

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
    def __init__(self, optimizers: xr.DataArray,
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
            X = jax.vmap(partial(convolute, population=1))(_X)

            Y = jnp.array(opt_data.Y)

            self.models_stage1[opt_name] = train_loop(model=self.models_stage1[opt_name],
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
        # list_y_pred_stage1 = []
        # stage0_predictions = stage0_predictions.sortby('problem')

        # for optimizer in self.optimizers.values:
        #     problems = stage0_predictions.where(
        #         stage0_predictions == optimizer, drop=True).problem

        #     if len(problems) == 0:
        #         continue

        #     data_optimization = data_test[optimizer].select_problems(problems)

        #     # data_optimization = data_selected.to_learning_data(
        #     #     optimizer=optimizer, t=self.t_switch,
        #     #     t_start=self.t_start
        #     # )

        #     y_features = self.models_stage0[optimizer].clf.predict(
        #         data_optimization._X_features.to_pandas())

        #     logits = self.states[optimizer].apply_fn(
        #         {'params': self.states[optimizer].params,
        #          'batch_stats': self.states[optimizer].batch_stats},
        #         jnp.array(y_features), data_optimization.X_trajectory,
        #         train=False)

        #     predictions = xr.DataArray(
        #         np.array(logits),
        #         dims=['problem', 'optimizer'],
        #         coords={'optimizer': self.optimizers,
        #                 'problem': data_optimization.problem}
        #     )

        #     list_y_pred_stage1.append(predictions)

        # stage1_predictions = xr.concat(
        #     list_y_pred_stage1, dim='problem').sortby('problem')

        # # Select the optimizer with the highest probability
        # # Select the optimizer with the best performance value
        # stage1_predictions = stage1_predictions.idxmin(
        #     dim='optimizer').sortby('problem')

        # def concat_with_plus(opt1, opt2):

        #     if str(opt2) == 'nan':
        #         return str(opt1)

        #     if opt1 == opt2:
        #         return str(opt1)
        #     return str(opt1) + '+' + str(opt2)

        # combined_predictions = xr.apply_ufunc(
        #     concat_with_plus, stage0_predictions, stage1_predictions,
        #     vectorize=True,
        #     dask='parallelized',
        #     output_dtypes=[object])

        # return combined_predictions.sortby('problem')

    def create_models_stage0(
            self, model: Stage0Model) -> Dict[str, Stage0Model]:
        """
        Create the Stage0 models

        Returns
        -------
        Dict[str, Stage0Model]
            The Stage0 models
        """
        return {optimizer: deepcopy(model)
                for optimizer in self.optimizers.values}

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

        return {optimizer: deepcopy(model)
                for optimizer in self.optimizers.values}

    # def create_states(self) -> Dict[str, TrainState]:
    #     """
    #     Create the TrainStates for the models

    #     Returns
    #     -------
    #     Dict[str, TrainState]
    #         The TrainStates for the models
    #     """
    #     return {optimizer: None for optimizer in self.optimizers.values}

    # def save(self, path: Path) -> None:
    #     """
    #     Save the model to a file

    #     Parameters
    #     ----------
    #     path : Path
    #         The path to the model
    #     """
    #     for optimizer, model in self.models_stage1.items():
    #         model.save(Path(path) / f"stage1_{optimizer}")


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
