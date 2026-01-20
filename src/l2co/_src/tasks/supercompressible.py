from functools import partial
from typing import Literal

import gpjax as gpx
import jax.numpy as jnp
from f3dasm import ExperimentData
from jaxtyping import Array, Float, PRNGKeyArray

from ..task import Task


def loss_fn(
    x: Float[Array, " d"],
    key: PRNGKeyArray,
    opt_posterior,
    train_data: gpx.Dataset,
) -> Float[Array, ""]:
    latent_dist = opt_posterior.predict(
        x.reshape(1, -1), train_data=train_data
    )
    predictive_dist = opt_posterior.likelihood(latent_dist)
    return predictive_dist.sample(key)[0]


def train_gp(D: gpx.Dataset):
    kernel = gpx.kernels.RBF(n_dims=D.in_dim)
    mean_function = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

    posterior = prior * likelihood

    opt_posterior, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
        trainable=gpx.parameters.Parameter,
        max_iters=500,
        verbose=False,
    )
    return opt_posterior


def create_supercompressible_task(
    experiment_data_path: str, output_name: Literal["energy", "sigma_crit"]
) -> Task:
    match output_name:
        case "energy":
            output_index = 1
        case "sigma_crit":
            output_index = 2
        case _:
            raise ValueError(f"Unknown output: {output_name}")

    experiment_data = ExperimentData.from_file(
        project_dir=experiment_data_path
    )
    tag = {}
    arr_in, arr_out = experiment_data.to_numpy()
    X_raw = jnp.array(arr_in)
    y_raw = jnp.array(arr_out[:, output_index])

    # Remove nans
    mask = (~jnp.isnan(X_raw).any(axis=1)) & (~jnp.isnan(y_raw))
    X = X_raw[mask]
    y = y_raw[mask]

    # Scale inputs

    X_mean = jnp.mean(X, axis=0)
    X_std = jnp.std(X, axis=0)
    X = (X - X_mean) / X_std

    # Scale outputs
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y = (y - y_mean) / y_std

    # Switch sign to make it a minimization problem
    y = -y

    # Create dataset
    D = gpx.Dataset(X, y.reshape(-1, 1))

    opt_posterior = train_gp(D)

    tag["dimensionality"] = X.shape[1]

    # Note: doesnt work with jax.Partial due to the partial argument D being
    # registered as a PyTree

    return Task(
        model=jnp.zeros(X.shape[1]),
        loss_fn=partial(loss_fn, opt_posterior=opt_posterior, train_data=D),
        tag=tag,
        pass_rng=True,
    )
