import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from gpjax.parameters import Real

from .performance import output_cummin

key = jax.random.key(0)


class SigmoidMean(gpx.mean_functions.AbstractMeanFunction):
    def __init__(self, a: float, b: float, c: float):
        super().__init__()
        self.a = Real(jnp.array(a))
        self.b = Real(jnp.array(b))
        self.c = Real(jnp.array(c))

    def __call__(self, x):
        return self.a.value / (1.0 + jnp.exp(-self.b.value * (x - self.c.value)))


def fit_gpr(
        x: jax.Array, optimizer: optax.GradientTransformation,
        inducing_points: int = 50, num_iters: int = 100,
        batch_size: int = 16) -> jax.Array:

    yy = x.T[:, None]
    xx = jnp.linspace(0.0, 1.0, x.shape[0])[:, None]

    z = jnp.linspace(0.0, 1.0, inducing_points)[:, None]

    D = gpx.Dataset(X=xx, y=yy)

    meanf = SigmoidMean(a=1.0, b=1.0, c=1.0)
    kernel = gpx.kernels.RBF()  # 1-dimensional inputs
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    posterior = prior * likelihood

    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=z
    )

    opt_posterior, history = gpx.fit(
        model=q,
        objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
        train_data=D,
        optim=optimizer,
        num_iters=num_iters,
        key=key,
        verbose=False,
    )

    sigmoid_a = opt_posterior.posterior.prior.mean_function.a.value
    sigmoid_b = opt_posterior.posterior.prior.mean_function.b.value
    sigmoid_c = opt_posterior.posterior.prior.mean_function.c.value
    rbf_lengthscale = opt_posterior.posterior.prior.kernel.lengthscale.value
    rbf_variance = opt_posterior.posterior.prior.kernel.variance.value

    return jnp.array(
        [sigmoid_a, sigmoid_b, sigmoid_c, rbf_lengthscale, rbf_variance])


def log_relative_improvement(data: xr.Dataset) -> xr.DataArray:
    def compute_log_relative_improvement(x, b):
        impv = -(x[0] - x[-1]) / (b[0] - b[-1]) / (abs(x[0]) + 1e-8)
        t = np.log(impv + 1e-8)
        return np.atleast_1d(t)

    return xr.apply_ufunc(
        compute_log_relative_improvement,
        output_cummin(data.output_min),
        data.budget,
        input_core_dims=[["budget"], ["budget"]],
        output_core_dims=[["log_relative_improvement"]],
        vectorize=True,
    )
