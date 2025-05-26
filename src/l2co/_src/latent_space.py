# Third-party
from functools import partial

import gpjax as gpx
import jax
import jax.numpy as jnp
import optax
from gpjax.parameters import Real

# =============================================================================

key = jax.random.key(0)

# =============================================================================


class SigmoidMean(gpx.mean_functions.AbstractMeanFunction):
    """
    Sigmoid mean function for Gaussian processes.

    Parameters
    ----------
    a : float
        Scale parameter of the sigmoid function.
    b : float
        Slope parameter of the sigmoid function.
    c : float
        Offset parameter of the sigmoid function.

    Attributes
    ----------
    a : Real
        Scale parameter as a Real object.
    b : Real
        Slope parameter as a Real object.
    c : Real
        Offset parameter as a Real object.
    """

    def __init__(self, a: float, b: float, c: float):
        super().__init__()
        self.a = Real(jnp.array(a))
        self.b = Real(jnp.array(b))
        self.c = Real(jnp.array(c))

    def __call__(self, x):
        return self.a.value / (1.0 + jnp.exp(
            -self.b.value * (x - self.c.value)))


def fit_gpr(
        x: jax.Array,
        optimizer: optax.GradientTransformation,
        inducing_points: int = 50,
        num_iters: int = 100,
        return_posterior: bool = False
) -> jax.Array | tuple[jax.Array, gpx.Posterior]:
    """
    Fits a Gaussian Process Regression (GPR) model.

    Parameters
    ----------
    x : jax.Array
        Input data for the GPR model.
    optimizer : optax.GradientTransformation
        Optimizer for training the GPR model.
    inducing_points : int, optional
        Number of inducing points for the variational approximation.
        Default is 50.
    num_iters : int, optional
        Number of iterations for optimization. Default is 100.
    return_posterior : bool, optional
        If True, returns the posterior along with the parameters.
        Default is False.

    Returns
    -------
    jax.Array or tuple[jax.Array, gpx.Posterior]
        The fitted parameters of the GPR model. If `return_posterior`
        is True, also returns the posterior object.
    """

    yy = x.T[:, None]
    xx = jnp.linspace(0.0, 1.0, x.shape[0])[:, None]

    # Select 100 random points
    # idx = jax.random.choice(key, x.shape[0], shape=(100,), replace=False)

    D = gpx.Dataset(X=xx, y=yy)

    meanf = SigmoidMean(a=1.0, b=1.0, c=1.0)
    kernel = gpx.kernels.RBF()  # 1-dimensional inputs
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    posterior = prior * likelihood

    z = jnp.linspace(0.0, 1.0, inducing_points)[:, None]
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=z
    )

    opt_posterior, _ = gpx.fit(
        model=q,
        objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
        train_data=D,
        optim=optimizer,
        num_iters=num_iters,
        key=key,
        verbose=False,
        safe=False,
        batch_size=16,
    )

    sigmoid_a = opt_posterior.posterior.prior.mean_function.a.value
    sigmoid_b = opt_posterior.posterior.prior.mean_function.b.value
    sigmoid_c = opt_posterior.posterior.prior.mean_function.c.value
    rbf_lengthscale = opt_posterior.posterior.prior.kernel.lengthscale.value
    rbf_variance = opt_posterior.posterior.prior.kernel.variance.value

    params = jnp.array(
        [sigmoid_a, sigmoid_b, sigmoid_c, rbf_lengthscale, rbf_variance])

    if return_posterior:
        return params, opt_posterior
    else:
        return params


def sigmoid_mean(x: jax.Array, a: float, b: float, c: float) -> jax.Array:
    """
    Compute the sigmoid function.

    Parameters
    ----------
    x : jax.Array
        Input data.
    a : float
        Scale parameter of the sigmoid function.
    b : float
        Slope parameter of the sigmoid function.
    c : float
        Offset parameter of the sigmoid function.

    Returns
    -------
    jax.Array
        Output of the sigmoid function.
    """

    return a / (1 + jnp.exp(-b * (x - c)))


def loss_fn(params: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Compute the mean squared error (MSE) loss.

    Parameters
    ----------
    params : jax.Array
        Parameters of the sigmoid function.
    x : jax.Array
        Input data.
    y : jax.Array
        Target data.

    Returns
    -------
    jax.Array
        Computed MSE loss.
    """

    a, b, c = params
    y_pred = sigmoid_mean(x, a, b, c)
    return jnp.mean((y - y_pred) ** 2)


def fit_basis_function(y: jnp.ndarray, max_iter: int = 100,
                       tol: float = 1e-6) -> jnp.ndarray:
    """
    Fit a basis function to the given data.

    Parameters
    ----------
    y : jax.numpy.ndarray
        Target data to fit the basis function to.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.

    Returns
    -------
    jax.numpy.ndarray
        Fitted parameters of the basis function.
    """

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
