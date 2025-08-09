# Standard
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from f3dasm import Block, ExperimentData
from f3dasm.design import Domain
from jax import Array
from jaxtyping import PyTree

# Local
from ..task import Task
from .models import mlp

# =============================================================================


def mse(x: Array, y: Array) -> Array:
    """
    Compute the mean squared error between two arrays.

    Parameters
    ----------
    x : Array
        Predicted values.
    y : Array
        Ground truth values.

    Returns
    -------
    Array
        Mean squared error.
    """
    return jnp.mean((x - y) ** 2)


def convection_pde_loss(
        model: PyTree,
        x_res: Array,
        x_ic: Array,
        x_bc_left: Array,
        x_bc_right: Array,
        t_res: Array,
        t_ic: Array,
        t_bc: Array,
        loss_fn_res: Callable[[Array, Array], Array],
        loss_fn_bc: Callable[[Array, Array], Array],
        loss_fn_ic: Callable[[Array, Array], Array],
        beta: float) -> jnp.ndarray:

    # Vectorize the model, since it processes only one point
    # Also note that the model takes in both x and t as input

    def u(x, t):
        return eqx.filter_vmap(model)(jnp.concatenate([x, t], axis=-1))

    # Compute predictions
    u_ic = u(x_ic, t_ic)

    u_bc_left = u(x_bc_left, t_bc)
    u_bc_right = u(x_bc_right, t_bc)

    # Compute first-order derivatives using jacobian reverse mode
    du_dxt = jax.jacrev(lambda x, t: u(x, t).sum(),
                        argnums=(0, 1))  # Computes [∂u/∂x, ∂u/∂t]

    u_x, u_t = du_dxt(x_res, t_res)

    # PDE Residual Loss: u_t + beta * u_x = 0
    loss_res = loss_fn_res(u_t + beta * u_x, jnp.zeros_like(u_t))

    # BC loss: u(0, t) = u(2π, t)
    loss_bc = loss_fn_bc(u_bc_left - u_bc_right, jnp.zeros_like(u_bc_left))

    # IC loss: Enforcing u(x,0) = sin(x)
    loss_ic = loss_fn_ic(u_ic, jnp.sin(x_ic))
    # Combine losses
    total_loss = loss_res + loss_bc + loss_ic
    return total_loss


# =============================================================================

def reaction_pde_loss(
        model: PyTree,
        x_res: Array,
        x_ic: Array,
        x_bc_left: Array,
        x_bc_right: Array,
        t_res: Array,
        t_ic: Array,
        t_bc: Array,
        loss_fn_res: Callable[[Array, Array], Array],
        loss_fn_bc: Callable[[Array, Array], Array],
        loss_fn_ic: Callable[[Array, Array], Array],
        rho: float) -> jnp.ndarray:

    # Vectorize the model
    def u(x, t):
        return eqx.filter_vmap(model)(jnp.concatenate([x, t], axis=-1))

    # Compute predictions
    u_ic = u(x_ic, t_ic)
    u_bc_left = u(x_bc_left, t_bc)
    u_bc_right = u(x_bc_right, t_bc)

    # Compute gradients for PDE residual
    u_t = jax.grad(lambda t: jnp.sum(u(x_res, t)))(t_res)
    u_res = u(x_res, t_res)

    # Nonlinear reaction term
    reaction_term = rho * u_res * (1 - u_res)

    # PDE Residual Loss: u_t - rho * u * (1 - u) = 0
    loss_res = loss_fn_res(u_t - reaction_term, jnp.zeros_like(u_t))

    # BC loss: u(0, t) = u(2π, t)
    loss_bc = loss_fn_bc(u_bc_left - u_bc_right, jnp.zeros_like(u_bc_left))

    # IC loss: u(x, 0) = exp(-((x - π)^2) / (2 * (π/4)^2))
    loss_ic = loss_fn_ic(
        u_ic, jnp.exp(-((x_ic - jnp.pi)**2) / (2 * (jnp.pi / 4)**2)))

    # Combine losses
    total_loss = loss_res + loss_bc + loss_ic
    return total_loss


def wave_pde_loss(
        model: PyTree,
        x_res: Array,
        x_ic: Array,
        x_bc_left: Array,
        x_bc_right: Array,
        t_res: Array,
        t_ic: Array,
        t_bc: Array,
        loss_fn_res: Callable[[Array, Array], Array],
        loss_fn_bc: Callable[[Array, Array], Array],
        loss_fn_ic: Callable[[Array, Array], Array],
        loss_fn_ic_dt: Callable[[Array, Array], Array],
        beta: float) -> jnp.ndarray:

    # Vectorize the model
    def u(x, t):
        return eqx.filter_vmap(model)(jnp.concatenate([x, t], axis=-1))

    # Compute predictions
    u_ic = u(x_ic, t_ic)
    u_bc_left = u(x_bc_left, t_bc)
    u_bc_right = u(x_bc_right, t_bc)

    # Compute first-order derivatives
    u_t = jax.grad(lambda t: u(x_res, t).sum())
    u_x = jax.grad(lambda x: u(x, t_res).sum())

    # Compute second-order derivatives
    u_tt = jax.grad(lambda t: u_t(t).sum())(t_res)
    u_xx = jax.grad(lambda x: u_x(x).sum())(x_res)

    u_ic_dt = (jax.grad(lambda t: u(x_ic, t).sum()))(t_ic)

    # PDE Residual Loss: u_tt - 4 * u_xx = 0
    loss_res = loss_fn_res(u_tt - 4 * u_xx, jnp.zeros_like(u_tt))

    # BC loss: u(0, t) = u(1, t) = 0
    loss_bc = loss_fn_bc(u_bc_left - u_bc_right, jnp.zeros_like(u_bc_left))

    # IC loss: u(x, 0) = sin(πx) + (1/2) sin(βπx)
    loss_ic = loss_fn_ic(
        u_ic,
        jnp.sin(jnp.pi * x_ic) + 0.5 * jnp.sin(beta * jnp.pi * x_ic))

    # Initial velocity condition: du/dt(x,0) = 0

    loss_ic_dt = loss_fn_ic_dt(u_ic_dt, jnp.zeros_like(u_ic_dt))

    # Combine losses
    total_loss = loss_res + loss_bc + loss_ic + loss_ic_dt
    return total_loss


def create_points_dataset(
    x_range: tuple[float], t_range: tuple[float], xgrid_resolution: int,
    tgrid_resolution: int, num_ic_points: int, num_bc_points: int,
    num_res_points: int, *, key: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """
    Create a dataset of points for PDE tasks.

    Parameters
    ----------
    x_range : Tuple[float]
        Range of x values.
    t_range : Tuple[float]
        Range of t values.
    xgrid_resolution : int
        Resolution of the x grid.
    tgrid_resolution : int
        Resolution of the t grid.
    num_ic_points : int
        Number of initial condition points.
    num_bc_points : int
        Number of boundary condition points.
    num_res_points : int
        Number of residual points.
    key : jnp.ndarray
        Random key for reproducibility.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing the generated points.
    """
    x_res_grid = jnp.linspace(x_range[0], x_range[1], xgrid_resolution)
    t_res_grid = jnp.linspace(t_range[0], t_range[1], tgrid_resolution)

    total_points = xgrid_resolution * tgrid_resolution
    indices_res = jr.choice(key, a=total_points,
                            shape=(num_res_points,), replace=False)

    t_indices = indices_res // xgrid_resolution
    x_indices = indices_res % xgrid_resolution

    x_res = x_res_grid[x_indices].reshape(-1, 1)
    t_res = t_res_grid[t_indices].reshape(-1, 1)

    # Initial conditions points
    x_ic = jnp.linspace(x_range[0], x_range[1], num_ic_points).reshape(-1, 1)
    t_ic = jnp.full_like(x_ic, t_range[0])

    # Boundary conditions points
    t_bc = jnp.linspace(t_range[0], t_range[1], num_bc_points).reshape(-1, 1)
    x_bc_left = jnp.full_like(t_bc, x_range[0])
    x_bc_right = jnp.full_like(t_bc, x_range[1])

    # Create the data dictionary
    return {
        "x_res": x_res,
        "t_res": t_res,
        "x_ic": x_ic,
        "t_ic": t_ic,
        "x_bc_left": x_bc_left,
        "x_bc_right": x_bc_right,
        "t_bc": t_bc,
    }


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


def create_pde_task(
    pde_task_name: str,
    seed: int,
    dataset_path: str,
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    xgrid_resolution: int,
    tgrid_resolution: int,
    num_ic_points: int,
    num_bc_points: int,
    num_res_points: int,
    hidden_size: int,
    beta: float = jnp.nan,
    rho: float = jnp.nan
) -> Task:
    """
    Create a PDE task for optimization.

    Parameters
    ----------
    pde_task_name : str
        Name of the PDE task (e.g., 'convection', 'reaction', 'wave').
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path to save the generated dataset.
    x_range : Tuple[float, float]
        Range of x values for the PDE.
    t_range : Tuple[float, float]
        Range of t values for the PDE.
    xgrid_resolution : int
        Resolution of the x grid.
    tgrid_resolution : int
        Resolution of the t grid.
    num_ic_points : int
        Number of initial condition points.
    num_bc_points : int
        Number of boundary condition points.
    num_res_points : int
        Number of residual points.
    hidden_size : int
        Hidden size of the neural network model.
    beta : float, optional
        Parameter for the convection or wave PDE, by default jnp.nan.
    rho : float, optional
        Parameter for the reaction PDE, by default jnp.nan.

    Returns
    -------
    Task
        The created PDE task.
    """
    tag = {}

    _path = Path(dataset_path + f'_{seed}').with_suffix('.npz')

    key = jr.key(int(seed))

    model_key, dataset_key = jr.split(key)

    tag['task_name'] = pde_task_name

    if pde_task_name == 'convection':
        tag['beta'] = beta

        loss_fn = partial(convection_pde_loss, loss_fn_res=mse,
                          loss_fn_bc=mse, loss_fn_ic=mse, beta=beta)
        tag['loss_fn'] = 'convection_pde_loss'

    elif pde_task_name == 'reaction':
        tag['rho'] = rho

        loss_fn = partial(reaction_pde_loss, loss_fn_res=mse,
                          loss_fn_bc=mse, loss_fn_ic=mse, rho=rho)
        tag['loss_fn'] = 'reaction_pde_loss'

    elif pde_task_name == 'wave':
        tag['beta'] = beta

        loss_fn = partial(wave_pde_loss, loss_fn_res=mse,
                          loss_fn_bc=mse, loss_fn_ic=mse,
                          loss_fn_ic_dt=mse, beta=beta)
        tag['loss_fn'] = 'wave_pde_loss'

    model, model_tags = mlp(
        in_size=2, out_size=1, hidden_size=hidden_size,
        num_layers=3, hidden_activation=jax.nn.tanh,
        key=model_key)

    # Count trainable parameters
    num_params = sum(p.size for p in jtu.tree_leaves(
        model) if isinstance(p, jnp.ndarray))
    tag["dimensionality"] = num_params

    if not _path.exists():
        dataset = create_points_dataset(
            x_range=x_range, t_range=t_range,
            xgrid_resolution=xgrid_resolution,
            tgrid_resolution=tgrid_resolution,
            num_ic_points=num_ic_points,
            num_bc_points=num_bc_points,
            num_res_points=num_res_points,
            key=key)
        save_dataset(dataset, _path)

    return Task(
        model=model,
        loss_fn=loss_fn,
        dataset={'dataset_path': _path,
                 'batch_size': None,
                 'seed': seed},
        tag=tag
    )


class PDETaskSampler(Block):
    def __init__(self, num_ic_points: int = 257, num_bc_points: int = 101,
                 num_res_points: int = 10000, xgrid_resolution: int = 255,
                 tgrid_resolution: int = 101):
        self.num_ic_points = num_ic_points
        self.num_bc_points = num_bc_points
        self.num_res_points = num_res_points
        self.xgrid_resolution = xgrid_resolution
        self.tgrid_resolution = tgrid_resolution

        self.convection_task = dict(
            pde_task_name='convection',
            dataset_path='./data/convection',
            x_range=(0., 2. * jnp.pi), t_range=(0., 1.),
            beta=40)

        self.reaction_task = dict(pde_task_name='reaction',
                                  dataset_path='./data/reaction',
                                  x_range=(0., 2. * jnp.pi), t_range=(0., 1.),
                                  rho=5)

        self.wave_task = dict(pde_task_name='wave',
                              dataset_path='./data/wave',
                              x_range=(0., 1.), t_range=(0., 1.),
                              beta=5)

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        hidden_size = data.domain.input_space['hidden_size'].categories

        task_list = []

        for task in [self.convection_task, self.reaction_task, self.wave_task]:
            for h in hidden_size:
                t = deepcopy(task)
                t.update(
                    dict(
                        seed=seed,
                        xgrid_resolution=self.xgrid_resolution,
                        tgrid_resolution=self.tgrid_resolution,
                        num_ic_points=self.num_ic_points,
                        num_bc_points=self.num_bc_points,
                        num_res_points=self.num_res_points,
                        hidden_size=h
                    )
                )
                task_list.append(t)

        domain = Domain()
        domain.add_parameter('pde_task_name')
        domain.add_parameter('seed')
        domain.add_parameter('dataset_path')
        domain.add_parameter('hidden_size')
        domain.add_parameter('x_range')
        domain.add_parameter('t_range')
        domain.add_parameter('xgrid_resolution')
        domain.add_parameter('tgrid_resolution')
        domain.add_parameter('num_ic_points')
        domain.add_parameter('num_bc_points')
        domain.add_parameter('num_res_points')
        domain.add_parameter('beta')
        domain.add_parameter('rho')

        return ExperimentData(domain=domain, input_data=task_list,
                              project_dir=data.project_dir)
