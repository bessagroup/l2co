from l2co import OptimizationStep
from l2co._src.optimizers import optimizers
from l2co._src.optimizers.optax_implementations import normalized_optax_normal

OPTIMIZATION_STEPS = [
    OptimizationStep(
        optimizer="sepcmaes",
        stopping_criteria={"type": "covariance_matrix", "tol": 1e-12},
    ),
    OptimizationStep(
        optimizer="adam",
        hyperparameters={"learning_rate": 1e-3},
        stopping_criteria={
            "type": "combined",
            "grad_tol": 1e-3,
            "loss_tol": 1e-6,
        },
    ),
    OptimizationStep(
        optimizer="lbfgs",
        stopping_criteria={
            "type": "combined",
            "grad_tol": 1e-3,
            "loss_tol": 1e-6,
        },
    ),
    OptimizationStep(optimizer="pso"),
    OptimizationStep(
        optimizer="randomsearch",
        hyperparameters={"sampler": "normal", "popsize": 8},
    ),
]


def _default_hyperparameters(name: str) -> dict:
    """Hyperparameters required to construct each optimizer in
    ``optimizers``. Optax-normal optimizers need a ``learning_rate``;
    everything else (L-BFGS, EvoSax algorithms, random search) is happy
    with the empty dict.
    """
    if name in normalized_optax_normal:
        return {"learning_rate": 1e-3}
    return {}


# Registry entries that cannot be constructed bare: "l2co" is a
# meta-optimizer whose hyperparameters must name trained artifacts
# (stage0_model, stage1_model, t, sampler, budget_fn), so a
# default-hyperparameter step has no meaning. Its registry path is
# covered end-to-end — with real fixture checkpoints — in
# tests/integration/test_strategy_wrapper_rollout.py.
_ARTIFACT_BACKED = {"l2co"}

ALL_OPTIMIZERS = [
    OptimizationStep(
        optimizer=name, hyperparameters=_default_hyperparameters(name)
    )
    for name in optimizers.keys()
    if name not in _ARTIFACT_BACKED
]
