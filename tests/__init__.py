from l2co import OptimizationStep
from l2co._src.optimizers import optimizers

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

ALL_OPTIMIZERS = list(optimizers.keys())
