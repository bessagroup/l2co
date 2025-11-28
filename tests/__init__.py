from l2co import OptimizationStep

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
        optimizer="lbfgsb",
        stopping_criteria={
            "type": "combined",
            "grad_tol": 1e-3,
            "loss_tol": 1e-6,
        },
    ),
]
