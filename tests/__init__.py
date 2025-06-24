from l2co import OptimizationSchedule, OptimizationStep

OPTIMIZATION_SCHEDULES = [
    OptimizationSchedule(steps=[OptimizationStep(
        optimizer='cmaes', hyperparameters={'popsize': 30})], budgets=1.0),
    OptimizationSchedule(steps=[OptimizationStep(
        optimizer='adam', hyperparameters={'learning_rate': 1e-3})], budgets=1.0),
    OptimizationSchedule(
        steps=[OptimizationStep(optimizer='lbfgsb')], budgets=1.0)
]
