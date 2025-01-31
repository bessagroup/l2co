from f3dasm.datageneration import DataGenerator
from omegaconf import DictConfig

from l2co import logger
from l2co._src.tasks.benchmarkfunctions.run_optimization import \
    run_optimization_schedule_multiple_realizations

# =============================================================================


class RunOptimization(DataGenerator):
    def __init__(self, config: DictConfig):
        self.config = config

    def execute(self):
        es = self.experiment_sample.input_data
        task = es['task']
        schedule = es['optimizer schedule']
        seed = es['seed']

        data_xarr = run_optimization_schedule_multiple_realizations(
            task=task,
            optimization_schedule=schedule,
            seed=seed,
            n_realizations=self.config.optimization.n_realizations,
            include_input=False,
            bounded=(self.config.optimization.lower_bound,
                     self.config.optimization.upper_bound),
        )

        self.experiment_sample.store(
            name='loss history',
            object=data_xarr, to_disk=True)

        logger.info(
            f"Finished optimization task {task} with {schedule.tag['name']} ")
