"""
Custom DataGenerator class to execute the L2CO experiment
"""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import perf_counter

# Third-party
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Local
from l2co import DataLoader, Trainer
from l2co import logger as l2co_logger
from l2co import preprocessing_factory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class RecommendOptimizer(DataGenerator):
    def __init__(self, config: OmegaConf) -> None:
        self.config = config

    def execute(self):
        config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=self.experiment_sample)

        training_project_dir = Path(config.data.training_project_dir)
        testing_project_dir = Path(
            config.data.testing_project_dir
        ) if config.data.testing_project_dir is not None else None

        t = perf_counter()

        d_train = DataLoader.from_file(
            path=training_project_dir, config=config
        )

        l2co_logger.info((
            f"Loaded training data from {training_project_dir} "
            f"(took {perf_counter() - t} seconds"))

        t = perf_counter()

        if testing_project_dir is None:
            d_train, d_test = d_train.train_test_split(
                test_size=config.test_size, random_state=config.seed)

            l2co_logger.info((
                f"Split training data into training and testing data )"
                f"(with a test size of {config.test_size} and seed "
                f"{config.seed} (took {perf_counter() - t} seconds")
            )
        else:
            d_test = DataLoader.from_file(
                path=testing_project_dir, config=config
            )

            l2co_logger.info((
                f"Loaded testing data from {testing_project_dir} "
                f"(took {perf_counter() - t} seconds"))

        # Construct the classifiers
        stage0_clf = instantiate(config.stage0_clf)
        stage1_clf = instantiate(config.stage1_clf)

        preprocessing_step = preprocessing_factory(
            feature_types=config.feature_types)

        # # ELA benchmark
        # ela_test = xr.open_dataarray(
        #     Path(config.data.ela_project_dir) / 'ela_test.nc', chunks='auto')
        # ela_train = xr.open_dataarray(
        #     Path(config.data.ela_project_dir) / 'ela_train.nc', chunks='auto')

        # clf = ELAClassifier(
        #     iterations=config.iterations-1,
        #     rel_tolerance=config.rel_tolerance,
        #     t_switch=config.t_switch)

        # clf.train(data_train=d_train, save_model=False, features=ela_train)
        # y_pred_ela = clf.predict(
        #     data_test=d_test, features=ela_test).rename('ELA')

        trainer = Trainer(
            data_train=d_train,
            data_test=d_test,
            stage0_clf=stage0_clf,
            stage1_clf=stage1_clf,
            iterations=config.iterations,  # config.iterations-1
            t_switch=config.t_switch,
            t_start=config.t_start,
            rel_tolerance=config.rel_tolerance,
            preprocessing_step=preprocessing_step,
            epochs=config.epochs,
            batch_size=config.batch_size,
            optimizer=instantiate(config.optimizer),
            name='trainer')

        l2co_logger.info("Created trainer")

        t = perf_counter()

        trainer.train()
        trainer.predict()

        l2co_logger.info(
            f"Predicted trainer (took {perf_counter() - t} seconds")

        self.experiment_sample.store(
            object=trainer.stage0_prediction.reset_index('problem'),
            name='stage0_prediction',
            to_disk=True)

        self.experiment_sample.store(
            object=trainer.stage1_prediction.reset_index('problem'),
            name='stage1_prediction',
            to_disk=True)
