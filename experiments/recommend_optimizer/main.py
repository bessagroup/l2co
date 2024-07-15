#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import perf_counter, sleep

# Third-party
import hydra
import matplotlib.pyplot as plt
import xarray as xr
from f3dasm import ExperimentData, logger
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from hydra.utils import instantiate
from omegaconf import OmegaConf

from l2co import LSTMMultiOutputClassifier
from l2co import logger as l2co_logger
from l2co._src.stage1_model import ELAClassifier
from l2co.data import DataLoader
from l2co.preprocessing import preprocessing_factory
from l2co.trainer import Trainer

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

        # Construct the classifiers
        stage0_clf = instantiate(config.stage0_clf)
        stage1_clf = LSTMMultiOutputClassifier(
            n_outputs=config.n_optimizers, length_sequence=config.t_switch,
            **config.stage1_clf)

        preprocessing_step = preprocessing_factory(
            feature_types=config.feature_types)

        training_project_dir = Path(config.data.training_project_dir)
        testing_project_dir = Path(
            config.data.testing_project_dir
        ) if config.data.testing_project_dir is not None else None

        t = perf_counter()

        d_train = DataLoader.from_saved_files(
            path=training_project_dir, config=config
        )

        l2co_logger.info(
            f"Loaded training data from {training_project_dir} (took {perf_counter() - t} seconds")

        t = perf_counter()

        if testing_project_dir is None:
            d_train, d_test = d_train.train_test_split(
                test_size=config.test_size, random_state=config.seed)

            l2co_logger.info((
                f"Split training data into training and testing data )"
                f"(with a test size of {config.test_size} and seed {config.seed} (took {perf_counter() - t} seconds")
            )
        else:
            d_test = DataLoader.from_saved_files(
                path=testing_project_dir, config=config
            )

            l2co_logger.info(
                f"Loaded testing data from {testing_project_dir} (took {perf_counter() - t} seconds")

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
            iterations=config.iterations-1,
            t_switch=config.t_switch,
            t_start=config.t_start,
            rel_tolerance=config.rel_tolerance,
            preprocessing_step=preprocessing_step,
            epochs=config.epochs,
            batch_size=config.batch_size,
            name='trainer')

        l2co_logger.info("Created trainer")

        # trainer.train()
        t = perf_counter()
        trainer.train()
        # trainer.train_stage0()

        # l2co_logger.info(f"Trained trainer (took {perf_counter() - t} seconds")

        # t = perf_counter()
        # if config.stage1_clf.path is not None:
        #     trainer.load_stage1(config.stage1_clf.path)
        #     l2co_logger.info(
        #         f"Loaded stage1 model from {config.stage1_clf.path}")

        # else:
        #     trainer.train_stage1(save_model=True)
        #     l2co_logger.info(
        #         f"Trained stage1 model (took {perf_counter() - t} seconds")

        # # l2co_logger.info(f"Trained trainer (took {perf_counter() - t} seconds")

        # t = perf_counter()
        trainer.predict()
        l2co_logger.info(
            f"Predicted trainer (took {perf_counter() - t} seconds")

        # accs = trainer.all_accuracies()

        # for key, acc in accs.items():
        #     self.experiment_sample.store(object=acc, name=f"accuracy_{key}",
        #                                  to_disk=False)

        # acc = trainer.accuracy_func(
        #     y_true=trainer.data_test.select_single_optimizers().query_t(
        #         trainer.iterations).y_binary(config.rel_tolerance),
        #     y_pred=trainer.stage0_prediction)

        # self.experiment_sample.store(
        #     object=acc, name='accuracy_first', to_disk=False)

        self.experiment_sample.store(
            object=trainer.stage0_prediction.reset_index('problem'),
            name='stage0_prediction',
            to_disk=True)

        self.experiment_sample.store(
            object=trainer.stage1_prediction.reset_index('problem'),
            name='stage1_prediction',
            to_disk=True)

        # self.experiment_sample.store(
        #     object=trainer.data_test.select_single_optimizers().query_t(
        #         trainer.iterations).y_binary(config.rel_tolerance).reset_index('problem'),
        #     name='y_true_single', to_disk=True)

        # self.experiment_sample.store(
        #     object=trainer.data_test.query_t(
        #         trainer.iterations).y_binary(config.rel_tolerance).reset_index('problem'),
        #     name='y_true_both', to_disk=True)

        # trainer.create_performance_profile(
        #     strategies=[trainer.stage1_prediction,
        #                 trainer.stage0_prediction,
        #                 y_pred_ela])

        # fig, axs = plt.subplots(
        #     2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))

        # _ = trainer.plot_performance_profile(
        #     strategies=[trainer.stage1_prediction, trainer.stage0_prediction], strategy_names=['trainer_stage1', 'trainer_stage0'], axes=axs)

        # self.experiment_sample.store(object=trainer.performance_profile,
        #                              name='performance_profile',
        #                              to_disk=True)

        # self.experiment_sample.store(
        #     object=float(
        #         trainer.performance_profile['trainer_stage1'].sel(f=1.0).values),
        #     name='pp_value', to_disk=False)

        # self.experiment_sample.store(object=fig, name='performance_profile_fig',
        #                              to_disk=True)

        # self.experiment_sample.store(
        #     object=trainer, name='trainer', to_disk=True)


@hydra.main(config_path=".", config_name="config")
def main(config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """

    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)
        # post_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)
        # post_processing(config)
    else:
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        process(config)


def pre_processing(config):
    experimentdata = ExperimentData.from_yaml(config.experimentdata)

    experimentdata.store()

    recommend_optimizer = RecommendOptimizer(config)

    experimentdata.evaluate(data_generator=recommend_optimizer,
                            mode='cluster')


def process(config):
    # Retrieve the ExperimentData object
    max_tries = 100
    tries = 0

    while tries < max_tries:
        try:
            logger.info("Open ExperimentData")

            experimentdata = ExperimentData.from_file(Path().cwd())
            break  # Break out of the loop if successful
        except FileNotFoundError:
            tries += 1
            logger.info("FileNotFoundError, sleeping!")
            sleep(10)

    recommend_optimizer = RecommendOptimizer(config)

    experimentdata.evaluate(data_generator=recommend_optimizer,
                            mode='cluster')


if __name__ == "__main__":
    main()
