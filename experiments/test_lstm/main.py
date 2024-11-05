# Third-party
import logging
import tracemalloc
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import xarray as xr
from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from hydra.utils import instantiate
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from l2co._src.data_loader import OptimizationDynamics, load_all_trajectories
from l2co._src.preprocessing import preprocessing_factory
from l2co._src.stage0_model import Stage0Model
from l2co._src.stage1_model_equinox import eval_step, train_loop

# =============================================================================


class CacheHandler:
    def __enter__(self):
        # Setup code if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Clear caches regardless of success or failure
        # eqx.clear_caches()
        jax.clear_caches()
        logging.info('Caches cleared')


# =============================================================================

class HyperParameterSearch(DataGenerator):
    def __init__(self, config, data_traj: OptimizationDynamics,
                 data_features: xr.DataArray):
        self.X_trajectory = jnp.array(data_traj.X_trajectory)
        # self.X_features = jnp.array(data_traj.X_features)
        self.X_features = jnp.array(data_features.values)
        self.Y = jnp.array(data_traj.Y)
        self.cfg = config

    def execute(self):
        # Take a snapshot and print current memory usage at the start of `execute()`
        snapshot = tracemalloc.take_snapshot()
        current_memory = sum(
            stat.size for stat in snapshot.statistics('lineno'))
        logging.info(
            f"Memory usage at the start of execute(): {current_memory / (1024 * 1024):.2f} MB")

        config = update_config_with_experiment_sample(
            config=self.cfg,
            experiment_sample=self.experiment_sample,
            force_add=True
        )

        # Context manager for cache clearing
        with CacheHandler():
            model = instantiate(config.model)
            logging.info("Created model")
            optimizer = instantiate(config.optimizer)
            logging.info("Starting training")
            model = train_loop(model, self.X_trajectory, self.X_features,
                               self.Y, optimizer, config.epochs)

            loss = eval_step(model=model, X_trajectory=self.X_trajectory,
                             X_features=self.X_features, Y=self.Y)

        del model
        del optimizer
        self.experiment_sample.store(name='loss', object=loss)

        # Take another snapshot and print memory usage at the end of `execute()`
        snapshot = tracemalloc.take_snapshot()
        current_memory = sum(
            stat.size for stat in snapshot.statistics('lineno'))
        logging.info(
            f"Memory usage at the end of execute(): {current_memory / (1024 * 1024):.2f} MB")


@hydra.main(config_path="conf", config_name="config")
def main(config):

    experimentdata = ExperimentData.from_yaml(config.experimentdata)
    opt_dyn = OptimizationDynamics.load(
        path=Path(config.path_opt_dyn) / 'Adam',
        rel_tolerance=config.rel_tolerance
    )

    opt_features = opt_dyn.to_feature_data()

    ####################################################################

    preprocessing_step = preprocessing_factory(config.feature_types)

    clf = MultiOutputRegressor(HistGradientBoostingRegressor())

    stage0_model = Stage0Model(
        model_name='stage0',
        clf=clf,
        preprocessing_step=preprocessing_step,
        type='regressor',
    )

    stage0_model.train(opt_features)
    data_features = stage0_model.predict_raw(
        opt_features)  # Shape: (n_samples, n_labels)

    ####################################################################

    hyperparameter_search = HyperParameterSearch(
        config=config, data_traj=opt_dyn,
        data_features=data_features)

    # Start tracemalloc to monitor memory before evaluation
    tracemalloc.start()

    # Print initial memory usage before `experimentdata.evaluate()`
    initial_snapshot = tracemalloc.take_snapshot()
    initial_memory = sum(
        stat.size for stat in initial_snapshot.statistics('lineno'))
    logging.info(
        f"Initial memory usage before evaluate(): {initial_memory / (1024 * 1024):.2f} MB")

    experimentdata.evaluate(
        data_generator=hyperparameter_search, mode='cluster')

    # Print memory usage after `experimentdata.evaluate()`
    final_snapshot = tracemalloc.take_snapshot()
    final_memory = sum(
        stat.size for stat in final_snapshot.statistics('lineno'))
    logging.info(
        f"Final memory usage after evaluate(): {final_memory / (1024 * 1024):.2f} MB")

    # Stop tracemalloc after evaluation
    tracemalloc.stop()


main()
