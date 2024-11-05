# Third-party
import logging
from pathlib import Path

import equinox as eqx
import hydra
import jax.numpy as jnp
from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from hydra.utils import instantiate

from l2co._src.data_loader import OptimizationDynamics, load_all_trajectories
from l2co._src.stage1_model_equinox import eval_step, train_loop

# =============================================================================


class HyperParameterSearch(DataGenerator):
    def __init__(self, config, data: OptimizationDynamics):
        self.X_trajectory = jnp.array(data.X_trajectory)
        self.X_features = jnp.array(data.X_features)
        self.Y = jnp.array(data.Y)
        self.cfg = config

    def execute(self):
        config = update_config_with_experiment_sample(
            config=self.cfg,
            experiment_sample=self.experiment_sample,
            force_add=True)

        model = instantiate(config.model)
        logging.info("Created model")
        optimizer = instantiate(config.optimizer)
        logging.info("Starting training")
        model = train_loop(model, self.X_trajectory, self.X_features,
                           self.Y, optimizer, config.epochs)

        loss = eval_step(model=model, X_trajectory=self.X_trajectory,
                         X_features=self.X_features, Y=self.Y)

        self.experiment_sample.store(name='loss', object=loss)
        del model

        # # Save model
        # eqx.tree_serialise_leaves("model.eqx", model)


@hydra.main(config_path="conf", config_name="config")
def main(config):
    experimentdata = ExperimentData.from_yaml(config.experimentdata)
    opt_dyn = OptimizationDynamics.load(
        path=Path(config.path_opt_dyn) / 'Adam',
        rel_tolerance=config.rel_tolerance)
    hyperparameter_search = HyperParameterSearch(config=config, data=opt_dyn)

    experimentdata.evaluate(data_generator=hyperparameter_search,
                            mode='cluster')

    # X_trajectory = jnp.array(opt_dyn.X_trajectory)
    # X_features = jnp.array(opt_dyn.X_features)
    # Y = jnp.array(opt_dyn.Y)

    # model = instantiate(config.model)
    # logging.info("Created model")

    # # check if config.saved_parameters exists
    # if config.saved_parameters is not None:
    #     model = eqx.tree_deserialise_leaves(
    #         config.saved_parameters, model)
    #     logging.info("Loaded saved parameters for model")

    # optimizer = instantiate(config.optimizer)

    # logging.info("Starting training")
    # model = train_loop(model, jnp.array(X_trajectory), jnp.array(X_features),
    #                    jnp.array(Y), optimizer, config.epochs)

    # loss = eval_step(model=model, X_trajectory=X_trajectory,
    #                  X_features=X_features, Y=Y)

    # # Save model
    # eqx.tree_serialise_leaves("model.eqx", model)


main()
