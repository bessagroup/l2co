# Third-party
import logging
from pathlib import Path

import equinox as eqx
import hydra
import jax.numpy as jnp
from hydra.utils import instantiate

from l2co._src.stage1_model_equinox import train_loop

# =============================================================================


@hydra.main(config_path="conf", config_name="config")
def main(config):

    # Load data
    p = Path(config.numpy_data)
    X_trajectory = jnp.load(p / 'X_trajectory.npz')['arr_0']
    X_features = jnp.load(p / 'X_features.npz')['arr_0']
    Y = jnp.load(p / 'Y.npz')['arr_0']

    model = instantiate(config.model)
    logging.info("Created model")

    # check if config.saved_parameters exists
    if config.saved_parameters is not None:
        model = eqx.tree_deserialise_leaves(
            config.saved_parameters, model)
        logging.info("Loaded saved parameters for model")

    optimizer = instantiate(config.optimizer)

    logging.info("Starting training")
    model = train_loop(model, X_trajectory, X_features, Y,
                       optimizer, config.epochs)

    # Save model
    eqx.tree_serialise_leaves("model.eqx", model)


main()
