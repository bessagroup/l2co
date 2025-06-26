from pathlib import Path

# Third-party
import hydra
from hydra.utils import instantiate

# Local
from l2co import DataLoader, L2CO_model, logger

# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config):
    """
    Main function to train the L2CO model.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Configuration object containing all parameters for the training
        process, including paths, model configurations, and training
        hyperparameters.

    Returns
    -------
    None
    """
    logger.info("Started model training script")
    logger.info(f"{config=}")

    # Load optimization data
    path = Path(config.opt_dir) / str(config.path)
    data = DataLoader.from_file(path)
    logger.info(f"Loaded data from {path}: {len(data.data)} data points")

    # Export data for training
    data.export_all_and_store(
        path=Path.cwd() / "data",
        t_start=config.t_start,
        t=config.t,
        t_end=config.t_end,
    )
    logger.info(f"Exported data to {Path.cwd() / 'data'}")

    # Initialize Stage 0 model
    stage0_model = instantiate(config.stage0_model)
    logger.info(f"Instantiated stage0 model: {stage0_model}")

    # Initialize Stage 1 model
    stage1_model_class = instantiate(config.stage1_model, _partial_=True)
    logger.info(f"Instantiated stage1 model: {stage1_model_class}")
    logger.info(f"Hyperparameters: {stage1_model_class.__dict__}")

    # Create L2CO model
    l2co_model = L2CO_model.init(
        stage0_model=stage0_model,
        stage1_model_class=stage1_model_class,
        feature_types=config.feature_types,
    )
    logger.info(f"Instantiated L2CO model: {l2co_model}")

    # Train the model
    optimizer = instantiate(config.optimizer)
    loss_fn = instantiate(config.loss_fn, _partial_=True)
    l2co_model.train(
        path=Path.cwd() / "data",
        optimizer=optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Trained L2CO model: {l2co_model} with {optimizer=}, "
        f"{config.loss_fn=}, {config.epochs=}, {config.batch_size=}, "
        f"{config.seed=}"
    )

    # Save the model
    l2co_model.save(path=Path.cwd() / "model")
    logger.info(f"Saved L2CO model to {Path.cwd() / 'model'}")

    logger.info("End of this program")


if __name__ == "__main__":
    main()

# =============================================================================
