
from pathlib import Path

# Third-party
import hydra
from hydra.utils import instantiate

# Local
from l2co import DataLoader, L2CO_model, logger

# =============================================================================


# =============================================================================

@hydra.main(config_path=".", config_name="config")
def main(config):
    logger.info("Started model training script")
    logger.info(f"{config=}")

    # Data loading
    data = DataLoader.from_file(config.dataset_path)

    logger.info((
        f"Loaded data from {config.dataset_path}: {len(data.data)} "
        f"data points"))

    data.export_all_and_store(path=Path.cwd() / "data",
                              t_start=config.t_start,
                              t=config.t,
                              t_end=config.t_end,)

    logger.info(f"Exported data to {Path.cwd() / 'data'}")

    stage0_model = instantiate(config.stage0_model)

    logger.info(f"Instantiated stage0 model: {stage0_model}")

    stage1_model_class = instantiate(config.stage1_model, _partial_=True)
    logger.info(f"Instantiated stage1 model: {stage1_model_class}")
    logger.info((f"Hyperparameters: {stage1_model_class.__dict__}"))

    # Create l2co model
    l2co_model = L2CO_model.init(
        stage0_model=stage0_model,
        stage1_model_class=stage1_model_class,
        feature_types=config.feature_types,)

    logger.info(f"Instantiated l2co model: {l2co_model}")

    # Train the model
    optimizer = instantiate(config.optimizer)
    loss_fn = instantiate(config.loss_fn, _partial_=True)

    l2co_model.train(path=Path.cwd() / "data",
                     optimizer=optimizer,
                     epochs=config.epochs,
                     batch_size=config.batch_size,
                     seed=config.seed,
                     loss_fn=loss_fn,
                     )

    logger.info((f"Trained l2co model: {l2co_model} with {optimizer=}"
                f"{config.loss_fn=}, {config.epochs=}, "
                 f"{config.batch_size=}, {config.seed=}"
                 ))

    # Save the model
    l2co_model.save(path=Path.cwd() / "model")

    logger.info(f"Saved l2co model to {Path.cwd() / 'model'}")

    logger.info("End of this program")


if __name__ == "__main__":
    main()

# =============================================================================
