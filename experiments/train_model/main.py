# Standard
from pathlib import Path
import logging
import socket
from time import perf_counter
from time import sleep

# Third-party
import jax.random as jr
import hydra
from hydra.utils import instantiate
from mpi4py import MPI
from omegaconf import DictConfig
from f3dasm import DataGenerator, ExperimentSample, ExperimentData, Block
from f3dasm.hydra_tools import update_config_with_experiment_sample

# Local
from l2co import (
    DataLoader,
    Stage0Loader,
    Stage0Model,
    OptimizationStep,
    FeatureProcessor,
    RunOptimization,
    Task,
)

# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================


class ModelTrainer(DataGenerator):
    def __init__(self, config: DictConfig):
        self.config = config

    def execute(
        self, experiment_sample: ExperimentSample, id: int
    ) -> ExperimentSample:
        t1 = perf_counter()

        config = update_config_with_experiment_sample(
            self.config, experiment_sample
        )
        logger.info(f"[{id}] Starting training...")

        _path = (
            Path(config.project_dir) / Path(config.path) / "entire_dataset"
        ).with_suffix(".odata")

        d = DataLoader.from_file(_path)

        opt_data = d.to_params(t=config.t, cost_metric=config.cost_metric)

        # TEMPORARY: nan to num
        opt_data.y = opt_data.y.fillna(0.0)

        optimizers = [
            OptimizationStep.from_yaml(opt) for opt in config.optimizers
        ]

        feature_processor = FeatureProcessor.init(
            schema=dict(config.tasks.schema),
            tags=opt_data.to_tag_list(),
        )

        # Initialize model
        l2co_model = Stage0Model.init(
            feature_processor=feature_processor,
            optimizers=optimizers,
            model=instantiate(config.models.model),
            type=config.models.type,
        )

        train_kwargs = (
            dict(config.models.train_kwargs)
            if config.models.train_kwargs
            else {}
        )

        # Train the model
        l2co_model, training_metrics = l2co_model.train(
            data=opt_data, **train_kwargs
        )

        # Store the trained model in the experiment sample
        experiment_sample.store(
            name="l2co_model",
            object=l2co_model,
            to_disk=True,
            store_function=Stage0Loader.store,
            load_function=Stage0Loader.load,
        )

        logger.info(
            f"[{id}] Finished training model.\t "
            f"(Took {perf_counter() - t1:.2f} s)"
        )

        return experiment_sample


def process(config: DictConfig, comm=None) -> None:
    trainer = ModelTrainer(config)

    experiment_data = ExperimentData.from_file(
        project_dir=".",
        wait_for_creation=True,
        max_tries=500,
    )

    if config.mode == "mpi":
        experiment_data = trainer.call(
            experiment_data, mode=config.mode, comm=comm, pass_id=True
        )

    if config.mode in ("cluster", "sequential"):
        experiment_data = trainer.call(
            experiment_data,
            mode=config.mode,
            pass_id=True,
        )

    if config.mode == "sequential":
        # Save the results to a file
        experiment_data.store()


def pre_process(config: DictConfig) -> None:
    # Initialize the domain
    experiment_data = ExperimentData.from_yaml(config.experimentdata)
    sampler: Block = instantiate(config.f3dasm_sampler)
    experiment_data = sampler.call(experiment_data)

    experiment_data.store()


# =============================================================================


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(config: DictConfig) -> None:
    if config.mode == "mpi":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"[{rank}/{size}] Running on {socket.gethostname()}")

        if rank == 0:
            pre_process(config)
            comm.bcast("ready", root=0)  # Signal other ranks to start
            process(config, comm)
        else:
            signal = comm.bcast(None, root=0)  # Ensure all ranks receive
            if signal == "ready":
                sleep(2)
                process(config, comm)

    if config.mode in ("cluster", "sequential"):
        pre_process(config)
        process(config)


if __name__ == "__main__":
    main()

# =============================================================================
