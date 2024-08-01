from pathlib import Path

import matplotlib.pyplot as plt
from f3dasm.datageneration import DataGenerator
from f3dasm.hydra_tools import update_config_with_experiment_sample
from hydra.utils import instantiate
from omegaconf import OmegaConf

from l2co._src.computervision.data import Dataset
from l2co._src.stage1_model import loss_fn_regression_hybrid, training_loop
from l2co.data import DataLoader


class LSTMArchitectures(DataGenerator):
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        self.data = DataLoader.from_saved_files(
            path=Path(config.data.training_project_dir), config=config
        )

    def execute(self):
        config = update_config_with_experiment_sample(
            config=self.config, experiment_sample=self.experiment_sample)

        model = instantiate(config.model)
        opt = instantiate(config.optimizer)

        for optimizer in self.data.select_single_optimizers().optimizers\
                .values:
            data_trajectory = Dataset.from_optimization_trajectory(
                d=self.data, optimizer=optimizer, t=config.t_switch,
                t_start=config.t_start, seed=config.seed)
            data_features = Dataset.from_problem_features(
                d=self.data, optimizer=optimizer, seed=config.seed)

            state, metrics = training_loop(
                model=model,
                data_features=data_features,
                data_trajectory=data_trajectory,
                iterations=config.iterations,
                seed=config.seed,
                batch_size=config.batch_size,
                loss_fn=loss_fn_regression_hybrid,
                track_loss=True,
                optimizer=opt,
            )

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax = ax.flatten()
            for i, key in enumerate(metrics.keys()):
                ax[i].plot(metrics[key])
                ax[i].set_title(key)

            self.experiment_sample.store(
                object=fig, name=f'metrics_{optimizer}', to_disk=True)
