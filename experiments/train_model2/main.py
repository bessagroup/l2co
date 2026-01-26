import hydra
import l2co
import equinox as eqx
import jax.random as jr
import optax
import math
import numpy as np
import logging
from pathlib import Path
from l2co import OptimizationStep
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from l2co._src.models import (
    LSTM_Encoder,
    EquinoxStage1Model,
    MixedEmbedding,
    MLP,
    Stage0Loader,
    Stage1Loader,
    ScikitStage0Model,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from l2co.naming import TASK_ID, REALIZATION, PROBLEM, SCHEDULE
from l2co.plotting import (
    ertd_per_category,
    ertd_per_dimension,
    ertd_per_function_raster,
    plot_r2,
)
from l2co._src.visualize.bbob_plot import _add_eval_line

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def main(config):
    # --- 1. JAX Key Management ---
    key = jr.key(config.seed)
    k_encoder, k_task, k_opt, k_mlp = jr.split(key, 4)

    # --- 2. Data Preparation ---
    _path = Path(config.project_dir) / str(config.path)
    d = l2co.DataLoader.from_file(_path / "entire_dataset.odata")
    o = d.to_params(t=config.t, t_end=config.t_end, schema=config.tasks.schema)

    # --- 3. Stage1 Model Initialization ---
    optimizers = [OptimizationStep.from_yaml(opt) for opt in config.optimizers]

    history_encoder = LSTM_Encoder(
        in_size=2,
        hidden_size=config.stage1_models.history_encoder.hidden_size,
        out_size=config.stage1_models.history_encoder.out_size,
        key=k_encoder,
    )

    mixed_embedding_task = MixedEmbedding.from_feature_processor(
        feature_processor=o.feature_processor_task,
        embed_dim=config.stage1_models.embedding.dim,
        output_dim=config.stage1_models.embedding.dim,
        key=k_task,
    )

    mixed_embedding_optimizer = MixedEmbedding.from_feature_processor(
        feature_processor=o.feature_processor_optimizer,
        embed_dim=config.stage1_models.embedding.dim,
        output_dim=config.stage1_models.embedding.dim,
        key=k_opt,
    )

    model = MLP.init(
        in_size=config.stage1_models.mlp.in_size,
        out_size=len(d.optimizers),
        width_size=config.stage1_models.mlp.width_size,
        depth=config.stage1_models.mlp.depth,
        key=k_mlp,
    )

    history_model = EquinoxStage1Model(
        feature_processor_task=o.feature_processor_task,
        feature_processor_optimizer=o.feature_processor_optimizer,
        mixed_embedding_task=mixed_embedding_task,
        mixed_embedding_optimizer=mixed_embedding_optimizer,
        history_encoder=history_encoder,
        model=model,
        optimizers=optimizers,
    )

    # --- 4. Training ---
    logger.info("Starting training...")
    new_history_model, metrics = history_model.train(
        data=o,
        optimizer=optax.adam(config.train.learning_rate),
        num_steps=config.train.num_steps,
        verbose=config.train.verbose,
    )

    Stage1Loader.store(object=new_history_model, path="stage1model")

    # Stage0 init and training

    model = instantiate(config.stage0_models)

    stage0model = ScikitStage0Model.init(
        feature_processor_task=o.feature_processor_task,
        model=model,
        optimizers=optimizers,
    )

    new_stage0_model, _ = stage0model.train(
        data=o,
    )

    Stage0Loader.store(object=new_stage0_model, path="stage0model")

    # --- 5. Metric Plotting (Hardcoded Layout) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["train_loss"], label="Train Loss")
    ax.plot(metrics["val_loss"], label="Validation Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig("training_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 6. Evaluation & Strategy Plots ---
    y_pred = new_history_model.evaluate(data=o)
    y_pred_logits = new_history_model.pred_logits(data=o.filter())
    fig, ax = plot_r2(o.filter().y, y_pred_logits)
    fig.savefig("r2_plot.png", dpi=300, bbox_inches="tight")

    for opt_ in np.unique(y_pred[SCHEDULE].values):
        logger.info(f"Generating plots for strategy: {opt_}")
        choices = (
            y_pred.unstack("sample")
            .sel({SCHEDULE: opt_}, drop=True)
            .stack({PROBLEM: (TASK_ID, REALIZATION)}, create_index=True)
        )

        d_with_l2co = d.create_strategy(choices=choices, name="l2co")

        # Global ERTD
        fig, ax = l2co.plotting.plot_ertd_profile(
            d_with_l2co, title_text=f"ERTD with L2CO - {opt_}"
        )
        _add_eval_line(ax, config.t_end)
        fig.savefig(f"ertd_{opt_}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Categorical ERTD
        fig, _ = ertd_per_category(d_with_l2co, line=config.t_end)
        fig.savefig(
            f"ertd_per_category_{opt_}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # Dimensional ERTD
        fig, _ = ertd_per_dimension(d_with_l2co, line=config.t_end)
        fig.savefig(
            f"ertd_per_dimension_{opt_}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
