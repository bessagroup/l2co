import hydra
import l2co
import equinox as eqx
import jax.random as jr
import optax
import math
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from l2co._src.models import LSTM_Encoder, HistoryModel, MixedEmbedding
from l2co.naming import TASK_ID, REALIZATION

log = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    # --- 1. JAX Key Management ---
    key = jr.key(config.train.seed)
    k_encoder, k_task, k_opt, k_mlp = jr.split(key, 4)

    # --- 2. Data Preparation ---
    d = l2co.DataLoader.from_file(Path(config.path) / "entire_dataset.odata")
    d = d.select_optimizer(list(config.opts))
    o = d.to_params(t=config.t, t_end=config.t_end, schema=config.schema)

    # --- 3. Model Initialization ---
    history_encoder = LSTM_Encoder(
        in_size=2,
        hidden_size=config.model.history_encoder.hidden_size,
        out_size=config.model.history_encoder.out_size,
        key=k_encoder,
    )

    mixed_embedding_task = MixedEmbedding.from_feature_processor(
        feature_processor=o.feature_processor_task,
        embed_dim=config.model.embedding.dim,
        output_dim=config.model.embedding.dim,
        key=k_task,
    )

    mixed_embedding_optimizer = MixedEmbedding.from_feature_processor(
        feature_processor=o.feature_processor_optimizer,
        embed_dim=config.model.embedding.dim,
        output_dim=config.model.embedding.dim,
        key=k_opt,
    )

    model = eqx.nn.MLP(
        in_size=config.model.mlp.in_size,
        out_size=len(d.optimizers),
        width_size=config.model.mlp.width_size,
        depth=config.model.mlp.depth,
        key=k_mlp,
    )

    history_model = HistoryModel(
        feature_processor_task=o.feature_processor_task,
        feature_processor_optimizer=o.feature_processor_optimizer,
        mixed_embedding_task=mixed_embedding_task,
        mixed_embedding_optimizer=mixed_embedding_optimizer,
        history_encoder=history_encoder,
        model=model,
        optimizers=d.optimizers.values,
    )

    # --- 4. Training ---
    log.info("Starting training...")
    new_history_model, metrics = history_model.train(
        data=o,
        optimizer=optax.adam(config.train.learning_rate),
        num_steps=config.train.num_steps,
        verbose=config.train.verbose,
    )

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

    for opt_ in config.opts:
        log.info(f"Generating plots for strategy: {opt_}")
        choices = (
            y_pred.unstack("sample")
            .sel(schedule2=opt_, drop=True)
            .stack(problem=(TASK_ID, REALIZATION), create_index=True)
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


def _add_eval_line(ax, line_val):
    if line_val is not None:
        ax.axvline(
            x=line_val, color="black", linestyle="--", linewidth=2, zorder=5
        )


def ertd_per_category(d, line=None):
    groups = [
        {
            "fn_name": [
                "sphere",
                "ellipsoid_seperable",
                "rastrigin_seperable",
                "skew_rastrigin_bueche",
                "linear_slope",
            ]
        },
        {
            "fn_name": [
                "attractive_sector",
                "step_ellipsoid",
                "rosenbrock",
                "rosenbrock_rotated",
            ]
        },
        {
            "fn_name": [
                "ellipsoid",
                "discuss",
                "bent_cigar",
                "sharp_ridge",
                "sum_of_different_power",
            ]
        },
        {
            "fn_name": [
                "rastrigin",
                "weierstrass",
                "schaffer_f7_condition_10",
                "schaffer_f7_condition_1000",
                "griewank_rosenbrock_f8f2",
            ]
        },
        {
            "fn_name": [
                "schwefel_xsinx",
                "gallagher_101_peaks",
                "gallagher_21_peaks",
                "katsuura",
                "lunacek_bi_rastrigin",
            ]
        },
    ]
    names = [
        "separable",
        "low_conditioning",
        "high_conditioning",
        "multimodal_adequate",
        "multimodal_weak",
    ]

    # Hardcoded figure size
    fig, axes = plt.subplots(1, len(groups), figsize=(20, 4), sharey=True)

    for i, (ax, group) in enumerate(zip(axes, groups)):
        d_group = d.select_feature(group)
        l2co.plotting.plot_ertd_profile(
            d_group,
            ax=ax,
            show_legend=(i == len(groups) - 1),
            title_text=names[i],
        )
        _add_eval_line(ax, line)
    return fig, axes


def ertd_per_dimension(d, line=None):
    dims = [2, 3, 5, 10, 20, 40]
    # Hardcoded figure size
    fig, axes = plt.subplots(1, len(dims), figsize=(25, 4), sharey=True)

    for i, (ax, dim) in enumerate(zip(axes, dims)):
        d_group = d.select_feature({"dimensionality": [dim]})
        l2co.plotting.plot_ertd_profile(
            d_group,
            ax=ax,
            show_legend=(i == len(dims) - 1),
            title_text=f"{dim}D",
        )
        _add_eval_line(ax, line)
    return fig, axes


if __name__ == "__main__":
    main()
