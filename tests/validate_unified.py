import l2co
import matplotlib.pyplot as plt
from l2co.plotting import OPTIMIZER_COLORS
from l2co._src.performance import output_cummin, scale_by_task, ertd
from l2co._src.tree_utils import traj_and_stair
import l2co.plotting
import jax.random as jr
from l2co._src.models import MLP, HistoryModel, LSTM_Encoder
import numpy as np
import optax
import xarray as xr
import jax
import jax.numpy as jnp
from l2co.naming import (
    FEATURE,
    TASK_ID,
    OUTPUT_MIN,
    BUDGET,
    FEVALS,
    OUTPUT_DIM,
    ITERATIONS,
    OUTPUT_MEAN,
    KIND,
    PROBLEM,
    SCHEDULE,
)
import equinox as eqx
from l2co import FeatureProcessor, MixedEmbedding


def main():
    print("Loading data...")
    # Load dataset
    d = l2co.DataLoader.from_file(
        "/Volumes/T7/data/8558179/entire_dataset.odata"
    )

    # Validation params
    t_end = 100
    print(f"Generating params for t_end={t_end}...")
    o = d.to_params(
        t=0.1, t_end=t_end, schema={"dimensionality": "continuous"}
    )

    # Initialize Model Components
    kind_size = o.x_history.sizes["kind"]  # Should be 2 + N_schedules
    n_schedules = len(d.optimizers)

    print(
        f"Initializing model (Input dim: {kind_size}, Output dim: {n_schedules})..."
    )

    history_encoder = LSTM_Encoder(
        in_size=kind_size,
        hidden_size=32,  # Increased slightly for unified task
        out_size=16,
        key=jr.key(3),
    )

    mixed_embedding = MixedEmbedding.from_feature_processor(
        feature_processor=o.feature_processor_task,
        embed_dim=8,
        output_dim=16,
        key=jr.key(3),
    )

    model = eqx.nn.MLP(
        in_size=32,  # 16 (history) + 16 (features)
        out_size=n_schedules,
        width_size=32,
        depth=3,
        key=jr.key(4),
    )

    history_model = HistoryModel(
        feature_processor_task=o.feature_processor_task,
        mixed_embedding_task=mixed_embedding,
        history_encoder=history_encoder,
        model=model,
        optimizers=d.optimizers.values,
    )

    # Train Unified Model
    print("Starting Unified Training (i=None)...")
    new_history_model, metrics = history_model.train(
        data=o,
        i=None,
        optimizer=optax.adam(1e-3),
        num_steps=10,
        scan=False,
        verbose=True,
    )

    print(f"Final Train Loss: {metrics['train_loss'][-1]:.4f}")

    # Plot Training Metrics
    fig, ax = plt.subplots()
    ax.plot(metrics["train_loss"], label="Train Loss")
    if "val_loss" in metrics and not jnp.isnan(metrics["val_loss"]).all():
        ax.plot(metrics["val_loss"], label="Validation Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig("unified_training_loss.png", dpi=300, bbox_inches="tight")

    # Evaluate Strategy
    print("Evaluating Unified Policy (i=None)...")
    # Evaluate across ALL (Problem, Schedule) pairs at once
    y_pred_all = new_history_model.evaluate(key=jr.key(3), data=o, i=None)
    print(
        f"Prediction output shape: {y_pred_all.shape} (Expected: ({o.x_features_task.sizes[PROBLEM]}, {len(d.optimizers)}))"
    )

    # Select a specific history context for visualization (e.g., Schedule 0)
    eval_schedule_idx = 0
    eval_schedule_name = str(d.optimizers[eval_schedule_idx].values)
    print(
        f"Extracting policy assuming history from schedule: {eval_schedule_name}"
    )

    y_pred_context = y_pred_all.isel({SCHEDULE: eval_schedule_idx})
    d_new = d.create_strategy(choices=y_pred_context, name="Unified-Model")

    # Plot ERTD
    print("Plotting ERTD Profile...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(x=t_end, color="k", linestyle="--")

    # Plot baseline (schedule used for history)
    # And the new strategy
    # Note: creating a strategy adds it to the dataloader, so plot_ertd_profile should show it.

    # Filter to show relevant curves (baselines + new strategy)
    # For clarity, let's just plot everything since plot_ertd_profile does that by default
    fig, ax = l2co.plotting.plot_ertd_profile(d=d_new, ax=ax)

    ax.set_title(
        f"ERTD Profile (History: {eval_schedule_name}, Budget: {t_end})"
    )

    output_filename = "unified_model_ertd.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Saved ERTD plot to {output_filename}")


if __name__ == "__main__":
    main()
