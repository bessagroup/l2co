import l2co
import matplotlib.pyplot as plt
from l2co.plotting import OPTIMIZER_COLORS
from l2co._src.performance import output_cummin, scale_by_task, ertd
from l2co._src.tree_utils import traj_and_stair
import l2co.plotting
import jax.random as jr
from l2co._src.models import MLP, EquinoxStage1Model, LSTM_Encoder
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

d = l2co.DataLoader.from_file("/Volumes/T7/data/8558179/entire_dataset.odata")

schema = {
    "dimensionality": "continuous",
    "separable": "boolean",
    "unimodal": "boolean",
}


def main(i: int, t_end: float):
    t = 0.1
    opt_name = str(d.optimizers[i].values)
    print(f"Starting training for optimizer {opt_name} with t_end={t_end}")
    o = d.to_params(t=t, t_end=t_end, schema=schema)

    history_encoder = LSTM_Encoder(
        in_size=2, hidden_size=16, out_size=8, key=jr.key(3)
    )

    mixed_embedding = MixedEmbedding.from_feature_processor(
        feature_processor=o.feature_processor_task,
        embed_dim=4,
        output_dim=8,
        key=jr.key(3),
    )

    model = eqx.nn.MLP(
        in_size=16,
        out_size=len(d.optimizers),
        width_size=16,
        depth=3,
        key=jr.key(4),
    )

    history_model = EquinoxStage1Model(
        feature_processor_task=o.feature_processor_task,
        mixed_embedding_task=mixed_embedding,
        history_encoder=history_encoder,
        model=model,
        optimizers=d.optimizers.values,
    )

    new_history_model, metrics = history_model.train(
        data=o, i=i, optimizer=optax.adam(1e-2), num_steps=1000, verbose=True
    )

    fig, ax = plt.subplots()
    ax.plot(metrics["train_loss"], label="Train Loss")
    ax.plot(metrics["val_loss"], label="Validation Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(
        f"metrics_{opt_name}_{t_end}.png", dpi=300, bbox_inches="tight"
    )

    y_pred = new_history_model.evaluate(key=jr.key(3), data=o, i=i)

    d_new = d.create_strategy(choices=y_pred, name="l2co")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(x=t_end, color="k", linestyle="--")
    fig, ax = l2co.plotting.plot_ertd_profile(d=d_new, ax=ax)

    fig.savefig(
        f"ertd_profile_{opt_name}_{t_end}.png", dpi=300, bbox_inches="tight"
    )
    print(f"Finished training for optimizer {opt_name} with t_end={t_end}")


# i = 1
# t_end = 1e5

for i in range(len(d.optimizers)):
    for t_end in [1e2, 1e3, 1e4, 1e5]:
        main(i, t_end)
