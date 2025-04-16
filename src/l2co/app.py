# Standard
from pathlib import Path

# Third-party
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import scienceplots  # noqa

# Local
from l2co import DataLoader
from l2co._src.data_loader import performance_profile, rtd
from l2co.plotting import format_plot

# ============================================================================

# Enable Panel extensions
pn.extension('tabulator')
matplotlib.use("agg")  # Set a non-interactive backend
# Set Matplotlib non-interactive mode and apply style
plt.style.use(['science', 'bright', 'no-latex'])


# =============================================================================

# Default experiment and path
DEFAULT_EXPERIMENT = ""
DEFAULT_PATH = "/media/martin/T7/"

DEFAULT_DATA_FOLDER = "data"
DEFAULT_L2CO_FOLDER = "l2co"

optimizer_line_style = {'sbs': '--'}
optimizer_color_map = {'sbs': 'black'}

# Define TextInput widgets for experiment and path
experiment_single_optimizer_widget = pn.widgets.TextInput(
    value=DEFAULT_EXPERIMENT, name="Data single optimizers")

experiment_l2co_widget = pn.widgets.TextInput(
    value=DEFAULT_EXPERIMENT, name="Data L2CO model")

path_widget = pn.widgets.TextInput(value=DEFAULT_PATH, name="Path")

load_button = pn.widgets.Button(name="Load", button_type="primary")
plot_button = pn.widgets.Button(name="Plot", button_type="primary")

task_df_table = pn.widgets.Tabulator(
    pd.DataFrame(),
    height=500,
    selectable='checkbox',  # 👈 Enables checkboxes
)

status_message = pn.pane.Markdown("Hello :)", sizing_mode="stretch_width")

optimizer_checkbox_group = pn.widgets.CheckBoxGroup(
    name="Optimizers", value=[], options=[])


plot_output = pn.Column()

# Global variable to store the dataloader
dataloader = None


# ============================================================================
def load_dataloader(event=None):
    global dataloader
    global optimizer_color_map
    status_message.object = "🔄 Loading..."

    filepath_single = Path(path_widget.value) / DEFAULT_DATA_FOLDER / \
        experiment_single_optimizer_widget.value

    try:
        d_single = DataLoader.from_file(filepath_single)
        status_message.object = f"✅ Loaded Single DataLoader from `{filepath_single}`"
    except FileNotFoundError:
        dataloader = None
        task_df_table.value = pd.DataFrame()
        status_message.object = f"❌ Error: Single DataLoader not found at `{filepath_single}`"
        return

    filepath_l2co = Path(path_widget.value) / DEFAULT_L2CO_FOLDER / \
        experiment_l2co_widget.value

    try:
        d_l2co = DataLoader.from_file(filepath_l2co)
        status_message.object += f"\n✅ Loaded L2CO DataLoader from `{filepath_l2co}`"
    except FileNotFoundError:
        d_l2co = None
        status_message.object += f"\n⚠️ L2CO DataLoader not found at `{filepath_l2co}`"

    dataloader = d_single + d_l2co if d_l2co else d_single
    df = retrieve_task_df()
    task_df_table.value = df

    if hasattr(dataloader, "optimizers"):

        default_colors = plt.cm.tab10.colors
        optimizer_color_map.update({optimizer: default_colors[i % len(
            default_colors)] for i, optimizer in enumerate(
                list(dataloader.optimizers.sortby('optimizer').values))})
        opts = list(dataloader.optimizers.values)
        optimizer_checkbox_group.options = opts
        optimizer_checkbox_group.value = opts  # Optional: pre-select all


# Button event binding
load_button.on_click(load_dataloader)

# =============


def retrieve_task_df():
    global dataloader
    if dataloader is None:
        return pd.DataFrame()
    return (dataloader.task_labels.to_pandas()
            .infer_objects()
            .convert_dtypes()
            .set_index(dataloader.task_labels.itemID.values)
            .drop_duplicates()
            )


MAX_LABEL_LENGTH = 20  # Adjust as needed


def trim_label(label, max_len=MAX_LABEL_LENGTH):
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def update_plot(event=None):
    global dataloader
    global optimizer_line_style
    global optimizer_color_map

    if dataloader is None:
        plot_output.objects = [pn.pane.Markdown("### ❌ No Data Available")]
        return

    # Get selected itemIDs from tabulator
    if not task_df_table.selection:
        plot_output.objects = [pn.pane.Markdown("### ⚠️ No items selected")]
        return

    selected_indices = task_df_table.selection
    selected_itemIDs = task_df_table.value.iloc[selected_indices].index.tolist(
    )
    selected_optimizers = optimizer_checkbox_group.value

    if not selected_optimizers:
        plot_output.objects = [pn.pane.Markdown(
            "### ⚠️ No optimizers selected")]
        return

    # Filter the dataloader
    filtered_d = dataloader.select_itemIDs(
        selected_itemIDs).select_optimizer(optimizers=selected_optimizers)

    performance = filtered_d._performance(include_best=False)

    # Plotting
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])

    ax_rtd = fig.add_subplot(gs[0, 0])
    ax_pp = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")  # Hide axis for the legend area

    _rtd = rtd(performance).sortby('optimizer')
    _pp = performance_profile(performance).sortby('optimizer')

    # Plot lines and collect handles for the legend
    handles = []
    labels = []

    for o in _rtd.optimizer.values:
        line_rtd, = _rtd.sel(optimizer=o).isel(output_dim=0).plot(
            ax=ax_rtd, linewidth=2, label=o, color=optimizer_color_map[o],
            linestyle=optimizer_line_style.get(o, '-')
        )
        _pp.sel(optimizer=o).isel(output_dim=0).plot(
            ax=ax_pp, linewidth=2, label=o, color=optimizer_color_map[o],
            linestyle=optimizer_line_style.get(o, '-')
        )

        handles.append(line_rtd)
        labels.append(trim_label(o))

    # Add shared legend to the bottom axes
    ax_legend.legend(handles, labels, ncol=3, loc="center", frameon=False)

    # Format plots
    format_plot(ax=ax_rtd, xlabel="Budget", ylabel="Solved Tasks",
                title="RTD", xlim_right=1.0)
    format_plot(ax=ax_pp, xlabel="Relative Error [%]",
                ylabel="Solved Tasks", title="PP", xlim_right=100.0)

    plt.tight_layout()
    plt.close(fig)

    plot_output.objects = [pn.pane.Matplotlib(fig, tight=True)]


# Button event binding
plot_button.on_click(update_plot)


# filter_panel = pn.Column(*filter_widgets.values())
# Layout definition
left_panel = pn.Column(
    "## Experiment Path", path_widget, experiment_single_optimizer_widget, experiment_l2co_widget,
    pn.Spacer(height=10),
    status_message,
    load_button,
    plot_button,
    pn.Spacer(height=10),
    "## Optimizers", optimizer_checkbox_group,
    "## Problems", task_df_table,
)


# Full layout
layout = pn.Row(left_panel, plot_output)
layout.servable()
