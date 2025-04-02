# Standard
from pathlib import Path

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import param
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

# Default experiment and path
DEFAULT_EXPERIMENT = ""
DEFAULT_PATH = "/media/martin/T7/data"
SUBFOLDER = "single"

# Define TextInput widgets for experiment and path
experiment_widget = pn.widgets.TextInput(
    value=DEFAULT_EXPERIMENT, name="Experiment")
path_widget = pn.widgets.TextInput(value=DEFAULT_PATH, name="Path")


class DataStore(param.Parameterized):
    """Reactive container for the DataLoader."""
    dataloader = param.ClassSelector(
        class_=DataLoader, allow_None=True, default=None)

    def load_dataloader(self, path, experiment):
        """Load DataLoader based on path and experiment."""
        # filepath = Path(f"{path}{experiment}/{SUBFOLDER}")
        filepath = Path(f"{path}") / f"{experiment}"
        try:
            self.dataloader = DataLoader.from_file(filepath)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            self.dataloader = None  # Handle missing file gracefully


# Create a global instance of DataStore
d_store = DataStore()


def load_dataloader():
    """Load DataLoader based on user-specified path and experiment."""
    # filepath = Path(
    #     f"{path_widget.value}{experiment_widget.value}/{SUBFOLDER}")
    filepath = Path(f"{path_widget.value}") / f"{experiment_widget.value}"
    try:
        return DataLoader.from_file(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None  # Handle gracefully


def update_dataloader(event):
    """Callback to update DataLoader when path/experiment changes."""
    new_dataloader = load_dataloader()
    if new_dataloader is not None:
        d_store.value = new_dataloader  # Update the stored DataLoader
        print("Loaded new DataLoader")
        update_widgets()


def retrieve_task_df():
    d: DataLoader = d_store.value
    if d is None:
        return pd.DataFrame()
    return d.task_labels.to_pandas().infer_objects().convert_dtypes(

    ).set_index(d.task_labels.itemID.values).drop_duplicates()


# Watch changes in path and experiment inputs
experiment_widget.param.watch(update_dataloader, 'value')
path_widget.param.watch(update_dataloader, 'value')

# Load initial DataLoader
d_store.value = load_dataloader()

# Define widgets for selecting iteration range and optimizers
iteration_widget = pn.widgets.RangeSlider(
    name="Iterations", start=0., end=1., value=(0, 1))
optimizer_widget = pn.widgets.CheckBoxGroup(name="Optimizers", inline=True)

# Define Tabulator widget for displaying tasks
itemID_widget = pn.widgets.Tabulator(
    width=1200, height=300, selectable="checkbox",
)

optimizer_line_style = {'sbs': '--'}
optimizer_color_map = {'sbs': 'black'}

# Define filter widgets container
filter_widgets = {}


def create_filter_widgets(task_df):
    """Dynamically create MultiSelect widgets for filtering."""
    global filter_widgets
    filter_widgets.clear()

    widgets = []
    for col in task_df.columns:
        unique_values = sorted(task_df[col].dropna().unique())
        widget = pn.widgets.MultiSelect(
            name=col, options=list(unique_values), size=5
        )
        filter_widgets[col] = widget
        widgets.append(widget)

    return pn.Column(*widgets)


def update_widgets():
    global optimizer_color_map
    global filter_panel
    d: DataLoader = d_store.value
    if d is None:
        return

    # Update iteration range
    iteration_widget.end = 1.0  # d.n_iterations
    iteration_widget.value = (0., 1.0)

    # Update optimizer selection
    optimizer_widget.options = list(d.optimizers.values)
    optimizer_widget.value = list(d.optimizers.values)

    # Get the default color cycle from Matplotlib
    # You can use other color cycles here if needed
    default_colors = plt.cm.tab10.colors
    optimizer_color_map.update({optimizer: default_colors[i % len(
        default_colors)] for i, optimizer in enumerate(
            list(d.optimizers.values))})

    task_df = retrieve_task_df()
    filter_panel.objects = [create_filter_widgets(task_df)]

    global filter_widgets
    filter_widgets.clear()
    for col in task_df.columns:
        unique_values = sorted(task_df[col].dropna().unique())
        filter_widgets[col] = pn.widgets.MultiSelect(
            name=col, options=list(unique_values), size=5)

    itemID_widget.value = task_df


filter_panel = pn.Column()


@pn.depends(*[w.param.value for w in filter_widgets.values()])
def filter_task_df():
    """Apply filtering based on selected values."""
    df = retrieve_task_df()
    if df.empty:
        return df

    for col, widget in filter_widgets.items():
        if widget.value:  # Apply filter only if there are selected values
            df = df[df[col].isin(widget.value)]

    return df


itemID_widget.value = filter_task_df()


# Update table whenever filters change
@pn.depends(*[w.param.value for w in filter_widgets.values()])
def update_table():
    return filter_task_df()


@pn.depends(
    selection=itemID_widget.param.selection,
    iterations=iteration_widget.param.value,
    optimizers=optimizer_widget.param.value
)
def update_plot(selection, iterations, optimizers):
    """Update performance plot based on selected items."""
    d = d_store.value
    if d is None or not selection:
        return pn.pane.Markdown("### No Data Available")

    # Filter DataLoader based on selections
    selected_itemIDs = itemID_widget.value.iloc[selection].index
    print(selected_itemIDs)
    filtered_d = d.select_itemIDs(selected_itemIDs).select_t(
        t_start=iterations[0], t=iterations[1]
    ).select_optimizer(optimizers=optimizers)

    performance = filtered_d._performance(include_best=True)

    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharey=True)

    _rtd = rtd(performance)
    _pp = performance_profile(performance)

    for o in _rtd.optimizer.values:
        _rtd.sel(optimizer=o).isel(output_dim=0).plot(
            ax=axes[0], linewidth=2, label=o, color=optimizer_color_map[o],
            linestyle=optimizer_line_style.get(o, '-')
        )
        _pp.sel(optimizer=o).isel(output_dim=0).plot(
            ax=axes[1], linewidth=2, label=o, color=optimizer_color_map[o],
            linestyle=optimizer_line_style.get(o, '-')
        )

    axes[1].legend()
    format_plot(ax=axes[0], xlabel="Budget",
                ylabel="Solved Tasks", title="RTD",
                xlim_right=1.0)
    format_plot(ax=axes[1], xlabel="Relative Error [%]",
                ylabel="Solved Tasks", title="PP",
                xlim_right=100.0)

    plt.close(fig)
    return pn.pane.Matplotlib(fig, tight=True)


filter_panel = pn.Column(*filter_widgets.values())
# Layout definition
left_panel = pn.Column(
    "## Experiment Path", experiment_widget, path_widget,
    "## Iterations", iteration_widget,
    "## Optimizers", optimizer_widget,
    "## Filters", filter_panel,
    "## Problems", itemID_widget, width=800
)

right_panel = pn.Column(update_plot)

# Full layout
layout = pn.Row(left_panel, right_panel)
layout.servable()
