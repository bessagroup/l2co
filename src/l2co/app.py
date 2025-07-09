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
from l2co._src.naming import (BUDGET, OUTPUT_DIM, OUTPUT_MEAN, OUTPUT_MIN,
                              OUTPUT_STD, PROBLEM, SCHEDULE, TASK_ID)
from l2co.plotting import format_plot

# =============================================================================

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
experiment_single_optimizer_widget = pn.widgets.Select(
    name="Data single optimizer",
    options=[],
)

experiment_l2co_widget = pn.widgets.MultiSelect(
    name="Data L2CO model",
    options=[],
    size=6  # Adjust height
)

path_widget = pn.widgets.TextInput(value=DEFAULT_PATH, name="Path")

load_button = pn.widgets.Button(name="Load", button_type="primary")
plot_button = pn.widgets.Button(name="Plot", button_type="primary")

task_df_table = pn.widgets.Tabulator(
    pd.DataFrame(),
    height=500,
    selectable='checkbox',
)

task_individual_df_table = pn.widgets.Tabulator(
    pd.DataFrame(),
    height=500,
    selectable='checkbox',
)

status_message = pn.pane.Markdown("Hello :)", sizing_mode="stretch_width")

optimizer_checkbox_group = pn.widgets.CheckBoxGroup(
    name="Optimizers", value=[], options=[])

exclude_optimizer_checkbox_group = pn.widgets.CheckBoxGroup(
    name="Optimizers", value=[], options=[])

plot_output = pn.Column()

# Spreadsheet tab widgets
spreadsheet_button = pn.widgets.Button(
    name="Generate Spreadsheet", button_type="success")
spreadsheet_output = pn.Column()

# Optimizer choices tab widgets
opt_choices_button = pn.widgets.Button(
    name="Generate SBS Choices", button_type="success")
opt_choices_output = pn.Column()

# L2CO choices tab widgets
l2co_choices_button = pn.widgets.Button(
    name="Show L2CO choices", button_type="success")
l2co_choices_output = pn.Column()


# Single plotter tab widgets
single_plot_button = pn.widgets.Button(
    name="Plot Single", button_type="success")

single_plot_output = pn.Column()

# Global variable to store the dataloader
dataloader = None

# =============================================================================


def update_single_optimizer_options(event=None):
    base_path = Path(path_widget.value) / DEFAULT_DATA_FOLDER
    if not base_path.exists():
        experiment_single_optimizer_widget.options = []
        return
    folders = [f.name for f in base_path.iterdir() if f.is_dir()]
    experiment_single_optimizer_widget.options = folders


path_widget.param.watch(update_single_optimizer_options, 'value')


def update_l2co_files(event=None):
    base_path = Path(path_widget.value) / DEFAULT_L2CO_FOLDER
    if not experiment_single_optimizer_widget.value:
        experiment_l2co_widget.options = []
        return
    folder = base_path / experiment_single_optimizer_widget.value
    if not folder.exists():
        experiment_l2co_widget.options = []
        return
    nc_files = [f.name for f in folder.glob("*.odata")]
    experiment_l2co_widget.options = nc_files


experiment_single_optimizer_widget.param.watch(update_l2co_files, 'value')


def load_dataloader(event=None):
    """
    Load the DataLoader for the selected experiment.

    Parameters
    ----------
    event : optional
        Event triggered by the UI interaction, by default None.

    Returns
    -------
    None
    """
    global dataloader
    global optimizer_color_map
    status_message.object = "🔄 Loading..."

    filepath_single = Path(path_widget.value) / DEFAULT_DATA_FOLDER / \
        experiment_single_optimizer_widget.value

    try:
        d_single = DataLoader.from_file(filepath_single)
        status_message.object = (
            f"✅ Loaded Single DataLoader from `{filepath_single}`\n")
    except FileNotFoundError:
        dataloader = None
        task_df_table.value = pd.DataFrame()
        status_message.object = (
            f"❌ Error: Single DataLoader not found at `{filepath_single}`\n")
        return

    d_l2co = None
    base_path = Path(path_widget.value) / DEFAULT_L2CO_FOLDER / \
        experiment_single_optimizer_widget.value
    for filename in experiment_l2co_widget.value:
        try:
            this_d = DataLoader.from_file(path=base_path,
                                          data_filename=filename)
            d_l2co = this_d if d_l2co is None else d_l2co + this_d
            status_message.object += (
                f"\n✅ Loaded L2CO DataLoader from `{filename}`\n")
        except FileNotFoundError:
            status_message.object += f"\n⚠️ Could not load `{filename}`\n"

    dataloader = d_single + d_l2co if d_l2co else d_single
    df = retrieve_task_df()
    df_individual = retrieve_task_df_individual()
    task_df_table.value = df
    task_individual_df_table.value = df_individual

    if hasattr(dataloader, "optimizers"):

        default_colors = plt.cm.tab10.colors
        optimizer_color_map.update({optimizer: default_colors[i % len(
            default_colors)] for i, optimizer in enumerate(
                list(dataloader.optimizers.sortby(SCHEDULE).values))})
        opts = list(dataloader.optimizers.values)
        optimizer_checkbox_group.options = opts
        exclude_optimizer_checkbox_group.options = opts
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
            .set_index(dataloader.task_labels[TASK_ID].values)
            .drop_duplicates()
            )


def retrieve_task_df_individual():
    global dataloader
    if dataloader is None:
        return pd.DataFrame()
    return (dataloader.task_labels.to_pandas()
            .infer_objects()
            .convert_dtypes()
            )


def generate_spreadsheet(event=None):
    """
    Generate a spreadsheet summarizing the optimization results.

    Returns
    -------
    None
    """
    global dataloader

    if dataloader is None:
        spreadsheet_output.objects = [
            pn.pane.Markdown("### ❌ No Data Available")]
        return

    try:
        styled_df = dataloader.to_spreadsheet()
        spreadsheet_output.objects = [pn.pane.DataFrame(
            styled_df, width=1000, height=600, sizing_mode="stretch_width")]
    except Exception as e:
        spreadsheet_output.objects = [
            pn.pane.Markdown(f"### ❌ Error:\n```{e}```")]


spreadsheet_button.on_click(generate_spreadsheet)


def generate_opt_choices(event=None):
    global dataloader

    if dataloader is None:
        opt_choices_output.objects = [
            pn.pane.Markdown("### ❌ No Data Available")]
        return

    exclude_optimizers = exclude_optimizer_checkbox_group.value

    if not exclude_optimizers:
        exclude_optimizers = None

    try:
        opt_choices_df = dataloader.display_sbs_table(
            exclude_optimizer=exclude_optimizers)

        opt_choices_output.objects = [pn.widgets.Tabulator(
            opt_choices_df,
            width=1000,
            height=600,
            sizing_mode="stretch_width"
        )]

    except Exception as e:
        opt_choices_output.objects = [
            pn.pane.Markdown(f"### ❌ Error:\n```{e}```")]


opt_choices_button.on_click(generate_opt_choices)


def generate_l2co_choices(event=None):
    global dataloader

    if dataloader is None:
        l2co_choices_output.objects = [
            pn.pane.Markdown("### ❌ No Data Available")]
        return

    try:
        l2co_choices_df = dataloader.display_l2co_schedules_table()

        l2co_choices_output.objects = [pn.widgets.Tabulator(
            l2co_choices_df,
            width=1000,
            height=600,
            sizing_mode="stretch_width"
        )]

    except Exception as e:
        l2co_choices_output.objects = [
            pn.pane.Markdown(f"### ❌ Error:\n```{e}```")]


l2co_choices_button.on_click(generate_l2co_choices)


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
    exclude_optimizers = exclude_optimizer_checkbox_group.value

    if not exclude_optimizers:
        exclude_optimizers = None

    if not selected_optimizers:
        plot_output.objects = [pn.pane.Markdown(
            "### ⚠️ No optimizers selected")]
        return

    # Filter the dataloader
    filtered_d = dataloader.select_itemIDs(
        selected_itemIDs).select_optimizer(optimizers=selected_optimizers)

    performance = filtered_d._performance(
        include_best=True,
        exclude_optimizer=exclude_optimizers)

    # Plotting
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])

    ax_rtd = fig.add_subplot(gs[0, 0])
    ax_pp = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")  # Hide axis for the legend area

    _rtd = rtd(performance).sortby(SCHEDULE)
    _pp = performance_profile(performance).sortby(SCHEDULE)

    # Plot lines and collect handles for the legend
    handles = []
    labels = []

    for o in _rtd[SCHEDULE].values:
        line_rtd, = _rtd.sel({SCHEDULE: o}).isel({OUTPUT_DIM: 0}).plot(
            ax=ax_rtd, linewidth=2, label=o, color=optimizer_color_map[o],
            linestyle=optimizer_line_style.get(o, '-')
        )
        _pp.sel({SCHEDULE: o}).isel({OUTPUT_DIM: 0}).plot(
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


# d.data.isel(problem=0, output_dim=0)


def update_single_plot(event=None):
    global dataloader
    if dataloader is None:
        single_plot_output.objects = [
            pn.pane.Markdown("### ❌ No Data Available")]
        return

    problem = task_individual_df_table.selection[0]
    log_scale = True
    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_2 = fig.add_subplot(gs[0, 1], sharey=ax_1)
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")

    handles = []
    labels = []

    for o in dataloader.optimizers.values:
        line, = dataloader.data.sel({SCHEDULE: 0}).isel({
            PROBLEM: problem, OUTPUT_DIM: 0})[OUTPUT_MIN].plot(ax=ax_1, label=o)

        m = dataloader.data.sel({SCHEDULE: 0}).isel({
            PROBLEM: problem, OUTPUT_DIM: 0})[OUTPUT_MEAN]
        std = dataloader.data.sel({SCHEDULE: 0}).isel({
            PROBLEM: problem, OUTPUT_DIM: 0})[OUTPUT_STD]

        dataloader.data.sel({SCHEDULE: 0}).isel({
            PROBLEM: problem, OUTPUT_DIM: 0})[OUTPUT_MEAN].plot(ax=ax_2, label=o)

        # 95% confidence interval
        ax_2.fill_between(m.budget, m - 2 * std, m + 2 * std, alpha=0.2)

        handles.append(line)
        labels.append(trim_label(o))

    ax_legend.legend(handles, labels, ncol=3, loc="center", frameon=False)

    if log_scale:
        ax_1.set_yscale('log')
        ax_2.set_yscale('log')
    ax_1.set_ylabel('Objective value')

    ax_1.set_title(f'Minimum loss: Problem {problem}')
    ax_2.set_title(f'Mean + std loss: Problem {problem}')

    plt.tight_layout()
    plt.close(fig)

    single_plot_output.objects = [pn.pane.Matplotlib(fig, tight=True)]


single_plot_button.on_click(update_single_plot)

update_single_optimizer_options()
update_l2co_files()


# First tab: Load data
load_tab = pn.Column(
    "## Experiment Path",
    path_widget,
    experiment_single_optimizer_widget,
    experiment_l2co_widget,
    pn.Spacer(height=10),
    status_message,
    load_button,
    "### Select Optimizers",
    optimizer_checkbox_group,
    "### Select Problems",
    task_df_table,
)

# Second tab: Selection & Plotting
plot_tab = pn.Column(
    "## Plotting",
    "### Exclude Optimizers for SBS",
    exclude_optimizer_checkbox_group,
    plot_button,
    pn.Spacer(height=10),
    plot_output
)

# Third tab: Spreadsheet view
spreadsheet_tab = pn.Column(
    "## Export to Spreadsheet",
    spreadsheet_button,
    spreadsheet_output,
)

opt_choices_tab = pn.Column(
    '## Optimizer Choices',
    exclude_optimizer_checkbox_group,
    opt_choices_button,
    opt_choices_output,
)

l2co_choices_tab = pn.Column(
    '## L2CO Choices',
    l2co_choices_button,
    l2co_choices_output,
)

single_plot_tab = pn.Column(
    '## Single Plot',
    single_plot_button,
    task_individual_df_table,
    single_plot_output,

)

tabs = pn.Tabs(
    ("📂 Load Data", load_tab),
    ("📊 Performance Plots", plot_tab),
    ("🧾 Spreadsheet", spreadsheet_tab),
    ("🗂️ Optimizer Choices", opt_choices_tab),
    ("🗂️ L2CO Choices", l2co_choices_tab),
    ("📊 Single run", single_plot_tab),
    dynamic=True,
)


tabs.servable()
