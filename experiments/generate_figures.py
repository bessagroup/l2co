import matplotlib.pyplot as plt
import bessaplots

plt.style.use(["dissertation"])

# =============================================================================
from pathlib import Path
import l2co
from l2co.naming import PROBLEM, REALIZATION, SCHEDULE, TASK_ID
import matplotlib as mpl


rootpath = Path("/Volumes/T7/data/wcci/")
# plotsdir = Path.cwd() / "plots_pdf"
plotsdir = (
    Path(
        "/Users/martin/Documents/GitHub/dissertation/-main_matter/learning_to_choose_optimizers/media"
    )
    / "rtd"
)

stage0_model_path = "10"

format = "pgf"
paper_size = "tudelft"
# n_side_by_side = 3
height = 1.0
# sharex = True
# sharey = True

schema = {
    "dimensionality": "continuous",
    "separable": "boolean",
    "unimodal": "boolean",
}

stage0_model: l2co.Stage0Model = l2co.Stage0Loader.load(
    rootpath / stage0_model_path
)

# RTD in-distribution
# =============================================================================

dataset = "bbob"
(plotsdir / dataset).mkdir(exist_ok=True, parents=True)

d = l2co.DataLoader.from_file(rootpath / dataset / "entire_dataset")


d_opts = d.select_optimizer(
    ["sepcmaes", "pso", "adam_learning_rate=0.01", "lbfgs"]
)

o = d_opts.to_params(
    t=0.01,
    t_end=10000,
    schema=schema,
    feature_processor_task=stage0_model.feature_processor_task,
)
o.x_features_task = o.x_features_task.fillna(0)

y_pred = stage0_model.evaluate(data=o)

choices = (
    y_pred.unstack("sample")
    .isel({SCHEDULE: 0}, drop=True)
    .stack({PROBLEM: (TASK_ID, REALIZATION)}, create_index=True)
)
d = d.create_strategy(choices=choices, name="stage0")

l2co.plotting.bbob.ertd_per_dimension(
    d=d,
    save_dir=plotsdir / dataset,
    paper_size=paper_size,
    n_side_by_side=3,
    format=format,
    height=height,
    sharey=True,
    sharex=False,
)

l2co.plotting.bbob.ertd_per_category(
    d=d,
    save_dir=plotsdir / dataset,
    paper_size=paper_size,
    n_side_by_side=3,
    format=format,
    height=height,
    sharey=True,
    sharex=False,
)

l2co.plotting.bbob.ertd_per_function(
    d=d,
    save_dir=plotsdir / dataset,
    paper_size=paper_size,
    n_side_by_side=3,
    format=format,
    height=height,
    sharey=True,
    sharex=False,
)

fig, ax = l2co.plotting.plot_ertd_profile(
    d, show_legend=False, xlim=(10**-1, 10**5)
)
fig_grid = bessaplots.FigureGrid(
    n_figures=2,
    paper_size=paper_size,
    span_columns=True,
    height=height,
    sharey=False,
    sharex=False,
)

fig_grid.save(
    fig=fig,
    path=plotsdir / dataset / "rtd_all.pdf",
    format=format,
)

# =============================================================================


# RTD distribution-shift
# =============================================================================

dataset = "bbob2"
(plotsdir / dataset).mkdir(exist_ok=True, parents=True)

d = l2co.DataLoader.from_file(rootpath / dataset / "entire_dataset")

d_opts = d.select_optimizer(
    ["sepcmaes", "pso", "adam_learning_rate=0.01", "lbfgs"]
)

o = d_opts.to_params(
    t=0.01,
    t_end=10000,
    schema=schema,
    feature_processor_task=stage0_model.feature_processor_task,
)
o.x_features_task = o.x_features_task.fillna(0)

y_pred = stage0_model.evaluate(data=o)

choices = (
    y_pred.unstack("sample")
    .isel({SCHEDULE: 0}, drop=True)
    .stack({PROBLEM: (TASK_ID, REALIZATION)}, create_index=True)
)
d = d.create_strategy(choices=choices, name="stage0")

l2co.plotting.bbob.ertd_per_dimension(
    d=d,
    save_dir=plotsdir / dataset,
    paper_size=paper_size,
    n_side_by_side=3,
    format=format,
    height=height,
    sharey=True,
    sharex=False,
    dims=[2, 4, 7, 15, 25, 50],
)

fig, ax = l2co.plotting.plot_ertd_profile(
    d, show_legend=False, xlim=(10**-1, 10**5)
)

fig_grid = bessaplots.FigureGrid(
    n_figures=2,
    paper_size=paper_size,
    span_columns=True,
    height=height,
    sharey=False,
    sharex=False,
)

fig_grid.save(
    fig=fig,
    path=plotsdir / dataset / "rtd_all.pdf",
    format=format,
)

# =============================================================================


# CEC 2005
# =============================================================================

dataset = "cec2005"
(plotsdir / dataset).mkdir(exist_ok=True, parents=True)

d = l2co.DataLoader.from_file(rootpath / dataset / "entire_dataset")


d_opts = d.select_optimizer(
    ["sepcmaes", "pso", "adam_learning_rate=0.01", "lbfgs"]
)

o = d_opts.to_params(
    t=0.01,
    t_end=10000,
    schema=schema,
    feature_processor_task=stage0_model.feature_processor_task,
)
o.x_features_task = o.x_features_task.fillna(0)

y_pred = stage0_model.evaluate(data=o)

choices = (
    y_pred.unstack("sample")
    .isel({SCHEDULE: 0}, drop=True)
    .stack({PROBLEM: (TASK_ID, REALIZATION)}, create_index=True)
)
d = d.create_strategy(choices=choices, name="stage0")

l2co.plotting.cec2005.ertd_per_dimension(
    d=d,
    save_dir=plotsdir / dataset,
    paper_size=paper_size,
    n_side_by_side=3,
    format=format,
    height=height,
    sharey=True,
    sharex=False,
)

l2co.plotting.cec2005.ertd_per_function(
    d=d,
    save_dir=plotsdir / dataset,
    paper_size=paper_size,
    n_side_by_side=3,
    format=format,
    height=height,
    sharey=True,
    sharex=False,
)

fig, ax = l2co.plotting.plot_ertd_profile(
    d, show_legend=False, xlim=(10**-1, 10**5)
)


fig_grid = bessaplots.FigureGrid(
    n_figures=2,
    paper_size=paper_size,
    span_columns=True,
    height=height,
    sharey=False,
    sharex=False,
)

fig_grid.save(
    fig=fig,
    path=plotsdir / dataset / "rtd_all.pdf",
    format=format,
)

# =============================================================================
