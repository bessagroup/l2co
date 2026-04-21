import matplotlib.pyplot as plt
import bessaplots

plt.style.use(["dissertation"])

# =============================================================================
from pathlib import Path
import l2co
from l2co.naming import PROBLEM, REALIZATION, SCHEDULE, TASK_ID
import matplotlib as mpl


rootpath = Path("/Volumes/T7/data/wcci/")

dataset = "bbob"

format = "pgf"
paper_size = "tudelft"
n_side_by_side = 3
height = 1.0
sharex = True
sharey = True

d = l2co.DataLoader.from_file(rootpath / dataset / "entire_dataset")

l2co.plotting.bbob.ertd_per_dimension(
    d=d,
    save_dir="./plots",
    paper_size=paper_size,
    n_side_by_side=n_side_by_side,
    format=format,
    height=height,
    sharey=sharey,
    sharex=sharex,
)

l2co.plotting.bbob.ertd_per_category(
    d=d,
    save_dir="./plots",
    paper_size=paper_size,
    n_side_by_side=n_side_by_side,
    format=format,
    height=height,
    sharey=sharey,
    sharex=False,
)

fig, ax = l2co.plotting.plot_ertd_profile(
    d, show_legend=False, xlim=(10**-1, 10**5)
)

bessaplots.savefig(
    fig=fig,
    path="./plots/rtd_all.pdf",
    n_side_by_side=n_side_by_side,
    span_columns=True,
    height=height,
    paper_size=paper_size,
    format=format,
)
