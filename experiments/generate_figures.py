from pathlib import Path
import l2co
from l2co.naming import PROBLEM, REALIZATION, SCHEDULE, TASK_ID
import bessaplots


rootpath = Path("/Volumes/T7/data/wcci/")

dataset = "bbob"


d = l2co.DataLoader.from_file(rootpath / dataset / "entire_dataset")

l2co.plotting.bbob.ertd_per_dimension(d=d, save_dir="./plots")

l2co.plotting.bbob.ertd_per_category(d=d, save_dir="./plots")

fig, ax = l2co.plotting.plot_ertd_profile(
    d, show_legend=False, xlim=(10**-1, 10**5)
)

bessaplots.savefig(
    fig=fig,
    path="./plots/rtd_gaussian.pdf",
    n_side_by_side=4,
    span_columns=True,
    height=1.0,
)
